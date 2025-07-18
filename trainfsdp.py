# -*- coding: utf-8 -*-
import contextlib
import json
import os
import torch
from zeus.monitor import ZeusMonitor
import torch.distributed as dist
from torch.distributed.fsdp.flat_param import FlatParamHandle
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from utils.argparser_utils import parse_args, get_dtype_from_str
from utils.train_utils import get_profiler_path, print_metrics
from utils.data_loader import get_dataset, split_microbatches
from utils.global_state import (
    get_compute_stream,
    init_split_state,
    set_split_state,
    configure_gradient_accumulation,
)
from utils.model import model_init
from utils.profile import get_profiler_context
from utils.patch import (
    _post_backward_hook_sync,
    _post_reduce_grad_callback_patch,
    prepare_gradient_for_optim_patch,
    _post_backward_hook_patch,
    shard_patch,
    unshard_patch,
    reshard_patch,
    enable_gradient_accumulation,
)
from utils.comm import (
    get_global_rank,
    dist_init,
    get_local_rank,
    is_leader,
)
from utils.profile import print_memory_stats
from utils.logger import get_logger, init_logger
from models.hub import get_all_layers, is_vision_model


def get_layer_output_shape_ga(args, model):
    if is_vision_model(model._model_name):
        size = (1, 3, args.image_size, args.image_size)
        input_data = torch.randn(size, device="cuda")
    else:
        input_data = torch.randint(
            low=0, high=args.vocab_size, size=(1, args.seq_length), device="cuda"
        )
    microbatch_size = args.local_batch_size // args.microbatches
    layer = get_all_layers(model)[0]
    x = layer(
        input_data,
        is_first_microbatch=True,
        is_last_microbatch=True,
        skip_reshard=True,
        in_backwards=False,
    )
    shape = x.shape
    # update to use microbatch size
    output_shape = torch.Size([microbatch_size] + list(shape[1:]))
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output_shape


def train_ga(args, model, optimizer, data_iter, zeus_monitor=None):
    """
    Training logic
    """
    # torch.autograd.set_detect_anomaly(True)
    logger = get_logger()

    if zeus_monitor and is_leader():
        zeus_monitor.begin_window("training")
    
    # start and end event for all iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # set training mode
    model.train()

    # setup vars
    warmup_iterations = args.warmup_iterations
    iterations = args.iterations
    microbatches = args.microbatches
    detailed_profiler_trace = args.detailed_profiler_trace
    total_iterations = warmup_iterations + iterations
    print_mem_step = max(warmup_iterations - 3, 0)
    profiler_path = get_profiler_path(args)
    local_rank = get_local_rank()
    device = torch.device("cuda", local_rank)
    # iteration end and start events
    iteration_start = torch.cuda.Event(enable_timing=True)
    iteration_end = torch.cuda.Event(enable_timing=True)

    # use to offload computation graph with autograd context
    if args.offload_activations:
        offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=False)
    else:
        offload_ctx = contextlib.nullcontext()

    ### ---------------------------------------------------------->>> setup activation buffers
    configure_gradient_accumulation(model, microbatches)
    model_layers = get_all_layers(model)
    # initialize ga specific split_state
    num_layers = len(model_layers)

    if args.use_activation_buffers:
        # make sure we don't autograd save on cpu context
        offload_ctx = contextlib.nullcontext()
        num_buffers = num_layers * microbatches
        buffer_shape = get_layer_output_shape_ga(args, model)

        # buffer for all activations
        cpu_buffers = [
            torch.empty(buffer_shape, pin_memory=True) for _ in range(num_buffers)
        ]
        # cpu buffers for gradients for next layer
        cpu_grad_buffers = [
            torch.empty(buffer_shape, pin_memory=True) for _ in range(microbatches)
        ]
        # forwards buffers -> to avoid CPU->GPU operations
        if args.use_forwards_gpu_buffer:
            gpu_forwards_buffers = [
                torch.empty(buffer_shape, device=device) for _ in range(microbatches)
            ]

        # stores current microbatch activation
        gpu_activation_buffer = torch.empty(buffer_shape, device=device)
        # stores gradients for next layer microbatch
        gpu_grad_buffer = torch.empty(buffer_shape, device=device)

        # initialize mem copy stream
        mem_copy_stream = torch.cuda.Stream()
        # event to synch with compute (used with no prefetch logic)
        copy_done_event = torch.cuda.Event()

        # events for synchronizing prefetch (used with prefetch logic)
        if args.use_prefetch_backwards:
            recompute_activations_prefetch_event = torch.cuda.Event()
            gradient_prefetch_event = torch.cuda.Event()

    ### ----------------------------------------------------------<<< setup activation buffers

    ### -------------------------------------------------------------------------->>> training loop

    for step_idx in range(total_iterations):
        if zeus_monitor and is_leader():
            zeus_monitor.begin_window(f"iteration_{step_idx}")
        # synch all processes
        if step_idx == warmup_iterations:
            dist.barrier()
            start_event.record()

        # by default there are 5 warmup iterations, using 1 profiling iterations by default
        profile_step = (
            step_idx
            in range(warmup_iterations - args.profiling_iterations, warmup_iterations)
            and not args.skip_profile
        )
        # set context for profiling
        if profile_step:
            out_dir = os.path.join(
                profiler_path,
                f"iteration_{step_idx}",
            )
            profiler_ctx = get_profiler_context(
                out_dir=out_dir, detailed_trace=detailed_profiler_trace, unique_gpus_only=True
            )
        else:
            profiler_ctx = contextlib.nullcontext()

        # get data for training and split into microbatches
        input_ids, labels = next(data_iter)
        input_ids_microbatches, _ = split_microbatches(input_ids, labels, microbatches)

        # iteration start
        iteration_start.record()

        with profiler_ctx:
            # memory print
            if step_idx == print_mem_step:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                print_memory_stats("pre-forward", all_ranks=False)
                if is_leader():
                    torch.cuda.memory._record_memory_history(max_entries=100000)

            # iteration 1 onwards, set unshard patch
            if step_idx > 0:
                for layer in model_layers:
                    layer._handle._needs_pre_forward_unshard = True

            # store activations during forwards pass
            activations = []

            ### -------------------------------------------------------------------------->>> forwards pass

            with torch.autocast(
                device_type="cuda", dtype=args.autocast_dtype
            ), offload_ctx:

                # loop over all layers
                for li, layer in enumerate(model_layers):

                    # don't run forwards for last layer ( ? )
                    last_layer = li == num_layers - 1
                    if last_layer:
                        break

                    logger.debug(f"+ f layer {li}")

                    # store microbatch activations for current layer
                    activations.append([])

                    # loop over all microbatches
                    for i in range(microbatches):
                        logger.debug(f" -- f microbatch {i}")
                        first_microbatch = i == 0
                        last_microbatch = i == microbatches - 1

                        # for the first layer use input ids as input
                        if li == 0:
                            layer_input = input_ids_microbatches[i]

                        # otherwise, use activations from previous layers
                        else:
                            if args.use_activation_buffers:
                                # if using forwards buffer, fetch activation for this microbatch from GPU buffer
                                if args.use_forwards_gpu_buffer:
                                    layer_input = gpu_forwards_buffers[i]
                                # otherwise, fetch the previous layer activation from CPU buffers
                                else:
                                    layer_input = cpu_buffers[
                                        (li - 1) * microbatches + i
                                    ]
                            else:
                                # otherwise, fetch directly from activations storage
                                layer_input = activations[-2][i]

                        # --------> forwards for microbatch
                        # Note: if forwards GPU buffer is not used, memcpy from CPU->GPU implicitly
                        layer_output = layer(
                            layer_input,
                            is_first_microbatch=first_microbatch,
                            is_last_microbatch=last_microbatch,
                            skip_reshard=last_layer,
                            in_backwards=False,
                        ).detach()

                        # fill up buffers after forwards for this microbatch
                        if args.use_activation_buffers:

                            # save activation to GPU buffer (forwards)
                            if args.use_forwards_gpu_buffer:
                                gpu_forwards_buffers[i] = layer_output

                            # use memcpy stream to offload activations to CPU buffers
                            with torch.cuda.stream(mem_copy_stream):
                                buffer_idx = li * microbatches + i
                                cpu_buffers[buffer_idx].copy_(
                                    layer_output.data, non_blocking=True
                                )
                        # otherwise fill up activations storage
                        else:
                            activations[-1].append(layer_output)

            ### --------------------------------------------------------------------------<<< forwards pass

            # memory print
            if step_idx == print_mem_step:
                print_memory_stats("post-forward", all_ranks=False)

            # get compute stream (for synchronization)
            compute_stream = get_compute_stream()

            if args.use_activation_buffers and args.use_prefetch_backwards:
                # prefetch last microbatch of second last layer for starting recomputing activations
                with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                    gpu_activation_buffer.copy_(
                        cpu_buffers[(num_layers - 2) * microbatches + microbatches - 1],
                        non_blocking=True,
                    )
                    # record to synch
                    recompute_activations_prefetch_event.record(stream=mem_copy_stream)

            ### -------------------------------------------------------------------------->>> backwards pass

            # loop over layers (starting from last)
            for li in range(num_layers - 1, -1, -1):

                last_layer = li == num_layers - 1
                layer = model_layers[li]
                logger.debug(f"+ b layer {li}")  # start backwards for layer li

                # loop over all microbatches (starting from last)
                for i in range(microbatches - 1, -1, -1):

                    first_microbatch = i == microbatches - 1
                    last_microbatch = i == 0
                    logger.debug(f" -- b microbatch {i} - forwards")

                    ### -------------------------------------------------------------------------->>> recompute activations for microbatch

                    with torch.autocast(
                        device_type="cuda", dtype=args.autocast_dtype
                    ), offload_ctx:

                        # for the first layer use input ids as input
                        if li == 0:
                            layer_input = input_ids_microbatches[i]
                            # [repeated:] -> also need this prefetch for layer 0
                            if (
                                args.use_activation_buffers
                                and args.use_prefetch_backwards
                            ):
                                # prefetch gradients for backwards
                                with torch.cuda.stream(
                                    mem_copy_stream
                                ), torch.no_grad():
                                    gpu_grad_buffer.copy_(
                                        cpu_grad_buffers[i], non_blocking=True
                                    )
                                    gradient_prefetch_event.record(
                                        stream=mem_copy_stream
                                    )

                        # otherwise, use activations from previous layers
                        else:
                            # Use activation buffers created
                            if args.use_activation_buffers:
                                # starting from the second last layer, prefetch the microbatchgradient for backwards
                                if args.use_prefetch_backwards:
                                    # use memcpy stream, no_grad to avoid copying errors
                                    with torch.cuda.stream(
                                        mem_copy_stream
                                    ), torch.no_grad():
                                        gpu_grad_buffer.copy_(
                                            cpu_grad_buffers[i], non_blocking=True
                                        )
                                        gradient_prefetch_event.record(
                                            stream=mem_copy_stream
                                        )

                                    # make sure the input activation is fetched for recompute
                                    recompute_activations_prefetch_event.wait(
                                        stream=compute_stream
                                    )

                                # otherwise don't use prefetching logic
                                else:
                                    # load activation from previous layer microbatch
                                    buffer_idx_curr = (li - 1) * microbatches + i
                                    # use memcpy stream, no_grad to avoid copying errors
                                    with torch.cuda.stream(
                                        mem_copy_stream
                                    ), torch.no_grad():
                                        gpu_activation_buffer.copy_(
                                            cpu_buffers[buffer_idx_curr],
                                            non_blocking=True,
                                        )
                                        copy_done_event.record(stream=mem_copy_stream)
                                    # make sure copying is finished
                                    copy_done_event.wait(stream=compute_stream)

                                # set input for recomputaion, set requires grad True for backwards
                                gpu_activation_buffer.requires_grad = True
                                layer_input = gpu_activation_buffer

                            # otherwise, use activations storage. Note: these are already on GPU
                            else:
                                activations[li - 1][i].requires_grad = True
                                layer_input = activations[li - 1][i]

                        # --------> forwards for recomputing activations for microbatch
                        outputs = layer(
                            layer_input,
                            is_first_microbatch=first_microbatch,
                            is_last_microbatch=last_microbatch,
                            skip_reshard=last_layer,
                            in_backwards=True,
                        )

                        ############### ---> synchronize compute stream with CPU thread

                        # this makes sure that unused memory is cleaned up
                        if not args.skip_compute_sync:
                            compute_stream.synchronize()
                        # reduced reserved memory! less cuda malloc retries!

                        ############### ---< synchronize compute stream with CPU thread

                        ### --------------------------------------------------------------------------<<< recompute activations for microbatch

                        ### -------------------------------------------------------------------------->>> backwards for microbatch
                        logger.debug(f" -- b microbatch {i} - backwards")

                        # for the last layer, no need for gradient
                        if li == num_layers - 1:
                            outputs = outputs.sum()
                            gradient = None
                        # otherwise, get gradient from activations recomputed
                        else:
                            # fetch from activation buffers
                            if args.use_activation_buffers:
                                if args.use_prefetch_backwards:
                                    # make sure gradient is prefetched
                                    gradient_prefetch_event.wait(stream=compute_stream)

                                # otherwise, just fetch from cpu grad buffers
                                else:
                                    with torch.cuda.stream(
                                        mem_copy_stream
                                    ), torch.no_grad():
                                        gpu_grad_buffer.copy_(
                                            cpu_grad_buffers[i], non_blocking=True
                                        )
                                        copy_done_event.record(stream=mem_copy_stream)

                                    # make sure gradient is copied
                                    copy_done_event.wait(stream=compute_stream)

                                gradient = gpu_grad_buffer

                            # otherwise, just use from activations storage
                            else:
                                gradient = activations[li][i].grad

                    # --------> backwards for microbatch
                    outputs.backward(gradient=gradient)

                    # after backwards gradient is computed
                    if (
                        args.use_activation_buffers
                        and gpu_activation_buffer.grad is not None
                    ):
                        # store gradient for microbatch i (for next layer)
                        with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                            cpu_grad_buffers[i].copy_(
                                gpu_activation_buffer.grad.detach().data,
                                non_blocking=True,
                            )

                        # prefetch activation for recompute activations after backwards
                        # this is to make sure that we don't copy into gpu_activation_buffer
                        # before copying the gradient into cpu
                        if (
                            args.use_prefetch_backwards
                            and li > 0
                            and not (li == 1 and i == 0)
                        ):
                            buffer_idx_prefetch = (li - 1) * microbatches + i - 1
                            with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                                gpu_activation_buffer.copy_(
                                    cpu_buffers[buffer_idx_prefetch], non_blocking=True
                                )
                            # record for synch
                            recompute_activations_prefetch_event.record(
                                stream=mem_copy_stream
                            )

                    ### --------------------------------------------------------------------------<<< backwards for microbatch

                if li < num_layers - 1:
                    del activations[li]

            ### -------------------------------------------------------------------------->>> backwards pass

            # manually call the final callback for each fsdp module after entire backwards completes
            for layer in model_layers:
                _post_backward_final_callback(layer, None)

            # print mem
            if step_idx == print_mem_step:
                max_allocated_mem = print_memory_stats("post-backward", all_ranks=True)[
                    "max_allocated"
                ]

            #### ----->  optimizer step
            if not args.optimizer_in_backwards:
                optimizer.step()
                optimizer.zero_grad()

            if step_idx == print_mem_step:
                print_memory_stats("post-optimize", all_ranks=False)
                if is_leader():
                    try:
                        # Stop recording memory snapshot history.
                        torch.cuda.memory._record_memory_history(enabled=None)
                        torch.cuda.memory._dump_snapshot(
                            f"{profiler_path}/../mem_trace.pickle"
                        )
                    except Exception as e:
                        logger.error(f"Failed to capture memory snapshot {e}")

        if profile_step and detailed_profiler_trace and is_leader():
            profiler_ctx.export_memory_timeline(
                f"{profiler_path}/../mem_timeline.html", device=labels.device
            )

        # iteration end
        iteration_end.record()
        # synch for next iteration
        torch.cuda.synchronize()
        if zeus_monitor and is_leader():
            iteration_measurement = zeus_monitor.end_window(f"iteration_{step_idx}")
            logger.info(f"Iteration {step_idx} Energy: {iteration_measurement.total_energy} J, "
                    f"Power: {iteration_measurement.total_energy / iteration_measurement.time} W")
        logger.info(
            f" ### Iteration {step_idx} Time {iteration_start.elapsed_time(iteration_end)}"
        )

    # record end all iterations
    end_event.record()
    torch.cuda.synchronize()
    avg_iteration_time = start_event.elapsed_time(end_event) / iterations
    print_metrics(args, max_allocated_mem, avg_iteration_time, profiler_path)
    if zeus_monitor and is_leader():
        measurement = zeus_monitor.end_window("training")
        logger.info(f"Energy consumption: {measurement.total_energy} J")
        logger.info(f"Average power: {measurement.total_energy / measurement.time} W")
        logger.info(f"Settings: use_activation_buffers={getattr(args, 'use_activation_buffers', False)}, "
                    f"microbatches={args.microbatches}, batch_size={args.local_batch_size}, "
                    f"iterations={args.iterations}")


def train(args, model, optimizer, data_iter):
    """
    Training logic
    """
    logger = get_logger()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    model.train()
    warmup_iterations = args.warmup_iterations
    iterations = args.iterations
    total_iterations = warmup_iterations + iterations
    print_mem_step = warmup_iterations - 3
    profiler_path = get_profiler_path(args)

    if args.offload_activations:
        offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
    else:
        offload_ctx = contextlib.nullcontext()

    for step_idx in range(total_iterations):
        if step_idx == warmup_iterations:
            dist.barrier()
            start_event.record()

        # by default there are 5 warmup iterations, using 1 profiling itersations by default
        if (
            step_idx
            in range(warmup_iterations - args.profiling_iterations, warmup_iterations)
            and not args.skip_profile
        ):
            out_dir = os.path.join(
                profiler_path,
                f"iteration_{step_idx}",
            )
            profiler_ctx = get_profiler_context(out_dir=out_dir)
        else:
            profiler_ctx = contextlib.nullcontext()

        input_ids, labels = next(data_iter)

        iteration_start = torch.cuda.Event(enable_timing=True)
        iteration_end = torch.cuda.Event(enable_timing=True)
        iteration_start.record()
        with profiler_ctx:
            if step_idx == print_mem_step:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            if step_idx == print_mem_step:
                print_memory_stats("pre-forward", all_ranks=False)

            with torch.autocast(
                device_type="cuda", dtype=args.autocast_dtype
            ), offload_ctx:
                outputs = model(input_ids, labels=labels)

            if step_idx == print_mem_step:
                print_memory_stats("post-forward", all_ranks=False)

            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = outputs.sum()
            loss.backward()

            if step_idx == print_mem_step:
                max_allocated_mem = print_memory_stats("post-backward", all_ranks=True)[
                    "max_allocated"
                ]

            if not args.optimizer_in_backwards:
                optimizer.step()
                optimizer.zero_grad()

        if step_idx == print_mem_step:
            print_memory_stats("post-optimize", all_ranks=False)

        iteration_end.record()
        torch.cuda.synchronize()
        logger.info(
            f"Iteration {step_idx} Loss {loss.item()} Time {iteration_start.elapsed_time(iteration_end)}"
        )

    end_event.record()
    torch.cuda.synchronize()
    avg_iteration_time = start_event.elapsed_time(end_event) / iterations
    print_metrics(args, max_allocated_mem, avg_iteration_time, profiler_path)


def maybe_load_config(args):
    if args.config_file:
        with open(args.config_file, "r") as f:
            config = json.load(f)
            for key, value in config.items():
                if "dtype" in key:
                    value = get_dtype_from_str(value)
                setattr(args, key, value)


def main():
    """
    - Contains logic for custom hooks and patches.
    - Initializes split state and distributed environment
    - Initializes model, data, optimizer and runs training
    """

    init_split_state()

    args = parse_args()
    world_size = dist_init()
    maybe_load_config(args)
    args.world_size = world_size
    gpu_id = get_global_rank()
    init_logger(name="trainfsdp.py", log_level=args.log_level)
    logger = get_logger()
    logger.info(f"Running with args {args}")

    if args.sync_backwards:
        torch.distributed.fsdp._runtime_utils._post_backward_hook = (
            _post_backward_hook_sync
        )
    if args.optimizer_in_backwards:
        torch.distributed.fsdp._runtime_utils._post_reduce_grad_callback = (
            _post_reduce_grad_callback_patch
        )
        torch.distributed.fsdp.flat_param.FlatParamHandle.prepare_gradient_for_optim = (
            prepare_gradient_for_optim_patch
        )
        set_split_state(key="optimizer_in_backwards", value=True)

    if args.split_uneven:
        set_split_state(key="split_uneven", value=True)
        set_split_state(
            key="model_partitions",
            value=[float(x) + args.tol for x in args.split_uneven_partitions],
        )
        if args.proportional_split:
            set_split_state(key="proportional_split", value=True)

    if args.scale_config_bs > 1:
        if len(args.uneven_batch_sizes) > 0:
            args.uneven_batch_sizes = [
                x * args.scale_config_bs for x in args.uneven_batch_sizes
            ]
        else:
            args.batch_size = args.batch_size * args.scale_config_bs
        if args.uneven_microbatches is not None:
            args.uneven_microbatches = [
                x * args.scale_config_bs for x in args.uneven_microbatches
            ]
        else:
            args.microbatches = args.microbatches * args.scale_config_bs

    if len(args.uneven_batch_sizes) > 0:
        args.local_batch_size = int(args.uneven_batch_sizes[gpu_id])
        args.global_batch_size = sum([int(x) for x in args.uneven_batch_sizes])
    else:
        args.local_batch_size = args.batch_size
        args.global_batch_size = args.batch_size * args.world_size

    # patches for supporting uneven sharding
    FlatParamHandle.shard = shard_patch
    FlatParamHandle.unshard = unshard_patch
    FlatParamHandle.reshard = reshard_patch
    # custom hook for post backward
    torch.distributed.fsdp._runtime_utils._post_backward_hook = (
        _post_backward_hook_patch
    )
    args.microbatches = (
        args.microbatches
        if args.uneven_microbatches is None or len(args.uneven_microbatches) == 0
        else int(args.uneven_microbatches[gpu_id])
    )
    gradient_accumulation = args.microbatches > 1 or args.ga
    if gradient_accumulation:
        logger.info(
            f"Gradient accumulation with {args.microbatches} microbatches GPU {gpu_id}"
        )
        enable_gradient_accumulation()
        set_split_state(key="unshard_in_compute", value=args.unshard_in_compute)

    if args.hybrid_shard:
        assert (
            not args.split_uneven and not args.optimizer_in_backwards
        ), "hybrid_shard not supported with split_uneven or optimizer_in_backwards"

    zeus_monitor = None
    if args.energy_monitor and is_leader():
        zeus_monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    
    # initialize model after patches are applied
    model = model_init(args)
    data = get_dataset(args)
    data_iter = iter(data)

    if args.optimizer_in_backwards:
        optim = None
    else:
        optim = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            fused=not (args.no_fused_optimizer or args.cpu_offload),
        )

    if gradient_accumulation:
        train_ga(args, model, optim, data_iter, zeus_monitor)
    else:
        train(args, model, optim, data_iter)


if __name__ == "__main__":
    main()
