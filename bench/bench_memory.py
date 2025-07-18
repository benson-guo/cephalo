# -*- coding: utf-8 -*-
import contextlib
import torch
import json
import torch.distributed as dist
from utils.comm import (
    is_local_leader,
    is_leader,
    dist_init,
)
from utils.model import model_init
from utils.argparser_utils import parse_args, get_dtype_str
from utils.data_loader import get_dataset, get_image_dataset, split_microbatches
from utils.profile import print_memory_stats, fit_line
from utils.patch import enable_gradient_accumulation
from utils.global_state import init_split_state, configure_gradient_accumulation
from models.hub import get_all_layers, is_vision_model
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback

COMMUNICATION_STREAM_KERNELS = [
    "ncclKernel_Broadcast_RING_LL_Sum",
    "ncclKernel_Reduce_RING_LL_Sum_half",
    "ncclKernel_AllGather_RING_LL_Sum",
]
MEMCPY_STREAM_KERNELS = [
    "Memcpy DtoD (Device -> Device)",
    "Memcpy DtoH (Device -> Pinned)",
    "Memcpy HtoD (Pinned -> Device)",
]


def profile_ga(args, model, optimizer):
    model_name = args.model_name
    model.train()
    warmup_iterations = args.warmup_iterations
    iterations = args.iterations
    cluster = args.cluster
    total_iterations = warmup_iterations + iterations
    print_mem_step = total_iterations - 1

    max_allocated_hist = []
    max_allocated_delta = []
    compute_memory = []
    model_param_size = model._total_params * 4 * 4 / 1024**3 / args.world_size

    with open("data/model_memory.json", "r") as file:
        model_memory = json.load(file)
        assert cluster in model_memory
        assert model_name in model_memory[cluster]

    model_layers = get_all_layers(model)
    # initialize ga specific split_state
    num_layers = len(model_layers)
    max_microbatches = args.microbatches if args.microbatches > 1 else 5
    if args.offload_activations:
        offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=False)
    else:
        offload_ctx = contextlib.nullcontext()

    try:
        # scale microbatches
        for microbatches in range(2, max_microbatches):
            args.local_batch_size = microbatches
            args.microbatches = microbatches
            configure_gradient_accumulation(model, microbatches)
            data = (
                get_dataset(args)
                if not is_vision_model(model_name)
                else get_image_dataset(args)
            )
            data_iter = iter(data)

            for step_idx in range(total_iterations):
                if step_idx == warmup_iterations:
                    dist.barrier()

                input_ids, labels = next(data_iter)
                input_ids_microbatches, _ = split_microbatches(
                    input_ids, labels, microbatches
                )

                torch.cuda.synchronize()
                if step_idx == print_mem_step:
                    if is_local_leader():
                        print(f"Microbatches: {microbatches} Memory Stats:")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                activations = [input_ids_microbatches]
                with torch.autocast(
                    device_type="cuda", dtype=args.autocast_dtype
                ), offload_ctx:
                    for li, layer in enumerate(model_layers):
                        last_layer = li == num_layers - 1
                        if last_layer:
                            break
                        activations.append([])
                        for i in range(microbatches):
                            first_microbatch = i == 0
                            last_microbatch = i == microbatches - 1
                            layer_output = layer(
                                activations[-2][i],
                                first_microbatch=first_microbatch,
                                last_microbatch=last_microbatch,
                                last_layer=last_layer,
                                backwards=False,
                            ).detach()
                            activations[-1].append(layer_output)

                if step_idx == print_mem_step:
                    print_memory_stats("post-forward", all_ranks=False)

                for li in range(num_layers - 1, -1, -1):
                    last_layer = li == num_layers - 1
                    layer = model_layers[li]
                    for i in range(microbatches - 1, -1, -1):
                        first_microbatch = i == microbatches - 1
                        last_microbatch = i == 0
                        with torch.autocast(
                            device_type="cuda", dtype=args.autocast_dtype
                        ), offload_ctx:
                            if li > 0:
                                activations[li][i].requires_grad = True
                            outputs = layer(
                                activations[li][i],
                                first_microbatch=first_microbatch,
                                last_microbatch=last_microbatch,
                                last_layer=last_layer,
                                backwards=True,
                            )
                        if li == num_layers - 1:
                            outputs.sum().backward()
                        else:
                            outputs.backward(gradient=activations[li + 1][i].grad)
                    if li < num_layers - 1:
                        del activations[li + 1]

                # manually call the final callback for each fsdp module after entire backwards completes
                for layer in model_layers:
                    _post_backward_final_callback(layer, None)

                if step_idx == print_mem_step:
                    dist.barrier()
                    max_allocated = print_memory_stats(
                        "post-backward", all_ranks=False
                    )["max_allocated"]
                    max_allocated_hist.append(max_allocated)
                    compute_memory.append(max_allocated - model_param_size)
                    if is_local_leader():
                        print(f"Compute Memory: {compute_memory}")
                        if len(max_allocated_hist) > 1:
                            max_allocated_delta.append(
                                max_allocated_hist[-1] - max_allocated_hist[-2]
                            )
                            print(
                                f"Marginal memory increase: {max_allocated_delta[-1]}"
                            )
                    break

                optimizer.step()
                optimizer.zero_grad()

                if is_local_leader():
                    print(f"Iteration {step_idx}")
    # catch any error
    except Exception as e:
        print(f"Caught CUDA OOM: {e}")
    finally:
        if is_leader():
            print(f"Max Allocated History: {max_allocated_hist}")
            print(f"Max Allocated Delta: {max_allocated_delta}")
            print(f"Compute Memory History: {max_allocated_hist}")
            # update model_memory.json
            dtype = get_dtype_str(args.autocast_dtype)
            fit_memory = [
                x
                - model_memory[cluster][model_name]["profiled_compute_memory"][dtype][0]
                for x in compute_memory
            ]
            slope, intercept = fit_line(fit_memory)
            model_memory[cluster][model_name]["profiled_ga_model"][dtype] = (
                slope,
                intercept,
            )
            with open("data/model_memory.json", "w") as file:
                json.dump(
                    model_memory, file, ensure_ascii=False, indent=4, sort_keys=True
                )
            print(f"Fit GA Memory: {fit_memory} slope: {slope} intercept: {intercept}")


def profile_memory(args, model, optimizer):
    model_name = args.model_name
    model.train()
    warmup_iterations = args.warmup_iterations
    iterations = args.iterations
    total_iterations = warmup_iterations + iterations
    print_mem_step = total_iterations - 1

    max_allocated_hist = []
    max_allocated_delta = []
    compute_memory = []
    model_param_size = model._total_params * 4 * 4 / 1024**3 / args.world_size
    if args.offload_activations:
        offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=False)
    else:
        offload_ctx = contextlib.nullcontext()

    try:
        # scale batch size
        for bs in range(1, args.batch_size + 1):
            args.local_batch_size = bs
            data = (
                get_dataset(args)
                if not is_vision_model(model_name)
                else get_image_dataset(args)
            )
            data_iter = iter(data)
            for step_idx in range(total_iterations):
                if step_idx == warmup_iterations:
                    dist.barrier()

                input_ids, labels = next(data_iter)

                torch.cuda.synchronize()
                if step_idx == print_mem_step:
                    if is_local_leader():
                        print(f"Batch size: {bs} Memory Stats:")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                with torch.autocast(
                    device_type="cuda", dtype=args.autocast_dtype
                ), offload_ctx:
                    outputs = model(input_ids, labels=labels)

                loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()
                loss.backward()

                if step_idx == print_mem_step:
                    max_allocated = print_memory_stats(
                        "post-backward", all_ranks=False
                    )["max_allocated"]
                    max_allocated_hist.append(max_allocated)
                    compute_memory.append(max_allocated - model_param_size)
                    if is_local_leader():
                        print(f"Compute Memory: {compute_memory}")
                        if len(max_allocated_hist) > 1:
                            max_allocated_delta.append(
                                max_allocated_hist[-1] - max_allocated_hist[-2]
                            )
                            print(
                                f"Marginal memory increase: {max_allocated_delta[-1]}"
                            )
                    break

                optimizer.step()
                optimizer.zero_grad()

                if is_local_leader():
                    print(f"Iteration {step_idx}")
    # catch any error
    except Exception as e:
        print(f"Caught CUDA OOM: {e}")
    finally:
        if is_leader():
            print(f"Max Allocated History: {max_allocated_hist}")
            print(f"Max Allocated Delta: {max_allocated_delta}")
            print(f"Compute Memory History: {max_allocated_hist}")
            # update model_memory.json
            cluster = args.cluster
            dtype = get_dtype_str(args.autocast_dtype)
            with open("data/model_memory.json", "r") as file:
                model_memory = json.load(file)

                if cluster not in model_memory:
                    model_memory[cluster] = {}
                if model_name not in model_memory[cluster]:
                    model_memory[cluster][model_name] = {
                        "parameters": model._total_params,
                        "profiled_compute_memory": {},
                        "profiled_ga_model": {},
                    }
                if (
                    dtype
                    not in model_memory[cluster][model_name]["profiled_compute_memory"]
                ):
                    model_memory[cluster][model_name]["profiled_compute_memory"][
                        dtype
                    ] = []
                if len(compute_memory) > len(
                    model_memory[cluster][model_name]["profiled_compute_memory"][dtype]
                ):
                    model_memory[cluster][model_name]["profiled_compute_memory"][
                        dtype
                    ] = compute_memory

            with open("data/model_memory.json", "w") as file:
                json.dump(
                    model_memory, file, ensure_ascii=False, indent=4, sort_keys=True
                )


def find_gaps_in_stream(events, gap_threshold=0):
    """
    Find gaps in a sorted stream/similar events
    Set gap_threshold for minimum gaps ( in us )
    """
    gaps = []
    for i in range(len(events) - 1):
        start_time = events[i]["ts"] + events[i].get("dur", 0)
        end_time = events[i + 1]["ts"]
        gap = end_time - start_time
        if gap > gap_threshold:
            # name is for the kernel exec just before
            gaps.append((start_time, end_time, gap))
    return gaps


def get_overhead_from_stream(gaps, compare_stream):
    """
    Get overhead from comparing stream
    """
    total_overhead = 0

    for gap_start, gap_end, gap in gaps:
        overhead = 0
        overlapping_ops = []

        # Find operations that overlap with the current gap
        for op in compare_stream:
            op_start = op["ts"]
            op_end = op_start + op.get("dur", 0)

            # Check if operation overlaps with the gap
            if op_start < gap_end and op_end > gap_start:
                overlapping_ops.append(op)
                # Calculate overhead if the operation starts before the gap
                if op_start <= gap_start and op_end <= gap_end:
                    overhead += op_end - gap_start
                    # if overhead > 0:
                    #     print(f"Gap (duration: {gap} us) has an overhead of {overhead} us caused by overlapping stream operations")

        # Add the duration of any operations during the gap
        for op in overlapping_ops:
            op_start = op["ts"]
            op_end = op_start + op.get("dur", 0)
            # If the operation is entirely within the gap, add its duration to the overhead
            if op_start >= gap_start and op_end <= gap_end:
                overhead += op.get("dur", 0)
                # if overhead > 0:
                #     print(f"Gap (duration: {gap} us) has an overhead of {overhead} us caused by stream operations")

        total_overhead += overhead

    return total_overhead


def get_overheads(pt_trace_file, args, compute_overheads=True):
    """
    Returns:
    1. Overhead from cudaMalloc/Free for compute stream
    2. Overhead from compute stream for communication stream
    3. Total duration for cuda Malloc/Free, compute stream, communication stream, network stream
    """
    with open(pt_trace_file, "r") as f:
        data = json.load(f)

    trace_events = data["traceEvents"]

    # setting default streams
    compute_stream = 7
    gpu_mem_copy_stream = None
    communication_stream = None

    # find streams
    for event in trace_events:
        for k in COMMUNICATION_STREAM_KERNELS:
            if k in event.get("name", ""):
                communication_stream = event.get("tid")
                break

    for event in trace_events:
        for k in MEMCPY_STREAM_KERNELS:
            if k in event.get("name", ""):
                gpu_mem_copy_stream = event.get("tid")
                break

    if gpu_mem_copy_stream is None:
        gpu_mem_copy_stream = 16
    if communication_stream is None:
        communication_stream = 27

    # cuda malloc and free events
    cuda_events = [
        e
        for e in trace_events
        if "name" in e and (e["name"] == "cudaMalloc" or e["name"] == "cudaFree")
    ]
    total_cuda_duration = sum([op.get("dur", 0) for op in cuda_events])

    # compute stream
    compute_events = [
        e
        for e in trace_events
        if e.get("tid") == compute_stream and e.get("dur", 0) > 0
    ]
    sorted_compute_events = sorted(compute_events, key=lambda x: x["ts"])
    # total_compute_duration = sum([op.get("dur", 0) for op in sorted_compute_events])
    total_compute_duration = 0
    for i in range(len(sorted_compute_events)):
        event = sorted_compute_events[i]
        total_compute_duration += event.get("dur", 0)
        if i < len(sorted_compute_events) - 1:
            start_time = event["ts"] + event.get("dur", 0)
            end_time = sorted_compute_events[i + 1]["ts"]
            gap = end_time - start_time
            if gap < 500:
                total_compute_duration += gap

    # network stream
    gpu_mem_copy_events = [
        e for e in trace_events if e.get("tid") == gpu_mem_copy_stream
    ]
    total_mem_copy_duration = sum([op.get("dur", 0) for op in gpu_mem_copy_events])

    # communication stream
    communication_events = [
        e for e in trace_events if e.get("tid") == communication_stream
    ]
    total_communication_duration = sum(
        [op.get("dur", 0) for op in communication_events]
    )

    # find overheads
    if compute_overheads:
        gaps_in_compute = find_gaps_in_stream(sorted_compute_events, args.gap_threshold)
        sorted_communication_events = sorted(
            communication_events, key=lambda x: x["ts"]
        )
        gaps_in_communication = find_gaps_in_stream(
            sorted_communication_events, args.gap_threshold
        )
        sorted_cuda_events = sorted(cuda_events, key=lambda x: x["ts"])
        cuda_overhead = get_overhead_from_stream(gaps_in_compute, sorted_cuda_events)
        compute_overhead = get_overhead_from_stream(
            gaps_in_communication, sorted_compute_events
        )
        communication_overhead = get_overhead_from_stream(
            gaps_in_compute, sorted_communication_events
        )
        sorted_gpu_mem_copy_events = sorted(gpu_mem_copy_events, key=lambda x: x["ts"])
        # gaps_in_gpu_mem_copy = find_gaps_in_stream(
        #     sorted_gpu_mem_copy_events, args.gap_threshold
        # )
        network_overhead = get_overhead_from_stream(
            gaps_in_compute, sorted_gpu_mem_copy_events
        )
    else:
        cuda_overhead = 0.0
        compute_overhead = 0.0
        communication_overhead = 0.0
        network_overhead = 0.0

    return (
        cuda_overhead,
        compute_overhead,
        communication_overhead,
        network_overhead,
        (
            total_cuda_duration,
            total_compute_duration,
            total_mem_copy_duration,
            total_communication_duration,
        ),
    )


def main():
    args = parse_args()
    world_size = dist_init()
    args.world_size = world_size
    if is_leader():
        print(args)

    if args.profile_ga:
        init_split_state()
        enable_gradient_accumulation()
    model = model_init(args)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        fused=not (args.no_fused_optimizer or args.cpu_offload),
    )

    if args.profile_memory:
        profile_memory(args, model, optim)
    elif args.profile_ga:
        profile_ga(args, model, optim)


if __name__ == "__main__":
    main()
