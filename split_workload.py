# -*- coding: utf-8 -*-
import argparse
import json
import pulp as lp
import math
import torch
import functools
from functools import lru_cache
from utils.comm import dist_init
from utils.argparser_utils import get_dtype_str
from utils.runtime_estimator import GPU_MEMORY


def split_batch(
    args,
    model_name,
    model_memory,
    gpus,
    batch_size,
    strategy="dp",
    seq_length=512,
    comm_mem_overhead=2.0,
    max_alloc_frac=0.75,
    max_microbatch_size=10000,
):
    with open("data/model_latencies.json", "r") as f:
        model_latencies = json.load(f)
    runtimes_slope = []
    runtimes_intercept = []
    runtimes_slope_forwards = []
    runtimes_intercept_forwards = []
    runtimes_slope_backwards = []
    runtimes_intercept_backwards = []
    runtimes_forwards = []
    runtimes_backwards = []
    dtype = get_dtype_str(args.dtype)
    for gpu in gpus:
        runtime = model_latencies[model_name][gpu][str(seq_length)][dtype]
        if isinstance(runtime, list):
            if isinstance(runtime[0], list):
                forward_slope, forward_intercept = runtime[0]
                backward_slope, backward_intercept = runtime[1]
            else:
                forward_slope, forward_intercept = runtime[0]
                backward_slope, backward_intercept = 0, 0
        else:
            forward_slope, forward_intercept = runtime, 0
            backward_slope, backward_intercept = 0, 0
        if isinstance(runtime, list) and len(runtime) == 4:
            runtimes_forwards.append(runtime[2])
            runtimes_backwards.append(runtime[3])
        else:
            runtimes_forwards.append([])
            runtimes_backwards.append([])
        runtimes_slope.append(forward_slope + backward_slope)
        runtimes_intercept.append(forward_intercept + backward_intercept)
        runtimes_slope_forwards.append(forward_slope)
        runtimes_intercept_forwards.append(forward_intercept)
        runtimes_slope_backwards.append(backward_slope)
        runtimes_intercept_backwards.append(backward_intercept)

    if strategy == "greedy":
        return split_batch_greedy(runtimes_slope, batch_size), None
    else:
        train_memory = model_memory["parameters"] * 4 * 4 / 1024**3 / len(gpus)
        max_batch_sizes_es = [
            max_batch_size(
                args,
                gpu,
                model_memory,
                comm_mem_overhead=comm_mem_overhead,
                max_alloc_frac=max_alloc_frac,
                train_memory=train_memory,
            )
            for gpu in gpus
        ]
        max_batch_sizes = [
            max_batch_size(
                args,
                gpu,
                model_memory,
                comm_mem_overhead=comm_mem_overhead,
                max_alloc_frac=max_alloc_frac,
            )
            for gpu in gpus
        ]
        print(f"Max batch sizes es: {max_batch_sizes_es}")
        print(f"Max batch sizes: {max_batch_sizes}")
        if strategy == "lp":
            return split_batch_lp(runtimes_slope, max_batch_sizes_es, batch_size), -1
        elif strategy == "dp":
            return (
                split_batch_dp(
                    runtimes_slope_forwards,
                    runtimes_intercept_forwards,
                    runtimes_slope_backwards,
                    runtimes_intercept_backwards,
                    runtimes_forwards,
                    runtimes_backwards,
                    max_batch_sizes_es,
                    max_batch_sizes,
                    batch_size,
                ),
                None,
            )
        elif strategy == "dp_ga":
            ag_time = args.ag_time
            rs_time = args.rs_time
            batch_sizes, microbatch_sizes = split_batch_dp_ga(
                runtimes_slope_forwards,
                runtimes_intercept_forwards,
                runtimes_slope_backwards,
                runtimes_intercept_backwards,
                runtimes_forwards,
                runtimes_backwards,
                max_batch_sizes_es,
                max_batch_sizes,
                batch_size,
                model_memory,
                gpus,
                ag_time=ag_time,
                rs_time=rs_time,
                comm_mem_overhead=comm_mem_overhead,
                max_alloc_frac=max_alloc_frac,
                max_microbatch_size=max_microbatch_size,
            )
            return batch_sizes, microbatch_sizes
        else:
            raise NotImplementedError


def split_batch_greedy(runtimes, batch_size):
    # Step 1: Calculate the Total Processing Power
    processing_rates = [1 / runtime for runtime in runtimes]
    total_power = sum(processing_rates)

    # Step 2: Determine the Proportional Share of Each Machine
    proportions = [rate / total_power for rate in processing_rates]

    # Step 3: Allocate Jobs Based on Proportion
    assigned_bs = [round(batch_size * prop) for prop in proportions]

    # Step 4: Adjust for Whole Numbers
    total_allocated = sum(assigned_bs)
    while total_allocated != batch_size:
        if total_allocated > batch_size:
            # Remove a job from the machine with the largest excess
            max_index = assigned_bs.index(max(assigned_bs))
            assigned_bs[max_index] -= 1
            total_allocated -= 1
        else:
            # Add a job to the machine with the largest deficit
            min_index = assigned_bs.index(min(assigned_bs))
            assigned_bs[min_index] += 1
            total_allocated += 1

    return assigned_bs


def mem_usage(args, batch_size, model_memory, comm_mem_overhead=2, microbatches=1):
    dtype = get_dtype_str(args.dtype)
    profiled_memory = model_memory["profiled_compute_memory"][dtype]
    # after offloading activations additional microbatches incur 0 memory overhead
    # profiled_ga_model = model_memory["profiled_ga_model"][dtype]
    profiled_ga_model = [0, 0]
    microbatch_size = math.ceil(batch_size / microbatches)
    largest_profiled_batch = len(profiled_memory)
    if microbatch_size > largest_profiled_batch:
        avg_len = min(5, len(profiled_memory))
        growth_rate = (profiled_memory[-1] - profiled_memory[-avg_len]) / (avg_len - 1)
        compute_memory = profiled_memory[-1] + growth_rate * (
            microbatch_size - largest_profiled_batch
        )
    else:
        compute_memory = profiled_memory[microbatch_size - 1]
    marginal_microbatch_memory = 0
    if microbatches > 1:
        marginal_microbatch_memory = (
            profiled_ga_model[0] * (microbatches - 1) * microbatch_size
            + profiled_ga_model[1]
        )

    return compute_memory + marginal_microbatch_memory + comm_mem_overhead


def max_batch_size(
    args, gpu, model_memory, comm_mem_overhead=2, max_alloc_frac=0.75, train_memory=0
):
    max_memory = max_alloc_frac * GPU_MEMORY[gpu] - train_memory
    i = 1
    while True:
        memory_util = mem_usage(
            args, i, model_memory, comm_mem_overhead=comm_mem_overhead
        )
        if memory_util > max_memory:
            return i - 1
        i += 1


def max_num_microbatches(
    args,
    gpu,
    model_memory,
    comm_mem_overhead=2,
    max_alloc_frac=0.75,
    train_memory=0,
    microbatch_size=1,
):
    # don't exceed max_alloc_frac of capacity and subtract out memory needed to store training state
    max_memory = max_alloc_frac * GPU_MEMORY[gpu] - train_memory
    microbatches = 1
    while True:
        batch_size = microbatches * microbatch_size
        if (
            mem_usage(
                args,
                batch_size,
                model_memory,
                comm_mem_overhead=comm_mem_overhead,
                microbatches=microbatches,
            )
            > max_memory
        ):
            return microbatches - 1
        microbatches += 1


def split_batch_lp(runtimes, max_batch_sizes, batch_size):
    num_gpus = len(runtimes)
    # Create the LP problem
    prob = lp.LpProblem("BatchSizeDistribution", lp.LpMinimize)

    # Variables: Batch size for each machine
    bs_vars = lp.LpVariable.dicts("BatchSizes", range(num_gpus), 0, None, lp.LpInteger)

    # Objective: Minimize the maximum runtime
    max_runtime = lp.LpVariable("MaxRuntime", 0)
    prob += max_runtime

    # Constraints
    # Total jobs assigned equals num_jobs
    prob += lp.lpSum([bs_vars[i] for i in range(num_gpus)]) == batch_size

    # Each machine's assigned jobs do not exceed its capacity
    for i in range(num_gpus):
        prob += bs_vars[i] <= max_batch_sizes[i]

    # Max runtime constraint for each machine
    for i in range(num_gpus):
        prob += bs_vars[i] <= max_runtime * (1 / runtimes[i])

    # Solve the problem
    prob.solve()
    # Extract the job distribution if the problem has a solution
    assert lp.LpStatus[prob.status] == "Optimal"
    distribution = [int(bs_vars[i].value()) for i in range(num_gpus)]
    return distribution


def split_batch_dp(
    runtimes_slope_forwards,
    runtimes_intercept_forwards,
    runtimes_slope_backwards,
    runtimes_intercept_backwards,
    runtimes_forwards,
    runtimes_backwards,
    max_batch_sizes_es,
    max_batch_sizes,
    batch_size,
    ag_time=0,
    rs_time=0,
):
    @lru_cache(maxsize=None)
    def backwards_compute_time(gpu_id, microbatch_size):
        if len(runtimes_backwards[gpu_id]) >= microbatch_size:
            mb_runtime = runtimes_backwards[gpu_id][microbatch_size - 1]
        else:
            mb_runtime = (
                microbatch_size * runtimes_slope_backwards[gpu_id]
                + runtimes_intercept_backwards[gpu_id]
            )
        return mb_runtime

    @lru_cache(maxsize=None)
    def forwards_compute_time(gpu_id, microbatch_size):
        if len(runtimes_forwards[gpu_id]) >= microbatch_size:
            mb_runtime = runtimes_forwards[gpu_id][microbatch_size - 1]
        else:
            mb_runtime = (
                microbatch_size * runtimes_slope_forwards[gpu_id]
                + runtimes_intercept_forwards[gpu_id]
            )
        return mb_runtime

    num_gpus = len(runtimes_slope_forwards)

    # DP[i][j][0] stores minimum runtime for first i gpus to process a batch size of j
    # DP[i][j][1] stores the batch size of the ith GPU
    dp = [[[math.inf, -1] for _ in range(batch_size + 1)] for _ in range(num_gpus + 1)]
    dp[0][0] = [0, 0]  # No processing time and batch size for 0 GPUs and batch size

    for i in range(1, num_gpus + 1):
        for j in range(1, batch_size + 1):
            max_batch_size_es = max_batch_sizes_es[i - 1]
            max_batch_size = max_batch_sizes[i - 1]
            for k in range(1, min(j, max_batch_size) + 1):
                cur_ag_time = 1.25 * ag_time if k > max_batch_size_es else ag_time
                cur_rs_time = 1.25 * rs_time if k > max_batch_size_es else rs_time
                compute_time_forwards = forwards_compute_time(i - 1, k)
                compute_time_backwards = backwards_compute_time(i - 1, k)
                forwards_time = max(
                    cur_ag_time,
                    compute_time_forwards,
                )
                backwards_time = max(
                    cur_ag_time + cur_rs_time,
                    compute_time_forwards + compute_time_backwards,
                )
                runtime = forwards_time + backwards_time
                # overall runtime is max of runtime of current and previous gpus
                time = max(dp[i - 1][j - k][0], runtime)
                if time < dp[i][j][0]:
                    dp[i][j] = [
                        time,
                        k,
                    ]  # Update with the better option and store the choice of k

    # Backtrack to find the solution
    solution = []
    current_batch_size = batch_size
    for i in range(num_gpus, 0, -1):
        k = dp[i][current_batch_size][1]
        solution.append(k)
        current_batch_size -= k

    # Reverse the solution to start from the first GPU
    solution.reverse()
    return solution


def split_batch_dp_ga(
    runtimes_slope_f,
    runtimes_intercept_f,
    runtimes_slope_b,
    runtimes_intercept_b,
    runtimes_forwards,
    runtimes_backwards,
    max_batch_sizes_es,
    max_batch_sizes,
    batch_size,
    model_memory,
    gpus,
    ag_time=0,
    rs_time=0,
    comm_mem_overhead=2,
    max_alloc_frac=0.75,
    ga_overhead=10.0,
    max_microbatch_size=10000,
):
    @lru_cache(maxsize=None)
    def backwards_compute_time(gpu_id, num_microbatches, microbatch_size):
        if len(runtimes_backwards[gpu_id]) >= microbatch_size:
            mb_runtime = runtimes_backwards[gpu_id][microbatch_size - 1]
        else:
            mb_runtime = (
                microbatch_size * runtimes_slope_b[gpu_id]
                + runtimes_intercept_b[gpu_id]
            )
        return (num_microbatches - 1) * ga_overhead + num_microbatches * mb_runtime

    @lru_cache(maxsize=None)
    def forwards_compute_time(gpu_id, num_microbatches, microbatch_size):
        if len(runtimes_forwards[gpu_id]) >= microbatch_size:
            mb_runtime = runtimes_forwards[gpu_id][microbatch_size - 1]
        else:
            mb_runtime = (
                microbatch_size * runtimes_slope_f[gpu_id]
                + runtimes_intercept_f[gpu_id]
            )
        return num_microbatches * mb_runtime

    num_gpus = len(runtimes_slope_f)

    # DP[i][j][k][0] stores minimum runtime for first i gpus to process a batch size of j with sum of microbatches = k
    # DP[i][j][k][1] stores the batch size of the ith GPU
    # DP[i][j][k][2] stores the microbatch size of the ith GPU
    dp = [
        [
            [[math.inf, -1, -1] for _ in range(batch_size + 1)]
            for _ in range(batch_size + 1)
        ]
        for _ in range(num_gpus + 1)
    ]
    dp[0][0][0] = [
        0,
        0,
        0,
    ]  # No processing time and batch size for 0 GPUs and batch size

    for i in range(1, num_gpus + 1):
        for j in range(1, batch_size + 1):
            for k in range(1, j + 1):
                max_mbs = min(max_microbatch_size, max_batch_sizes[i - 1], k)
                for microbatch_size in range(1, max_mbs + 1):
                    max_mbs_es = max_batch_sizes_es[i - 1]
                    # assume 25% runtime overhead if we need to uneven shard
                    cur_ag_time = (
                        1.25 * ag_time if microbatch_size > max_mbs_es else ag_time
                    )
                    cur_rs_time = (
                        1.25 * rs_time if microbatch_size > max_mbs_es else rs_time
                    )

                    for num_microbatch in range(1, j // microbatch_size + 1):
                        cur_batch_size = num_microbatch * microbatch_size
                        # overall compute time is max of compute time of current and previous gpus
                        compute_time_forwards = forwards_compute_time(
                            i - 1, num_microbatch, microbatch_size
                        )
                        compute_time_backwards = backwards_compute_time(
                            i - 1, num_microbatch, microbatch_size
                        )
                        forwards_time = max(
                            cur_ag_time,
                            compute_time_forwards,
                        )
                        backwards_time = max(
                            cur_ag_time + cur_rs_time,
                            compute_time_forwards + compute_time_backwards,
                        )
                        compute_time = forwards_time + backwards_time
                        time = max(
                            dp[i - 1][j - cur_batch_size][k - microbatch_size][0],
                            compute_time,
                        )
                        if time < dp[i][j][k][0]:
                            dp[i][j][k] = [
                                time,
                                cur_batch_size,
                                microbatch_size,
                            ]  # Update with the better option and store the choice of k

    # Backtrack to find the solution
    aggregate_gpu_memory = sum([GPU_MEMORY[gpu] for gpu in gpus])
    memory_cap = max_alloc_frac * aggregate_gpu_memory
    best_batch_sizes = None
    best_microbatch_sizes = None
    best_throughputs = 0
    for microbatch_sum in range(1, batch_size + 1):
        batch_sizes = []
        microbatch_sizes = []
        current_batch_size = batch_size
        current_microbatch_sum = microbatch_sum
        for i in range(num_gpus, 0, -1):
            j = dp[i][current_batch_size][current_microbatch_sum][1]
            k = dp[i][current_batch_size][current_microbatch_sum][2]
            if k < 0:
                break

            current_batch_size -= j
            current_microbatch_sum -= k
            batch_sizes.append(j)
            microbatch_sizes.append(k)

        if len(batch_sizes) < num_gpus:
            continue
        # Reverse the solution to start from the first GPU
        batch_sizes.reverse()
        microbatch_sizes.reverse()
        throughput = batch_size / dp[num_gpus][batch_size][microbatch_sum][0]
        solution_memory = sum(
            [
                mem_usage(
                    args,
                    bs,
                    model_memory,
                    comm_mem_overhead=comm_mem_overhead,
                    microbatches=bs // mbs,
                )
                for bs, mbs in zip(batch_sizes, microbatch_sizes)
            ]
        )
        solution_memory += model_memory["parameters"] * 4 * 4 / 1024**3
        print(
            f"Microbatch sum : {microbatch_sum} Batch sizes : {batch_sizes} Microbatch sizes : {microbatch_sizes} Throughput : {throughput} Memory : {solution_memory} / {memory_cap}"
        )
        if throughput > best_throughputs and (
            solution_memory < memory_cap or best_throughputs == 0
        ):
            best_batch_sizes = batch_sizes
            best_microbatch_sizes = microbatch_sizes
            best_throughputs = throughput

    best_num_microbatches = [
        bs // ms for bs, ms in zip(best_batch_sizes, best_microbatch_sizes)
    ]
    compute_times = [
        backwards_compute_time(gpu_id, bs // ms, ms)
        + forwards_compute_time(gpu_id, bs // ms, ms)
        for gpu_id, (bs, ms) in enumerate(zip(best_batch_sizes, best_microbatch_sizes))
    ]
    print(f"Solution: Throughput : {best_throughputs}")
    print(f"Batch sizes : {best_batch_sizes}")
    print(f"Num Microbatches: {best_num_microbatches}")
    print(f"Microbatch sizes : {best_microbatch_sizes}")
    print(f"Compute times : {max(compute_times)} : {compute_times}")
    return best_batch_sizes, best_num_microbatches


# split parameters unevenly with the goal of minimizing the max parameters assigned to any GPU
# greedily move parameters from GPUs that don't have enough memory to GPUs with extra memory
def split_parameters_min_params(
    args,
    model_memory,
    gpus,
    batch_sizes,
    uneven_microbatches=None,
    profiled_base_memories=None,
    comm_mem_overhead=2,
    max_alloc_frac=0.75,
):
    max_memories = [GPU_MEMORY[gpu] for gpu in gpus]
    extra_param_space = []
    total_params = model_memory["parameters"]
    delta_param_gb = 4 * 4 / 1024**3
    even_split_params = total_params / len(gpus)
    even_split_size = even_split_params * delta_param_gb
    params_to_move = 0
    assigned_params = []
    for idx, batch_size in enumerate(batch_sizes):
        if profiled_base_memories is not None:
            base_memory = profiled_base_memories[idx]
        else:
            microbatches = (
                1
                if uneven_microbatches is None
                else batch_size // uneven_microbatches[idx]
            )
            base_memory = mem_usage(
                args,
                batch_size,
                model_memory,
                comm_mem_overhead=comm_mem_overhead,
                microbatches=microbatches,
            )
        # this GPU can support extra_params params more than an even split of the params without exceeding GPU memory limits
        # this can be negative, in which case it needs this many parameters fewer than an split share of gpu parameters
        extra_params = (
            max_alloc_frac * max_memories[idx] - base_memory - even_split_size
        ) // delta_param_gb
        if extra_params > 0:
            extra_param_space.append((extra_params, idx))
            assigned_params.append(even_split_params)
        else:
            params_to_move += -extra_params
            assigned_params.append(even_split_params + extra_params)

    extra_param_space.sort()
    prev_added_params = 0
    extra_gpus = len(extra_param_space)
    for i in range(extra_gpus):
        additional_params = extra_param_space[i][0] - prev_added_params
        params_needed = math.ceil(params_to_move // (extra_gpus - i))
        param_increase_per_gpu = min(additional_params, params_needed)
        for j in range(i, extra_gpus):
            gpu_idx = extra_param_space[j][1]
            assigned_params[gpu_idx] += param_increase_per_gpu
            params_to_move -= param_increase_per_gpu

        if params_to_move <= 0:
            break
        prev_added_params = extra_param_space[i][0]

    return assigned_params


def split_parameters_min_util(
    args,
    model_memory,
    gpus,
    batch_sizes,
    uneven_microbatches=None,
    profiled_base_memories=None,
    comm_mem_overhead=2,
    max_balance_util=0.5,
):
    num_gpus = len(gpus)
    max_memories = [GPU_MEMORY[gpu] for gpu in gpus]
    base_memories = []
    mem_utilization = []
    total_params = model_memory["parameters"]
    for idx, batch_size in enumerate(batch_sizes):
        if profiled_base_memories is not None:
            base_memory = profiled_base_memories[idx]
        else:
            microbatches = (
                1
                if uneven_microbatches is None
                else batch_size // uneven_microbatches[idx]
            )
            base_memory = mem_usage(
                args,
                batch_size,
                model_memory,
                comm_mem_overhead=comm_mem_overhead,
                microbatches=microbatches,
            )
        base_memories.append(base_memory)
        mem_utilization.append((base_memory / max_memories[idx], idx))

    # params, gradients, 2x optimizer state, 4 data size
    delta_param_gb = 4 * 4 / 1024**3
    max_es_util = max(
        [
            (bm + total_params * delta_param_gb / num_gpus) / mm
            for bm, mm in zip(base_memories, max_memories)
        ]
    )
    if max_es_util < max_balance_util:
        print(
            f"Max Even Shard utilization {max_es_util} < max_balance_util {max_balance_util}, using even shard"
        )
        assigned_params = [total_params / num_gpus for _ in gpus]
    else:
        print(
            f"Max Even Shard utilization {max_es_util} >= max_balance_util {max_balance_util}, uneven shard"
        )
        assigned_params = [0 for _ in gpus]
        mem_utilization.sort()
        # dummy GPU
        mem_utilization.append((1.0, -1))

        remaining_params = total_params
        mem_sum = 0.0
        for i in range(num_gpus):
            cur_mem_utilization = mem_utilization[i][0]
            mem_sum += max_memories[mem_utilization[i][1]]
            if abs(mem_utilization[i + 1][0] - cur_mem_utilization) < 0.001:
                continue
            # percentage of memory utilization we can increase by and how many parameters that corresponds to
            diff_util = mem_utilization[i + 1][0] - mem_utilization[i][0]
            split_params = min(
                int(mem_sum * diff_util / delta_param_gb), remaining_params
            )

            # assign parameters proportional to max memory / mem sum
            total_added = 0
            for j in range(i + 1):
                gpu_idx = mem_utilization[j][1]
                if j == i:
                    num_params = split_params - total_added
                else:
                    num_params = int(split_params * max_memories[gpu_idx] / mem_sum)
                assigned_params[gpu_idx] += num_params
                total_added += num_params

            remaining_params -= split_params
            if remaining_params == 0:
                break

    print("Estimated Utilization:", end="")
    for (base_memory, params, max_mem) in zip(
        base_memories, assigned_params, max_memories
    ):
        memory_usage = base_memory + params * delta_param_gb
        print(f" {memory_usage:0.2f} ({100 * memory_usage / max_mem:.2f}%)", end="")
    print("")

    return assigned_params


def split_workload(args):
    model_name = args.model_name
    cluster = args.cluster
    gpus = args.gpus
    batch_size = args.batch_size
    seq_length = args.seq_length
    max_microbatch_size = args.max_microbatch_size
    extrapolate_bs_factor = args.extrapolate_bs_factor
    profiled_base_memories = [float(x.rstrip(",")) for x in args.profiled_base_memories]
    uneven_microbatches = None

    with open("data/model_memory.json", "r") as f:
        model_memory = json.load(f)[cluster][model_name]

    # Split the batch size
    if args.even_bs:
        local_bs = batch_size // len(gpus)
        assigned_bs = [local_bs for _ in gpus]
        for i in range(batch_size % len(gpus)):
            assigned_bs[i] += 1
        uneven_microbatches = [
            math.ceil(bs / max_microbatch_size) for bs in assigned_bs
        ]
    else:
        if extrapolate_bs_factor > 1 and not args.strategy == "dp_ga":
            raise NotImplementedError(
                "Extrapolating batch size is only supported for DP GA"
            )
        assigned_bs, uneven_microbatches = split_batch(
            args,
            model_name,
            model_memory,
            gpus,
            batch_size // extrapolate_bs_factor,
            strategy=args.strategy,
            seq_length=seq_length,
            comm_mem_overhead=args.comm_mem_overhead,
            max_alloc_frac=args.max_alloc_frac,
            max_microbatch_size=max_microbatch_size,
        )
        if extrapolate_bs_factor > 1:
            for i in range(len(assigned_bs)):
                assigned_bs[i] *= extrapolate_bs_factor
                uneven_microbatches[i] *= extrapolate_bs_factor

    # Split the parameters
    if args.even_params:
        assigned_params = [1 for _ in gpus]
    elif args.split_param_objective == "min_util":
        assigned_params = split_parameters_min_util(
            args,
            model_memory,
            gpus,
            assigned_bs,
            uneven_microbatches=uneven_microbatches,
            profiled_base_memories=profiled_base_memories
            if len(profiled_base_memories) == len(gpus)
            else None,
            comm_mem_overhead=args.comm_mem_overhead,
            max_balance_util=args.max_balance_util,
        )
    elif args.split_param_objective == "min_params":
        assigned_params = split_parameters_min_params(
            args,
            model_memory,
            gpus,
            assigned_bs,
            uneven_microbatches=uneven_microbatches,
            profiled_base_memories=profiled_base_memories
            if len(profiled_base_memories) == len(gpus)
            else None,
            comm_mem_overhead=args.comm_mem_overhead,
        )
    param_ratios = [params / sum(assigned_params) for params in assigned_params]

    print(f"Assigned batch sizes: {assigned_bs}")
    print(f"Assigned num microbatches: {uneven_microbatches}")
    print(f"Assigned parameters: {assigned_params}")
    aggregate_gpu_memory = sum([GPU_MEMORY[gpu] for gpu in gpus])
    print(f"Aggregate GPU memory: {aggregate_gpu_memory}")
    ratio_str = ""
    for ratio in param_ratios:
        ratio_str += f"{ratio:.4f} "
    print(f"Parameter ratios: {ratio_str}")

    return assigned_bs, uneven_microbatches, param_ratios


def main(args):
    dist_init()
    experiment_name = (
        args.experiment_name
        if args.experiment_name is not None
        else f"{args.cluster}_{args.model_name}_{args.strategy}_{args.dtype}_{'ep' if args.even_params else 'su'}_{'ebs' if args.even_bs else 'ec'}_s{args.seq_length}_b{args.batch_size}"
    )
    if args.strategy == "dp_ga":
        if args.max_microbatch_size < 10000:
            experiment_name += f"_mbs{args.max_microbatch_size}"
        experiment_name += f"_ext{args.extrapolate_bs_factor}"
    config_file = (
        args.config_file
        if args.config_file is not None
        else f"{args.config_file_dir}/{experiment_name}.json"
    )
    print(f"Splitting workload for {experiment_name}")
    assigned_bs, uneven_microbatches, param_ratios = split_workload(args)
    print(f"Writing config to {config_file}")

    # write to config json file
    use_ga = args.strategy == "dp_ga" or (args.even_bs and args.strategy == "greedy")
    dtype = get_dtype_str(args.dtype)
    reduce_dtype = get_dtype_str(args.reduce_dtype)
    with open(config_file, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "uneven_batch_sizes": assigned_bs,
                "split_uneven_partitions": param_ratios,
                "split_uneven": not args.even_params,
                "uneven_microbatches": uneven_microbatches,
                "seq_length": args.seq_length,
                "recompute_layer": not use_ga,  # ga automatically applies recomputation
                "experiment_name": experiment_name,
                "ga": use_ga,
                # use activation buffers and prefetch so ga memory growth is constant
                "use_activation_buffers": use_ga,
                "use_prefetch_backwards": use_ga,
                "reduce_dtype": reduce_dtype,
                "autocast_dtype": dtype,
                "image_size": args.image_size,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    parser.add_argument(
        "--reduce_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float32,
    )
    parser.add_argument("--model_name", type=str, default="gpt_1b")
    parser.add_argument("--cluster", type=str, default="paper")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--config_file_dir", type=str, default="/tmp")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument(
        "--strategy", type=str, default="dp", help="greedy, lp, dp, dp_ga"
    )
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--extrapolate_bs_factor", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_microbatch_size", type=int, default=10000)
    parser.add_argument("--ag_time", type=float, default=0)
    parser.add_argument("--rs_time", type=float, default=0)
    parser.add_argument("--max_alloc_frac", type=float, default=0.75)
    parser.add_argument("--max_balance_util", type=float, default=0.5)
    parser.add_argument("--comm_mem_overhead", type=float, default=2)
    parser.add_argument("--even_bs", action="store_true")
    parser.add_argument("--even_params", action="store_true")
    parser.add_argument(
        "-g",
        "--gpus",
        nargs="+",
        default=["p40"],
        help="",
    )
    parser.add_argument(
        "-pm",
        "--profiled_base_memories",
        nargs="+",
        default=[],
        help="",
    )
    parser.add_argument(
        "-gm",
        "--ga_microbatches",
        nargs="+",
        default=[],
        help="",
    )
    parser.add_argument(
        "--split_param_objective",
        type=str,
        default="min_params",
        help="min_params, min_util. min_params minimizes the max parameters assigned to a GPU"
        " and min_util minimizes the max gpu utilization.",
    )
    args = parser.parse_args()
    main(args)
