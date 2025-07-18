# -*- coding: utf-8 -*-
from typing import Dict
import json


# returns 1 layer forward + backward latencies for given model, dtype, batch_sze
# per gpu based on profiled linear model
def load_model_latencies_per_gpu(
    model_name, dtype, batch_size, seq_length
) -> Dict[str, float]:
    with open("data/model_latencies.json", "r") as f:
        model_latencies = json.load(f)

    compute_times = {}
    for gpu in model_latencies[model_name].keys():
        runtime = model_latencies[model_name][gpu][seq_length][dtype]

        # For smaller microbatches we have profiled, use runtime directly
        if isinstance(runtime, list) and len(runtime) == 4:
            runtimes_forwards = runtime[2]
            runtimes_backwards = runtime[3]
        else:
            runtimes_forwards = []
            runtimes_backwards = []
        # For larger microbatches we have not profiled, extrapolate with linear model
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
        runtime_slope = forward_slope + backward_slope
        runtime_intercept = forward_intercept + backward_intercept

        if len(runtimes_backwards) >= batch_size:
            mb_runtime = (
                runtimes_forwards[batch_size - 1] + runtimes_backwards[batch_size - 1]
            )
        else:
            mb_runtime = runtime_slope * batch_size + runtime_intercept
        compute_times[gpu] = mb_runtime

    return compute_times

