# -*- coding: utf-8 -*-
import argparse
import torch
import functools
from models.hub import get_model
from utils.data_loader import get_dataset
from utils.profile import print_memory_stats
from utils.comm import (
    dist_init,
)


def train(args, model, data_iter, iterations):
    for step_idx in range(iterations):
        profile_mem_step = step_idx == iterations - 1
        input_ids, labels = next(data_iter)
        if profile_mem_step:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        with torch.autocast(device_type="cuda", dtype=args.autocast_dtype):
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        loss.backward()

        if profile_mem_step:
            max_allocated = print_memory_stats("post-backward", all_ranks=False)[
                "max_allocated"
            ]

    return max_allocated


def profile_memory(args):
    iterations = args.iterations
    model_name = args.model_name

    # profile memory as you increase number of layers
    layer_mem = []
    for layers in range(1, args.layers + 1):
        args.local_batch_size = 1
        data = get_dataset(args)
        data_iter = iter(data)
        model = get_model(
            model_name,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            layers=layers,
        ).cuda()

        max_allocated = train(args, model, data_iter, iterations)
        print(f"Layers: {layers} Max allocated memory: {max_allocated} GB")
        layer_mem.append(max_allocated)

    # profile memory as you increase batch size
    batch_mem = []
    for bs in range(1, args.batch_size + 1):
        args.local_batch_size = bs
        data = get_dataset(args)
        data_iter = iter(data)
        model = get_model(
            model_name, vocab_size=args.vocab_size, seq_length=args.seq_length, layers=1
        ).cuda()

        max_allocated = train(args, model, data_iter, iterations)
        print(f"Batch Size: {bs} Max allocated memory: {max_allocated} GB")
        batch_mem.append(max_allocated)

    print(f"Layer Memory: {layer_mem}")
    print(f"Batch Memory: {batch_mem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt_85m")
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=16)
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--autocast_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    args = parser.parse_args()
    dist_init()
    profile_memory(args)


if __name__ == "__main__":
    main()
