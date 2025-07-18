# Cephalo
Cephalo is a system for training transformers and LLMs on heterogeneous GPU clusters. It parallelizes training with FSDP, balancing memory and compute utilization across different GPUs within the cluster. 


## Setup
The setup script will create a conda environment and install Pytorch 2.1.2 as well as other libraries needed to run the code. Tested with CUDA 12.1, PyTorch 2.1.2.

```sh
./setup.sh
```

## Execution
The following is an example for training GPT 2.7B model with a batch size of 64 on a cluster of 4 GPUS: 2xL4, 1xA6000, 1xP40.

<h4>Profiling</h4>
First run the profiler to profile the model layer runtimes. This will profile every distinct GPU on the cluster:

```sh
./profile_models.sh float16 512 29500 deepspeedgpt_2.7b 
```

Then, profile the memory utilization. The following commands will profile a few iterations of training and construct a memory utilization model which will be used in the optimizer. Two passes need to be run, one with `--profile_memory`, and one for `--profile_ga`.
```sh
torchrun --nproc_per_node=4 -m bench.bench_memory --batch_size 5 --recompute_layer --model_name deepspeedgpt_2.7b --profile_memory;
torchrun --nproc_per_node=4 -m bench.bench_memory --batch_size 5 --recompute_layer --model_name deepspeedgpt_2.7b --profile_ga
```

<h4>Optimization</h4>
Then, run the optimizer to determine the batch size and parameter sharding to train with across the GPUs.

```sh
torchrun --nproc_per_node=1 split_workload.py --gpus l4 l4 a6000 p40 --model_name deepspeedgpt_2.7b --batch_size 64 --config_file example/gpt_2.7b.json --strategy dp_ga
```

<h4>Training</h4>
Then run the trainer with the assigned batch sizes and parameter sharding produced from the optimizer:

```sh
torchrun --nproc_per_node=4 trainfsdp.py --recompute_layer --config_file example/gpt_2.7b.json
```

<h4>Distributed Training</h4>
When running training on a cluster with multiple nodes, the memory profiling and training commands need to be run on all nodes in the cluster. You will need to add and configure the following flags to torchrun: --nnodes, --node_rank, --master_addr, and --master_port


## Citation

If you use Cephalo in your research, please cite our paper:

```
@inproceedings{guo2025cephalo,
  author       = {Guo, Runsheng and Anand, Utkarsh and Chen, Arthur and Daudjee, Khuzaima},
  title        = {Cephalo: Harnessing Heterogeneous GPU Clusters for Training Transformer Models},
  booktitle    = {Proceedings of the 2025 International Conference on Supercomputing},
  series       = {ICS ’25},
  location     = {Salt Lake City, UT, USA},
  year         = {2025},
  publisher    = {Association for Computing Machinery},
  address      = {New York, NY, USA},
  isbn         = {979-8-4007-1537-2/25/06},
  doi          = {10.1145/3721145.3730418},
  articleno    = {58},
  numpages     = {16},
}
```

