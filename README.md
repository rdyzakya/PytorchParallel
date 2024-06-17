# PytorchParallel
Learning How to Train Pytorch in Parallel Manner

## Resource Link
1. https://pytorch.org/tutorials/beginner/dist_overview.html
2. https://www.youtube.com/watch?v=gXDsVcY8TXQ
3. https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj (DDP Tutorial)
4. https://youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&feature=shared (FSDP TUTORIAL)

## Scripts Overview
1. normal : Without Parallelism
2. dp : Using Pytorch's DataParallel, split the batched data yielded from the dataloader to several GPUs (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
3. mp : Using Pytorch's Model Parallelism by scattering model params to several devices (https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
4. mp_pipeline : Using Model Parallelism + Input Pipelining
5. ddp : Using Pytorch's Distributed Data Parallel, the model is replicated to all devices and the data were scattered all over the devices (https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py)
6. ddp_torchrun : Using DDP but with fault tolerance (saving training snapshots)
7. fsdp : Using Fully Sharded Data Parallel when the model is not fit in to single GPU, similar to mp but more advance, use sharding (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
8. accelerate