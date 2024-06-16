# PytorchParallel
Learning How to Train Pytorch in Parallel Manner

1. normal : Without Parallelism
2. dp : Using Pytorch's DataParallel, split the batched data yielded from the dataloader to several GPUs (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
3. mp : Using Pytorch's Model Parallelism by scattering model params to several devices (https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)