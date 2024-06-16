import torch

def track_gpu_memory():
    if not torch.cuda.is_available():
        return []

    gpu_memory_info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        free_memory = total_memory - reserved

        gpu_memory_info.append({
            'GPU': i,
            'Allocated Memory (MB)': allocated / 1024**2,
            'Reserved Memory (MB)': reserved / 1024**2,
            'Free Memory (MB)': free_memory / 1024**2,
            'Total Memory (MB)': total_memory / 1024**2,
        })

    return gpu_memory_info