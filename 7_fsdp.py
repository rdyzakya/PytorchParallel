import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm import tqdm
from gpu import track_gpu_memory
import time
import json
import os

if "TRANSFORMERS_CACHE" in os.environ.keys():
    del os.environ["TRANSFORMERS_CACHE"]

# IMPORT NECESSARY LIBRARIES
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy, # default_auto_wrap_policy for earlier version before v1.12
    enable_wrap,
    wrap,
)
import functools

# DISTRIBUTED TRAINING SETUP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    # SETUP
    setup(rank, world_size)
    # Load the dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    dataset = dataset['train'].select(range(1000))  # Use only 100 texts

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token ID to EOS token ID

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Use the default DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create dataloaders
    train_dataloader = DataLoader(tokenized_dataset, 
                                  batch_size=8, 
                                  shuffle=False,
                                  collate_fn=data_collator,
                                  pin_memory=True, 
                                  sampler=DistributedSampler(tokenized_dataset),
                                  num_workers=2)

    # Define the BiLSTM model
    class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
            super(BiLSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        def forward(self, x):
            x = x.long()
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            logits = self.fc(lstm_out)
            return logits
        
        @property
        def device(self):
            devices = {param.device for param in self.parameters()}
            return list(devices)

    # Initialize the model, loss function, and optimizer
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=2000
    )
    # torch.cuda.set_device(rank)

    model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    ## UPDATED : WRAP
    model = model.to(rank)
    model = FSDP(model,
                 auto_wrap_policy=my_auto_wrap_policy)
                # cpu_offload=CPUOffload(offload_params=True))
    print("My model device is :", model.device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1)

    # Training function
    def train(model, dataloader, criterion, optimizer, device):
        model.train()
        ## PREVIOUSLY
        # total_loss = 0
        ## UPDATED
        ddp_loss = torch.zeros(2).to(device)
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            ## PREVIOUSLY
            # total_loss += loss.item()
            ## UPDATED
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(input_ids)
        
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        ## PREVIOUS
        # return total_loss / len(dataloader)
        ## UPDATED
        return ddp_loss[0] / ddp_loss[1]

    # Training loop
    ## PREVIOUS
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    num_epochs = 3
    all_gpu_info = []
    start = time.time()
    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch) # SET DATA LOADER EPOCH
        train_loss = train(model, train_dataloader, criterion, optimizer, device=rank)
        scheduler.step()
        gpu_info = track_gpu_memory()
        all_gpu_info.append(gpu_info)
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
            print("GPU Info :", gpu_info)
    end = time.time()
    if rank == 0:
        print(f"Training for {num_epochs} done in {end - start} s")

    os.makedirs("./gpu_info", exist_ok=True)
    with open("./gpu_info/fsdp.json", 'w') as fp:
        json.dump({
            "time" : end - start,
            "gpu_info" : all_gpu_info
        }, fp)

    # if rank == 0:
    #     dist.barrier() --> BUG STUCK, SEMAPHORE APALAH
    #     states = model.state_dict()
    #     if rank == 0:
    #         os.makedirs("./model", exist_ok=True)
    #         torch.save(states, "./model/model.pt")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() # GET THE WORLD SIZE
    mp.spawn(main, args=(world_size,), nprocs=world_size) # SPAWN THE PROCESS