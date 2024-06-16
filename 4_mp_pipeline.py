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
train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, split_size=4):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to("cuda:0")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True).to("cuda:1")
        self.fc = nn.Linear(hidden_dim * 2, vocab_size).to("cuda:1")
        self.split_size = split_size

    def forward(self, x):
        x = x.long()
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.embedding(s_next.to("cuda:0"))
        ret = []

        for s_next in splits:
            # A. ``s_prev`` runs on ``cuda:1``
            s_prev = self.lstm(s_prev.to("cuda:1"))
            ret.append(self.fc(s_prev.to("cuda:1")))

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.embedding(s_next.to("cuda:0"))

        logits = torch.cat(ret)
        return logits

# Initialize the model, loss function, and optimizer
vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512
num_layers = 2

model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
device = torch.device('cuda:1')
# model.to(device)

num_epochs = 3
all_gpu_info = []
start = time.time()
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
    gpu_info = track_gpu_memory()
    print("GPU Info :", gpu_info)
    all_gpu_info.append(gpu_info)
end = time.time()
print(f"Training for {num_epochs} done in {end - start} s")

os.makedirs("./gpu_info", exist_ok=True)
with open("./gpu_info/mp_pipeline.json", 'w') as fp:
    json.dump({
        "time" : end - start,
        "gpu_info" : all_gpu_info
    }, fp)