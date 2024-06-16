import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm import tqdm
from gpu import track_gpu_memory
import time

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to("cuda:0")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True).to("cuda:1")
        self.fc = nn.Linear(hidden_dim * 2, vocab_size).to("cuda:1")

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x.to("cuda:0"))
        lstm_out, _ = self.lstm(embedded.to("cuda:1"))
        logits = self.fc(lstm_out.to("cuda:1"))
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
start = time.time()
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
    print("GPU Info :", track_gpu_memory())
end = time.time()
print(f"Training for {num_epochs} done in {end - start} s")