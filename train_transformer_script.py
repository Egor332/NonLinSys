from SMILESTokenizer import SMILESTokenizer
from HIVMoleculeTransformer import HIVMoleculeTransformer
from  HIVDataset import HIVDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 64
learning_rate = 0.0001
epochs = 10
max_len = 100
embedding_dim = 128

print("Loading data...")
df = pd.read_csv('dataset\HIV_train.csv')
df_val = pd.read_csv('dataset\HIV_validation.csv')

tokenizer = SMILESTokenizer()

print("Building vocabulary...")
tokenizer.build_vocabulary(df["smiles"].tolist())
vocab_size = len(tokenizer.stoi)
pad_idx = tokenizer.stoi[tokenizer.pad_token]

dataset = HIVDataset(df, tokenizer, max_len=max_len)
dataset_val = HIVDataset(df_val, tokenizer, max_len=max_len)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

NUM_HEADS = 4
NUM_LAYERS = 3

model = HIVMoleculeTransformer(
vocab_size=vocab_size,
    embed_dim=embedding_dim,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=max_len,
    num_classes=2,
    pad_idx=pad_idx
).to(device)

targets = df["HIV_active"].values
neg_count = len(targets) - sum(targets)
pos_count = sum(targets)
if pos_count > 0:
    pos_weight = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float).to(device)
else:
    pos_weight = None

criterion = nn.CrossEntropyLoss(weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, leave=True)

    for batch_idx, (inputs, labels) in enumerate(loop):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"End of Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

print("Saving model...")


checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_stoi': tokenizer.stoi,
    'vocab_itos': tokenizer.itos,
    'config': {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'max_len': max_len,
        'num_classes': 2,
        'pad_idx': pad_idx
    }
}

torch.save(checkpoint, 'models\hiv_transformer1_checkpoint.pth')
print("Model saved to 'hiv_transformer_checkpoint.pth'")



