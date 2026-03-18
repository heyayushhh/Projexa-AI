import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# ===================== CONFIG =====================
CSV_PATH = "../../Data_downloading/train_dataset.csv"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_LABELS = 7

# ===================== LOAD DATA =====================
df = pd.read_csv(CSV_PATH)

train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"])

# ===================== PROCESSOR =====================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# ===================== DATASET =====================
class StutterDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio, sr = librosa.load(row["path"], sr=16000)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(row["label"])
        }

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    input_values = nn.utils.rnn.pad_sequence(input_values, batch_first=True)

    return {
        "input_values": input_values,
        "labels": labels
    }

train_dataset = StutterDataset(train_df)
val_dataset = StutterDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ===================== MODEL =====================
class StutterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, NUM_LABELS)

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)

        x = self.dropout(pooled)
        logits = self.classifier(x)

        return logits

model = StutterModel().to(DEVICE)

# ===================== CLASS WEIGHTS =====================
counts = df["label"].value_counts().sort_index()
weights = 1.0 / counts
weights = weights / weights.sum()
weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ===================== TRAIN =====================
def train_epoch():
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        input_values = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(input_values)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ===================== VALIDATION =====================
def evaluate():
    model.eval()

    preds = []
    truths = []

    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_values)
            pred = torch.argmax(logits, dim=-1)

            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    print("\n📊 Validation Report:")
    print(classification_report(truths, preds))

# ===================== TRAIN LOOP =====================
best_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch()
    print(f"Train Loss: {train_loss:.4f}")

    evaluate()

    # Save model
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model saved!")

print("\n🎉 Training Complete!")