import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd

from stutter_dataset import StutterDataset
from stutter_model import StutterCNN

# =============================
# CONFIG
# =============================
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# =============================
# LOAD DATA
# =============================
dataset = StutterDataset("Data_downloading/dataset/final_dataset.csv")

# Train / Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =============================
# CLASS WEIGHTS (IMPORTANT)
# =============================
df = pd.read_csv("Data_downloading/dataset/final_dataset.csv")

label_cols = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]

pos_counts = df[label_cols].sum()
neg_counts = len(df) - pos_counts

pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

# =============================
# MODEL
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = StutterCNN().to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =============================
# TRAIN LOOP
# =============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for mel, labels in train_loader:
        mel, labels = mel.to(device), labels.to(device)

        outputs = model(mel)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", total_loss / len(train_loader))

    # =============================
    # VALIDATION
    # =============================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for mel, labels in val_loader:
            mel, labels = mel.to(device), labels.to(device)

            outputs = model(mel)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    print("Val Loss:", val_loss / len(val_loader))