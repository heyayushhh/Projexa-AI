import torch
from stutter_dataset import StutterDataset
from model import StutterCNN

dataset = StutterDataset("Data_downloading/dataset/final_dataset.csv")

mel, label = dataset[0]

mel = mel.unsqueeze(0)  # add batch

model = StutterCNN()

output = model(mel)

print("Input shape:", mel.shape)
print("Output shape:", output.shape)
print("Output:", output)