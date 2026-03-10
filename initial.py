import scipy.io
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoModel
import os
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset

class HubertECGClassifier(nn.Module):
    """
    Single-lead ECG -> HuBERT-ECG (black box) -> mean pooling -> FC -> logits
    Softmax is not applied in forward because CrossEntropyLoss expects logits.
    """
    def __init__(
        self,
    ):
        super().__init__()
        #using pre-trained weights of Hubert-ECG
        self.hubert = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-small",trust_remote_code=True)
        self.hidden_size = self.hubert.config.hidden_size  # typically 512 for hubert-ecg-small
        self.classifier = nn.Linear(self.hidden_size,4)

#     def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None):
#         """
#         input_values: (B, L) float32 waveform
#         labels: (B,) long (optional)
#         """
#
#         out = self.hubert(input_values=input_values)
#
#         # out.last_hidden_state: (B, T, H)
#         x = out.last_hidden_state
#
#         # Mean pooling over time dimension T: (B, H)
#         x = x.mean(dim=1)
#         logits = self.classifier(x)  # (B, num_labels)
#
#         loss = None
#         if labels is not None:
#             loss = nn.CrossEntropyLoss()(logits, labels)
#
#         return {"loss": loss, "logits": logits}
#
# LABEL2ID = {"N": 0, "A": 1, "O": 2, "~": 3}
#
# class PhysioNet2017Dataset(Dataset):
#     def __init__(self, csv_path, ecg_dir, target_len=9000, training=False):
#         self.df = pd.read_csv(csv_path)
#         self.ecg_dir = ecg_dir
#         self.target_len = target_len
#         self.training = training
#
#     def _load_ecg(self, record):
#         mat = scipy.io.loadmat(os.path.join(self.ecg_dir, record + ".mat"))
#         ecg = mat["val"][0].astype(np.float32)   # 1D waveform
#         return ecg
#
#     #what is the purpose of this funciton?
#     def _normalize(self, x, eps=1e-8):
#         return (x - x.mean()) / (x.std() + eps)
#
#     #what does this function do?
#     def _crop_or_pad(self, x):
#         L = len(x)
#         T = self.target_len
#         if L == T:
#             return x
#         if L > T:
#             start = np.random.randint(0, L - T + 1) if self.training else (L - T) // 2
#             return x[start:start+T]
#         # pad
#         return np.pad(x, (0, T - L), mode="constant")
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         record = str(self.df.iloc[idx]["record"])
#         label_str = str(self.df.iloc[idx]["label"])
#
#         ecg = self._load_ecg(record)          # <-- uses record to load .mat
#         ecg = self._normalize(ecg)
#         ecg = self._crop_or_pad(ecg)
#
#         return {
#             "input_values": torch.tensor(ecg, dtype=torch.float32),   # (L,)
#             "labels": torch.tensor(LABEL2ID[label_str], dtype=torch.long)
#         }
#
#
# train_ds = PhysioNet2017Dataset(
#     csv_path = "train_dataset.csv",
#     ecg_dir = "/Users/byeon-useog/desktop/training2017",
#     target_len = 9000,
#     training=True
# )
# train_loader = DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=0)
#
# val_ds = PhysioNet2017Dataset(
#     csv_path="validation_dataset.csv",
#     ecg_dir="/Users/byeon-useog/desktop/training2017",
#     target_len=9000,
#     training=False
# )
# val_loader = DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=0)
#
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert device.type=="cpu"
#
# model = HubertECGClassifier()
# assert model.hidden_size == 512
#
# optimizer = AdamW(model.parameters(), lr=2e-5)
#
# num_epochs = 10
# for epoch in range(num_epochs):
#
#     model.train()
#     total_loss = 0.0
#
#     for batch in train_loader:  # train_loader must yield {"input_values": (B,L), "labels": (B,)}
#         input_values = batch["input_values"].to(device)
#         labels = batch["labels"].to(device)
#
#         out = model(input_values=input_values, labels=labels)
#         loss = out["loss"]
#
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f}")






