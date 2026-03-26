import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F


class StutterDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.labels = [
            "Prolongation",
            "Block",
            "SoundRep",
            "WordRep",
            "Interjection"
        ]

        # Audio config
        self.sample_rate = 16000
        self.target_length = 94  # fixed time dimension (important)

        # Mel Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        # Resampler (initialized once for efficiency)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_path = row["file_path"]

        # =============================
        # LOAD AUDIO
        # =============================
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # =============================
        # MEL SPECTROGRAM
        # =============================
        mel = self.mel_transform(waveform)

        # Log scale
        mel = torch.log(mel + 1e-9)

        # =============================
        # FIX LENGTH (PAD / TRIM)
        # =============================
        if mel.shape[2] < self.target_length:
            pad_size = self.target_length - mel.shape[2]
            mel = F.pad(mel, (0, pad_size))
        else:
            mel = mel[:, :, :self.target_length]

        # =============================
        # LABELS (MULTI-HOT)
        # =============================
        label = torch.tensor(
            [row[l] for l in self.labels],
            dtype=torch.float32
        )

        return mel, label