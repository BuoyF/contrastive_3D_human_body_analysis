import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mesh_features import mesh_to_feat5


class ObstetricDataset(Dataset):
    """
    Expects CSV with:
    Mesh_Path, Age, Sex, Race, Height, Previous_C_Section, Chronic_Disease, Pregnancy_History, Weight,
    Delivery, head_circumference
    """
    def __init__(self, csv_path: str, n_points: int = 12500, seed: int = 42):
        self.df = pd.read_csv(csv_path).fillna(0.0)
        self.n_points = n_points
        self.seed = seed

        self.rows = []
        for _, r in self.df.iterrows():
            mp = str(r["Mesh_Path"])
            if os.path.exists(mp) and mp.lower().endswith(".obj"):
                self.rows.append(r)

        print(f"Loaded {len(self.rows)} samples from {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        feat = mesh_to_feat5(str(r["Mesh_Path"]), n_points=self.n_points, seed=self.seed + idx)
        verts = torch.tensor(feat, dtype=torch.float32)

        demo = torch.tensor([
            float(r["Age"]),
            float(r["Sex"]),
            float(r["Race"]),
            float(r["Height"]),
            float(r["Previous_C_Section"]),
            float(r["Chronic_Disease"]),
            float(r["Pregnancy_History"]),
            float(r["Weight"]),
        ], dtype=torch.float32)

        delivery = torch.tensor(float(r["Delivery"]), dtype=torch.float32)
        head_circ = torch.tensor(float(r["head_circumference"]), dtype=torch.float32)

        return verts, demo, delivery, head_circ


class GeriatricDataset(Dataset):
    """
    Your geriatric CSV with Mesh_Path and various numeric targets.
    You can edit target_cols as needed.
    """
    def __init__(self, csv_path: str, target_cols: list[str], n_points: int = 12500, seed: int = 42):
        self.df = pd.read_csv(csv_path).fillna(0.0)
        self.target_cols = [c.strip() for c in target_cols]
        self.n_points = n_points
        self.seed = seed

        # numeric cleanup + normalization stats
        self.means = {}
        self.stds = {}
        for c in self.target_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0)
            self.means[c] = float(self.df[c].mean())
            self.stds[c] = float(self.df[c].std() + 1e-6)

        self.rows = []
        for _, r in self.df.iterrows():
            mp = str(r["Mesh_Path"])
            if os.path.exists(mp) and mp.lower().endswith(".obj"):
                self.rows.append(r)

        print(f"Loaded {len(self.rows)} samples from {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        feat = mesh_to_feat5(str(r["Mesh_Path"]), n_points=self.n_points, seed=self.seed + idx)
        verts = torch.tensor(feat, dtype=torch.float32)

        demo = torch.tensor([
            float(r["Age"]),
            float(r["Gender"]),
            float(r["Ethnicity"]),
            float(r["Height"]),
            float(r["Weight"]),
        ], dtype=torch.float32)

        targets = []
        for c in self.target_cols:
            val = float(r[c])
            targets.append((val - self.means[c]) / self.stds[c])
        targets = torch.tensor(targets, dtype=torch.float32)

        return verts, demo, targets
