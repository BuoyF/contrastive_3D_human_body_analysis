import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from tqdm import tqdm

from src.datasets import ObstetricDataset
from src.hybrid_encoder import HybridMeshTransformerEncoder, load_encoder_ckpt_stripping_module


class ObstetricMultitaskModel(nn.Module):
    def __init__(self, latent_dim=128, demo_dim=8):
        super().__init__()
        self.encoder = HybridMeshTransformerEncoder(input_dim=5, latent_dim=latent_dim)
        self.fuse = nn.Linear(latent_dim + demo_dim, 64)
        self.delivery_head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.hc_head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, verts, demo):
        z = self.encoder(verts)
        x = self.fuse(torch.cat([z, demo], dim=1))
        return self.delivery_head(x).squeeze(1), self.hc_head(x).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_metrics", default="obstetric_metrics.csv")
    args = ap.parse_args()

    ds = ObstetricDataset(args.csv)
    idx = list(range(len(ds)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)

    tr_dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObstetricMultitaskModel().to(device)

    load_encoder_ckpt_stripping_module(model.encoder, args.ckpt)
    for p in model.encoder.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for verts, demo, delivery, hc in tqdm(tr_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            verts, demo = verts.to(device), demo.to(device)
            delivery, hc = delivery.to(device), hc.to(device)

            out_d, out_hc = model(verts, demo)
            loss = loss_bce(out_d, delivery) + loss_mse(out_hc, hc)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Train Loss: {total / max(1, len(tr_dl)):.4f}")

    # ---- eval
    model.eval()
    all_prob, all_bin, all_gt = [], [], []
    all_hc_pred, all_hc_gt = [], []
    with torch.no_grad():
        for verts, demo, delivery, hc in va_dl:
            verts, demo = verts.to(device), demo.to(device)
            out_d, out_hc = model(verts, demo)

            prob = torch.sigmoid(out_d).cpu().numpy()
            all_prob.extend(prob.tolist())
            all_bin.extend((prob >= 0.5).astype(int).tolist())
            all_gt.extend(delivery.numpy().astype(int).tolist())

            all_hc_pred.extend(out_hc.cpu().numpy().tolist())
            all_hc_gt.extend(hc.numpy().tolist())

    # classification metrics
    acc = accuracy_score(all_gt, all_bin)
    prec = precision_score(all_gt, all_bin, zero_division=0)
    rec = recall_score(all_gt, all_bin, zero_division=0)
    f1 = f1_score(all_gt, all_bin, zero_division=0)
    auc = roc_auc_score(all_gt, all_prob) if len(set(all_gt)) > 1 else float("nan")

    # regression metrics
    hc_mae = float(np.mean(np.abs(np.array(all_hc_pred) - np.array(all_hc_gt))))
    hc_rmse = float(np.sqrt(mean_squared_error(all_hc_gt, all_hc_pred)))

    out = pd.DataFrame([{
        "task": "C-section classification",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }, {
        "task": "Head circumference regression",
        "mae": hc_mae,
        "rmse": hc_rmse,
    }])
    out.to_csv(args.out_metrics, index=False)
    print(f"Saved metrics to {args.out_metrics}")
    print(out)


if __name__ == "__main__":
    main()
