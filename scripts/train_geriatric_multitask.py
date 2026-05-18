import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datasets import GeriatricDataset
from src.hybrid_encoder import HybridMeshTransformerEncoder, load_encoder_ckpt_stripping_module
from src.metrics import regression_metrics


class GeriatricMultitaskModel(nn.Module):
    def __init__(self, out_dim: int, latent_dim=128, demo_dim=5):
        super().__init__()
        self.encoder = HybridMeshTransformerEncoder(input_dim=5, latent_dim=latent_dim)
        self.fuse = nn.Linear(latent_dim + demo_dim, 128)
        self.shared = nn.Sequential(nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, out_dim)

    def forward(self, verts, demo):
        z = self.encoder(verts)
        x = self.fuse(torch.cat([z, demo], dim=1))
        x = self.shared(x)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_csv", default="geriatric_metrics.csv")
    args = ap.parse_args()

    # default target list (edit freely)
    target_cols = [
        "Average_MVC_kg", "Total_Bone_Mass", "Total_Lean_Mass", "Total_Fat_Free_Mass",
        "Right_60 deg EXT Peak Torque                               (Nm)",
        "Left_60 deg EXT Peak Torque                               (Nm)",
        "Right_120 deg EXT Peak Torque                               (Nm)",
        "Left_120 deg EXT Peak Torque                               (Nm)",
        "Right_60 deg FLEX Peak Torque                               (Nm)",
        "Left_60 deg FLEX Peak Torque                               (Nm)",
        "Right_120 deg FLEX Peak Torque                               (Nm)",
        "Left_120 deg FLEX Peak Torque                               (Nm)",
        "Right_60 deg EXT AVG. Power                                (W)",
        "Left_60 deg EXT AVG. Power                                        (W)",
        "Right_60 deg FLEX AVG. Power                                (W)",
        "Left_60 deg FLEX AVG. Power                                        (W)",
        "Right_120 deg EXT AVG. Power                                (W)",
        "Left_120 deg EXT AVG. Power                                        (W)",
        "Right_120 deg FLEX AVG. Power                                (W)",
        "Left_120 deg FLEX AVG. Power                                        (W)",
    ]

    ds = GeriatricDataset(args.csv, target_cols=target_cols)
    idx = list(range(len(ds)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)

    tr_dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeriatricMultitaskModel(out_dim=len(target_cols)).to(device)

    load_encoder_ckpt_stripping_module(model.encoder, args.ckpt)
    for p in model.encoder.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for verts, demo, targets in tqdm(tr_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            verts, demo, targets = verts.to(device), demo.to(device), targets.to(device)
            pred = model(verts, demo)
            loss = loss_fn(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Train Loss: {total / max(1, len(tr_dl)):.4f}")

    # ---- eval: denormalize per-target, compute metrics
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for verts, demo, targets in va_dl:
            verts, demo = verts.to(device), demo.to(device)
            pred = model(verts, demo).cpu().numpy()
            preds.append(pred)
            gts.append(targets.numpy())

    preds = np.vstack(preds)
    gts = np.vstack(gts)

    rows = []
    for i, name in enumerate(ds.target_cols):
        pred_denorm = preds[:, i] * ds.stds[name] + ds.means[name]
        gt_denorm = gts[:, i] * ds.stds[name] + ds.means[name]
        rmse, mae, r2, pr, pv = regression_metrics(gt_denorm, pred_denorm)
        rows.append({
            "Metric": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson": pr,
            "P-Value": pv,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
