import argparse
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from src.mesh_features import compute_vertex_normals, compute_mean_curvature, compute_radial_distance, sample_or_pad
from src.hybrid_encoder import HybridMeshTransformerEncoder


class GeoMeshDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, n_points: int = 12500):
        self.mesh_files = []
        for dp, _, fnames in os.walk(root_dir):
            for f in fnames:
                if f.lower().endswith(".obj"):
                    self.mesh_files.append(os.path.join(dp, f))
        self.transform = transform
        self.n_points = n_points

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh = trimesh.load(self.mesh_files[idx], process=True)
        normals = compute_vertex_normals(mesh)
        curvature = compute_mean_curvature(mesh)
        radial = compute_radial_distance(mesh)
        feat = np.concatenate([normals, curvature, radial], axis=1).astype(np.float32)
        feat = sample_or_pad(feat, n_points=self.n_points, seed=idx)
        feat = torch.tensor(feat, dtype=torch.float32)

        if self.transform:
            return self.transform(feat.clone()), self.transform(feat.clone())
        return feat, feat


def geo_augment_fn(v: torch.Tensor) -> torch.Tensor:
    angle = np.random.uniform(0, 2 * np.pi)
    rot = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0, 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=torch.float32)
    return v @ rot.T


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z.device)
    return F.cross_entropy(sim, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Root folder of unlabeled .obj meshes")
    ap.add_argument("--out_ckpt", default="mesh_combine.pth")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_points", type=int, default=12500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GeoMeshDataset(args.root_dir, transform=geo_augment_fn, n_points=args.n_points)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = HybridMeshTransformerEncoder(input_dim=5).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for v1, v2 in tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}"):
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch} Loss: {total / max(1, len(dl)):.4f}")

    torch.save(model.state_dict(), args.out_ckpt)
    print(f" Saved encoder to {args.out_ckpt}")


if __name__ == "__main__":
    main()
