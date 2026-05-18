import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import trimesh
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from tqdm import tqdm

# ------------------ Utility Functions ------------------

def compute_vertex_normals(mesh):
    return mesh.vertex_normals.astype(np.float32)

def compute_mean_curvature(mesh):
    V = mesh.vertices.shape[0]
    curvatures = np.zeros(V, dtype=np.float32)
    if mesh.face_adjacency is None or mesh.face_adjacency_angles is None:
        return curvatures.reshape(-1, 1)
    for (edge, angle) in zip(mesh.face_adjacency_edges, mesh.face_adjacency_angles):
        i, j = edge
        length = np.linalg.norm(mesh.vertices[i] - mesh.vertices[j])
        cur = 0.25 * length * angle
        curvatures[i] += cur
        curvatures[j] += cur
    return curvatures.reshape(-1, 1)

def compute_radial_distance(mesh):
    centroid = mesh.vertices.mean(axis=0)
    distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
    return distances.reshape(-1, 1).astype(np.float32)

# ------------------ Dataset ------------------

class PregnancyDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path).fillna(0.0)
        self.samples = []
        for _, row in self.data.iterrows():
            mesh_path = row['Mesh_Path']
            if os.path.exists(mesh_path) and mesh_path.endswith('.obj'):
                self.samples.append(row)
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        mesh = trimesh.load(row['Mesh_Path'], process=True)
        normals = compute_vertex_normals(mesh)
        curvature = compute_mean_curvature(mesh)
        radial = compute_radial_distance(mesh)
        feat = np.concatenate([normals, curvature, radial], axis=1)

        if feat.shape[0] > 12500:
            feat = feat[np.random.choice(feat.shape[0], 12500, replace=False)]
        else:
            feat = np.pad(feat, ((0, 12500 - feat.shape[0]), (0, 0)), mode='edge')

        verts = torch.tensor(feat, dtype=torch.float32)

        demo = torch.tensor([
            float(row['Age']),
            float(row['Sex']),
            float(row['Race']),
            float(row['Height']),
            float(row['Previous_C_Section']),
            float(row['Chronic_Disease']),
            float(row['Pregnancy_History']),
            float(row['Weight']),
        ], dtype=torch.float32)

        delivery = torch.tensor(row['Delivery'], dtype=torch.float32)
        head_circ = torch.tensor(row['head_circumference'], dtype=torch.float32)

        return verts, demo, delivery, head_circ

# ------------------ Encoder ------------------

class GlobalSelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        res = x
        x, _ = self.attn(x, x, x)
        x = self.norm(x + res)
        return self.norm(self.ffn(x) + x)

class LocalPointTransformer(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.pos_encoder = nn.Linear(5, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def knn(self, x):
        dist = torch.cdist(x, x)
        return dist.topk(self.k, dim=-1, largest=False)[1]

    def index_points(self, x, idx):
        B = x.shape[0]
        return x[torch.arange(B).view(B, 1, 1), idx]

    def forward(self, x):
        B, N, D = x.shape
        pos = x[..., :5]
        idx = self.knn(pos)
        x_knn = self.index_points(x, idx)
        pos_knn = self.index_points(pos, idx)

        q = self.q(x).unsqueeze(2)
        k, v = self.kv(x_knn).chunk(2, dim=-1)

        pos_proj = self.pos_encoder(pos)
        pos_proj_knn = self.index_points(pos_proj, idx)
        rel = self.pos_mlp(pos_proj.unsqueeze(2) - pos_proj_knn)

        attn = self.attn_mlp(q - k + rel)
        attn = F.softmax(attn, dim=2)
        agg = torch.sum(attn * (v + rel), dim=2)
        return agg

class HybridMeshTransformerEncoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=128, depth=4, k=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList([
            LocalPointTransformer(latent_dim, k) if i % 2 == 0 else GlobalSelfAttentionBlock(latent_dim)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, verts):
        x = self.input_proj(verts)
        for block in self.blocks:
            x = x + block(x) if isinstance(block, LocalPointTransformer) else block(x)
        x = self.norm(x).transpose(1, 2)
        return F.normalize(self.pool(x).squeeze(-1), dim=1)

# ------------------ Multitask Model ------------------

class PregnancyMultitaskModel(nn.Module):
    def __init__(self, latent_dim=128, demo_dim=8):
        super().__init__()
        self.encoder = HybridMeshTransformerEncoder(input_dim=5, latent_dim=latent_dim)
        self.fuse = nn.Linear(latent_dim + demo_dim, 64)
        self.delivery_head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.circ_head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, verts, demo):
        z = self.encoder(verts)
        x = self.fuse(torch.cat([z, demo], dim=1))
        return self.delivery_head(x), self.circ_head(x)

# ------------------ Training Loop ------------------

def train():
    dataset = PregnancyDataset("pregnancy_dataset.csv")
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=4, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PregnancyMultitaskModel().to(device)

    # Load pretrained encoder
    state_dict = torch.load("mesh_combine.pth")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.encoder.load_state_dict(state_dict)

    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for verts, demo, delivery, head_circ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            verts, demo = verts.to(device), demo.to(device)
            delivery, head_circ = delivery.to(device), head_circ.to(device)
            out_delivery, out_circ = model(verts, demo)
            loss = loss_bce(out_delivery.squeeze(), delivery) + loss_mse(out_circ.squeeze(), head_circ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    pred_bin, pred_circ, gt_bin, gt_circ = [], [], [], []
    with torch.no_grad():
        for verts, demo, delivery, head_circ in val_loader:
            verts, demo = verts.to(device), demo.to(device)
            out_d, out_c = model(verts, demo)
            pred_d = torch.sigmoid(out_d).cpu().numpy().squeeze()
            pred_c = out_c.cpu().numpy().squeeze()
            pred_bin.extend((pred_d >= 0.5).astype(int))
            pred_circ.extend(pred_c)
            gt_bin.extend(delivery.numpy())
            gt_circ.extend(head_circ.numpy())

    print("Delivery Classification:")
    print(f"Accuracy:  {accuracy_score(gt_bin, pred_bin):.4f}")
    print(f"Precision: {precision_score(gt_bin, pred_bin):.4f}")
    print(f"Recall:    {recall_score(gt_bin, pred_bin):.4f}")
    print(f"F1 Score:  {f1_score(gt_bin, pred_bin):.4f}")
    print(f"AUC:       {roc_auc_score(gt_bin, pred_bin):.4f}")

    print("📏 Head Circumference Regression:")
    print(f"MAE:  {np.mean(np.abs(np.array(pred_circ) - np.array(gt_circ))):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(gt_circ, pred_circ)):.4f}")

if __name__ == "__main__":
    train()
