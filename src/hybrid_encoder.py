import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x, _ = self.attn(x, x, x)
        x = self.norm(x + residual)
        return self.norm(self.ffn(x) + x)


class LocalPointTransformer(nn.Module):
    """
    Matches your pretraining implementation:
    - KNN on raw 5D input (pos = x[..., :5])
    - pos_encoder: Linear(5 -> dim)
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.pos_encoder = nn.Linear(5, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def knn(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, x)
        return dist.topk(self.k, dim=-1, largest=False)[1]

    def index_points(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x[torch.arange(B, device=x.device).view(B, 1, 1), idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        pos = x[..., :5]
        idx = self.knn(pos)
        x_knn = self.index_points(x, idx)  # (B, N, k, D)

        q = self.q(x).unsqueeze(2)         # (B, N, 1, D)
        k, v = self.kv(x_knn).chunk(2, dim=-1)

        pos_proj = self.pos_encoder(pos)               # (B, N, D)
        pos_proj_knn = self.index_points(pos_proj, idx)
        rel = self.pos_mlp(pos_proj.unsqueeze(2) - pos_proj_knn)

        attn = self.attn_mlp(q - k + rel)
        attn = F.softmax(attn, dim=2)
        agg = torch.sum(attn * (v + rel), dim=2)
        return agg


class HybridMeshTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int = 5, latent_dim: int = 128, depth: int = 4, k: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:
                self.blocks.append(LocalPointTransformer(latent_dim, k))
            else:
                self.blocks.append(GlobalSelfAttentionBlock(latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(verts)  # (B, N, latent)
        for blk in self.blocks:
            if isinstance(blk, LocalPointTransformer):
                x = x + blk(x)
            else:
                x = blk(x)
        x = self.norm(x).transpose(1, 2)  # (B, latent, N)
        z = self.pool(x).squeeze(-1)      # (B, latent)
        return F.normalize(z, dim=1)


def load_encoder_ckpt_stripping_module(encoder: nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    encoder.load_state_dict(sd, strict=True)
