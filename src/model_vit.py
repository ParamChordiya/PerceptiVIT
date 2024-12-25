# src/model_vit.py
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Splits image into patches, then flattens and projects to embedding dimension.
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g., 256/16=16 => 16x16 patches

        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, 3, H, W]
        x = self.proj(x)  
        # shape: [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  
        # shape: [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  
        # shape: [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # x: [B, N, C], where N = number of patches

        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)

        # q, k, v: [B, num_heads, N, head_dim]

        # Scaled Dot-Product
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = attn_scores.softmax(dim=-1)  # [B, num_heads, N, N]
        attn_probs = self.attn_drop(attn_probs)

        out = (attn_probs @ v).transpose(1,2).reshape(B, N, C)
        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim*mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x -> LN -> MHSA -> residual
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # x -> LN -> MLP -> residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=2,
                 embed_dim=192,  # lowered from 384 for speed
                 depth=4,        # fewer layers for quick tests
                 num_heads=4,    # fewer heads
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pos_embed, std=1e-6)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Expand cls_token for each batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, num_patches+1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)
        # Extract the [CLS] token
        cls_token_final = x[:, 0]  # [B, embed_dim]

        # Classify
        logits = self.fc(cls_token_final)  # [B, num_classes]
        return logits
