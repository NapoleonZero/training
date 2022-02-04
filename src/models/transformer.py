import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.to_aux_embedding = nn.Linear(3, 3 * dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation='gelu',
                    norm_first=True,
                    batch_first=True
                    ),
                num_layers=depth,
                norm=nn.LayerNorm(dim)
                )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
        )

    def forward(self, img, aux):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        aux = self.to_aux_embedding(aux)
        aux = aux[:, None, :] # Add dummy (n) dimension
        aux = aux.chunk(3, dim = -1) # Split the last dimension in 3 to have embeddings for side, ep, castling
        x = torch.cat((aux[0], x), dim=1)
        x = torch.cat((aux[1], x), dim=1)
        x = torch.cat((aux[2], x), dim=1)
        x += self.pos_embedding[:, :(n + 3)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class BitboardTransformer(nn.Module):
    def __init__(self, patch_size=2, dim=64, depth=12, heads=32, mlp_dim=256, dropout=0.1, emb_dropout=0.0):
        super(BitboardTransformer, self).__init__()
        self.vit = ViT(
                    image_size=8,
                    patch_size=patch_size,
                    num_classes=1,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    channels=12,
                    pool='mean',
                    dropout=dropout,
                    emb_dropout=emb_dropout
                    )

    def forward(self, x, aux):
        return self.vit(x, aux)
