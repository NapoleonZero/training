import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.cnn import Conv2dBlock

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

class CNN(nn.Module):
    # TODO: try GELU and kernels > 2
    # TODO: add residual connections
    def __init__(self, in_channels, out_channels, layers=4, kernel_size=2, residual=True, pool=True):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        
        conv_layers = []
        for i in range(layers):
            # Input layer
            if i == 0:
                conv_layers.append(Conv2dBlock(in_channels, out_channels, kernel_size=kernel_size))
            # Output/Pooling layer
            elif i == layers - 1:
                conv_layers.append(Conv2dBlock(out_channels, out_channels, kernel_size=kernel_size, pool=pool))
            # Intermediate layer
            else:
                conv_layers.append(Conv2dBlock(out_channels, out_channels, kernel_size=kernel_size))

        # self.conv_stack = nn.Sequential(*conv_layers)
        self.conv_stack = nn.ModuleList(conv_layers)

    def forward(self, x):
        y = x
        for layer in self.conv_stack:
            y = layer(y)
            if self.residual and x.shape == y.shape:
                y = y + x
            x = y
        return y

        # return self.conv_stack(x)

#TODO: try CNN residual connections
class BitboardTransformer(nn.Module):
    def __init__(self,
                 cnn_projection=True,
                 cnn_out_channels=128,
                 cnn_layers=4,
                 cnn_kernel_size=2,
                 cnn_residual=True,
                 cnn_pool=True,
                 patch_size=2,
                 dim=64,
                 depth=12,
                 heads=32,
                 mlp_dim=256,
                 dropout=0.1,
                 emb_dropout=0.0):
        super(BitboardTransformer, self).__init__()
        self.cnn_projection = cnn_projection

        cnn_out_dim = 4
        vit_channels = cnn_out_channels if self.cnn_projection else 12

        if self.cnn_projection:
            assert cnn_out_channels, 'You must specify the number of CNN output channels'
            self.cnn = CNN(
                        in_channels=12,
                        out_channels=cnn_out_channels,
                        layers=cnn_layers,
                        kernel_size=cnn_kernel_size,
                        residual=cnn_residual,
                        pool=cnn_pool
                        )
        self.vit = ViT(
                    image_size=cnn_out_dim,
                    patch_size=patch_size,
                    num_classes=1,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    channels=vit_channels,
                    pool='mean',
                    dropout=dropout,
                    emb_dropout=emb_dropout
                    )

    def forward(self, x, aux):
        if self.cnn_projection:
            x = self.cnn(x)
        return self.vit(x, aux)
