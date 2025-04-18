import torch
from torch import nn
from torch.jit import Final

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.cnn import Conv2dBlock, DepthwiseSeparable2d
from torchvision.ops import StochasticDepth
from torch import Tensor
from typing import Optional

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def freeze(layer):
    for w in layer.parameters():
        w.requires_grad = False
    return layer

class SDTransformerEncoder(nn.TransformerEncoder):
    """ Implements a Transformer encoder architecture with batch or row
        stochastic depth regularization.

        when mode == 'row', the implementation boils down to sampling a binary
        mask from a bernoulli distribution, which is multiplied with each
        layer's output

        when mode == 'batch', a single bernoulli variable is sampled for each
        layer and with probability `p` it is skipped from execution (for the
        entire training batch)
    """
    def __init__(self, *args, p, mode, **kwargs):
        super(SDTransformerEncoder, self).__init__(*args, **kwargs)
        self.p = p
        self.mode = mode
        self.stochastic_depth = StochasticDepth(p, mode)

    def forward(self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src
        for layer in self.layers:
            res = output

            if (not self.training) or self.mode == 'row' or self.p <= 0.0 or (self.mode == 'batch' and torch.any(torch.rand(1) > self.p)):
                output = layer(output,
                        src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask)

            if self.mode == 'row' and self.p > 0.0:
                output = self.stochastic_depth(output)
                if self.norm is not None:
                    output = self.norm(output)
                output = output + res

            if self.norm is not None:
                output = self.norm(output)

        return output


class PatchMerging(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim, original_height=8, original_width=8, mode='1d'):
        super(PatchMerging, self).__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.h = original_height
        self.w = original_width
        self.mode = mode

        assert mode in ['1d', '2d', 'cnn'], 'mode must be either `1d`, `2d` or `cnn`'

        old_p = patch_size // 2

        # TODO: test with no normalization or activation
        # TODO: test depthwise separable
        if mode == 'cnn':
            self.patch_merging = nn.Sequential(
                    Rearrange('b (h w) (d) -> b d h w', h=8 // old_p, w=8 // old_p),
                    DepthwiseSeparable2d(in_dim, out_dim, kernel_size=2, stride=1, dilation=4, padding=0, normalize=False, activation=None),
                    # Conv2dBlock(in_dim, out_dim, kernel_size=2, stride=1, dilation=4, padding=0, normalize=False, activation=None),
                    Rearrange('b d h w -> b (h w) d')
            )

        elif mode == '2d':
            self.patch_merging = nn.Sequential(
                    Rearrange('b (h w) (d) -> b d h w', h=8 // old_p, w=8 // old_p),
                    Rearrange('b d (h p1) (w p2) -> b (h w) (p1 p2 d)', p1 = 2, p2 = 2)
            )
        else:
            self.patch_merging = Rearrange('b (p1 p2 n) d -> b n (p1 p2 d)', p1 = 2, p2 = 2)

        # TODO: use nn.Bilinear to also project aux tokens
        if mode != 'cnn':
            self.projection = nn.Linear(in_dim * 4, out_dim)
        self.aux_projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        aux = x[:, -1:, :]
        x = x[:, :-1, :]

        b, n, _ = x.shape
        x = self.patch_merging(x)

        assert x.shape[1] == n // 4

        if self.mode != 'cnn':
            x = self.projection(x)
        aux = self.aux_projection(aux)

        x = torch.cat((aux, x), dim=1)
        return x


class ViT(nn.Module):
    pool: Final[str]

    def __init__(self, *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads, 
            mlp_dim,
            hierarchical = False,
            merging_blocks = 0,
            merging_strategy = '1d',
            stages_depth = [],
            stochastic_depth_p = 0.0,
            stochastic_depth_mode = 'row',
            pool = 'cls',
            channels = 3,
            random_patch_projection = True,
            dim_head = 64,
            dropout = 0.0,
            emb_dropout = 0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.hierarchical = hierarchical
        self.merging_blocks = merging_blocks
        self.stochastic_depth_p = stochastic_depth_p
        self.stochastic_depth_mode = stochastic_depth_mode
        self.pool = pool

        # TODO: error messages
        assert merging_blocks == 0 or hierarchical
        assert not hierarchical or merging_blocks > 0
        assert merging_blocks == 0 or len(stages_depth) > 0

        if hierarchical:
            assert len(stages_depth) == merging_blocks + 1

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim)
        )
        if random_patch_projection:
            freeze(self.to_patch_embedding)

        self.to_aux_embedding = nn.Linear(3, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
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
                num_layers=stages_depth[0] if hierarchical else depth,
                norm=nn.LayerNorm(dim)
                )

        out_dim=dim
        merging_layers = []
        if hierarchical:
            for l in range(merging_blocks):
                in_dim = out_dim
                out_dim *= 2
                k = stages_depth[l + 1]
                merge = nn.Sequential(
                        PatchMerging(patch_size*(2**(l+1)), in_dim, out_dim, mode=merging_strategy),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(
                                d_model=out_dim,
                                nhead=heads,
                                # dim_feedforward=(out_dim * 4),
                                dim_feedforward=mlp_dim,
                                dropout=dropout,
                                activation='gelu',
                                norm_first=True,
                                batch_first=True
                                ),
                            num_layers=k,
                            norm=nn.LayerNorm(out_dim)
                            )
                        )
                merging_layers.append(merge)

        self.stages_stack = nn.ModuleList(merging_layers)
        self.mlp_head = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, num_classes)
        )

    def forward(self, img, aux):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        aux = self.to_aux_embedding(aux)
        aux = aux[:, None, :] # Add dummy (n) dimension

        x = torch.cat((aux, x), dim=1)

        # TODO: make ViT pos embedding optional
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        # Hierarchical (pooling) stages
        for stage in self.stages_stack:
            x = stage(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

# TODO: try other normalization layers
class CNN(nn.Module):
    residual: Final[bool]

    def __init__(self, in_channels, out_channels, layers=4, kernel_size=2, residual=True, pool=True, depthwise=False, squeeze=False):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.activation = nn.ELU()

        block = DepthwiseSeparable2d if depthwise else Conv2dBlock
        
        conv_layers = []
        for i in range(layers):
            # pad = 'same' # TODO: causes problem with mobile optimizations
            pad = (1,1) if kernel_size > 2 else (0, 0) # TODO: causes problem with mobile optimizations
            # Input layer
            if i == 0:
                conv_layers.append(block(in_channels, out_channels, kernel_size=kernel_size, activation=None, padding=pad, se_layer=squeeze))
            # Output/Pooling layer
            elif i == layers - 1:
                conv_layers.append(block(out_channels, out_channels, kernel_size=kernel_size, activation=None, padding=pad, pool=pool, se_layer=squeeze))
            # Intermediate layer
            else:
                conv_layers.append(block(out_channels, out_channels, kernel_size=kernel_size, activation=None, padding=pad, se_layer=squeeze))

        self.conv_stack = nn.ModuleList(conv_layers)

    def forward(self, x):
        y = x
        for layer in self.conv_stack:
            y = layer(y)
            y = self.activation(y)
            # TODO: try 1x1 conv projection to match the shape
            if self.residual and x.shape == y.shape:
                y = y + x
            x = y
        return y

class BitboardTransformer(nn.Module):
    cnn_projection: Final[bool]
    material_head: Final[bool]
    channel_pos_encoding: Final[bool]

    def __init__(self,
                 cnn_projection=True,
                 cnn_out_channels=128,
                 cnn_layers=4,
                 cnn_kernel_size=2,
                 cnn_residual=True,
                 cnn_pool=True,
                 cnn_depthwise=False,
                 cnn_squeeze=False,
                 hierarchical=False,
                 hierarchical_blocks=1,
                 stages_depth=[],
                 merging_strategy='1d',
                 stochastic_depth_p=0.0,
                 stochastic_depth_mode='row',
                 material_head=False,
                 patch_size=2,
                 dim=64,
                 depth=12,
                 heads=32,
                 mlp_dim=256,
                 random_patch_projection=True,
                 channel_pos_encoding=False,
                 learned_pos_encoding=False,
                 dropout=0.1,
                 emb_dropout=0.0):
        super(BitboardTransformer, self).__init__()
        self.cnn_projection = cnn_projection
        self.cnn_pool = cnn_pool
        self.material_head = material_head
        self.hierarchical = hierarchical
        self.hierarchical_blocks = hierarchical_blocks
        self.stochastic_depth_p=stochastic_depth_p
        self.stochastic_depth_mode=stochastic_depth_mode
        self.channel_pos_encoding = channel_pos_encoding
        self.learned_pos_encoding = learned_pos_encoding
        self.in_channels = 12

        cnn_out_dim = 4 if self.cnn_pool else 8
        vit_channels = cnn_out_channels if self.cnn_projection else self.in_channels + self.channel_pos_encoding

        if self.learned_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(8, 8))

        # TODO: try upsampling convolutions (deconvolutions).
        # this might allow the use of augmentation strategies
        # and more aggressive probabilistic regularizations (like spatial dropout)
        if self.cnn_projection:
            assert cnn_out_channels, 'You must specify the number of CNN output channels'
            self.cnn = CNN(
                        in_channels=self.in_channels + self.channel_pos_encoding,
                        out_channels=cnn_out_channels,
                        layers=cnn_layers,
                        kernel_size=cnn_kernel_size,
                        residual=cnn_residual,
                        pool=cnn_pool,
                        depthwise=cnn_depthwise,
                        squeeze=cnn_squeeze
                        )
        self.vit = ViT(
                    image_size=cnn_out_dim,
                    patch_size=patch_size,
                    num_classes=1,
                    hierarchical=hierarchical,
                    merging_blocks=hierarchical_blocks,
                    stages_depth=stages_depth,
                    merging_strategy=merging_strategy,
                    stochastic_depth_p=stochastic_depth_p,
                    stochastic_depth_mode=stochastic_depth_mode,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    channels=vit_channels,
                    random_patch_projection=random_patch_projection,
                    pool='mean',
                    dropout=dropout,
                    emb_dropout=emb_dropout
                    )

        # self.material_mlp = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(12*8*8, 1024),
        #             nn.ELU(),
        #             nn.Linear(1024, 1)
        #             )
        if self.material_head:
            self.material_mlp = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(12*8*8, 1)
                        )

    def _pos_encoding(self, x, height:int = 8, width:int = 8, channels:int = 12, scale:float = 1.0, learned:bool =True):
        """ Encodes positional information to the input as an additional channel """
        device = x.device

        if learned:
            pe = self.pos_embedding
        else:
            pe = ((torch.arange(height*width, device=device) + 1) * scale).reshape(height, width)

        pe = pe.expand(x.shape[0], 1, height, width)
        return torch.cat((x, pe), dim=1)

    def forward(self, x, aux):
        if self.material_head:
            material = self.material_mlp(x)

        if self.channel_pos_encoding:
            x = self._pos_encoding(x, scale=0.01, learned=self.learned_pos_encoding, channels=self.in_channels)

        if self.cnn_projection:
            x = self.cnn(x)

        x = self.vit(x, aux)

        if self.material_head:
            x = x + material

        return x
