import copy
import math
import torch
from torch import nn
from torch.jit import Final
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.cnn import Conv2dBlock, DepthwiseSeparable2d
from torchvision.ops import StochasticDepth
from torch import Tensor
from typing import Optional, Union

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def freeze(layer):
    for w in layer.parameters():
        w.requires_grad = False
    return layer

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    dim = None
) -> Tensor:
    if not dim:
        dim = k.shape[-1]

    # Batched Query-Value matrix multiplications over the last two dims:
    # the remaining are considered as batch dimensions
    attn = torch.matmul(q, k.transpose(-1, -2))

    # Normalization: we scale by the sqrt of the dimension of each head because
    # QK^T computes, for each head, dot products with vectors of dimension
    # inner_dim. If the vectors were (independent and) randomly
    # distributed with mean 0 and unit variance then the variance of the
    # dot product would be inner_dim. So scaling by the standard
    # deviation is a sound normalization scheme.
    attn = attn / math.sqrt(dim)

    if attention_mask is not None:
        attn = attn + attention_mask

    # Row softmax
    attn = torch.softmax(attn, dim = -1)

    # Hack to prevent softmax from producing `nan`s when entire rows of the
    # activation matrix are "masked" with -inf. This should be better
    # approached with MaskedTensors, but they are still a propotype
    # feature. An alternative approach would be to use
    # torch.finfo(attn.dtype).min as filling value instead of -inf, but this
    # would produce a uniform distribution instead of all zeros. These
    # values are not considered during computation due to column masking,
    # but they might interfere during the last projections.

    # NOTE: disabled because it breaks gradient flow
    # attn = torch.nan_to_num(attn, 0.0)

    # Value multiplication
    attn = torch.matmul(attn, v)

    return attn

def is_initializable(module: nn.Module) -> bool:
    return isinstance(module, tuple([nn.Linear, nn.LayerNorm]))

class Replicated(nn.Module):
    """ Wrapper module that stacks a given Module a number of times.
        The constructor tries to reinitialize the parameters of each copied
        layer by calling `reset_parameters()` for each child module
        recursively.
    """
    def __init__(
        self,
        layer: nn.Module,
        n_layers: int
    ) -> None:
        super().__init__()
        layers = [copy.deepcopy(layer) for i in range(n_layers)]
        self.stacked = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(lambda m: m.reset_parameters() if is_initializable(m) else None)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for layer in self.stacked:
            x = layer(x, *args, **kwargs)
        return x


class MultiHeadAttention(nn.Module):
    """ Multi head attention module.
        Query, key and value vectors share the same dimension `dim`.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        inner_dim: int = None
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.inner_dim = inner_dim

        if not self.inner_dim:
            self.inner_dim = dim // n_heads

        # TODO: elaborate on not using biases (LayerNormalization adds biases
        # or cancels them out by subtracting mean?)
        self.qkv_proj = nn.Linear(dim, self.inner_dim * self.n_heads * 3, bias = False)
        self.out_proj = nn.Linear(self.inner_dim * n_heads, dim, bias = False)

    def _qkv_proj(self, query, key, value):
        if query is key and key is value:
            # Compute projection for query, key, value for all heads in parallel and split it.
            qkv = self.qkv_proj(query).chunk(3, dim=-1) # tuple of 3x(B, N, DIM)
        else:
            # weight.T \in R^{dim \times 3 * inner_dim * n_heads}
            d = self.inner_dim * self.n_heads
            q_proj_weight = self.qkv_proj.weight[0:d, :]
            k_proj_weight = self.qkv_proj.weight[d:2*d, :]
            v_proj_weight = self.qkv_proj.weight[2*d:, :]

            # No biases
            q = F.linear(query, q_proj_weight)
            k = F.linear(key, k_proj_weight)
            v = F.linear(value, v_proj_weight)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads, d = self.inner_dim), qkv)
        return q, k, v

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        causal: bool = False,
        key_mask: Tensor = None,
        query_mask: Tensor = None,
    ) -> Tensor:
        if (key_mask is None) != (query_mask is None):
            raise ValueError("Either both key_mask and query_mask must be None, or both must be provided.")

        q, k, v = self._qkv_proj(query, key, value)

        # fill_value = float('-inf')
        fill_value = torch.finfo(q.dtype).min
        mask = None
        if key_mask is not None:

            assert key_mask.shape[1] == key.shape[1] # must match sequence length
            assert query_mask.shape[1] == query.shape[1] # must match sequence length

            assert key_mask is not None and query_mask is not None

            if key_mask.dtype is not torch.bool:
                key_mask = key_mask.bool()

            if query_mask.dtype is not torch.bool:
                query_mask = query_mask.bool()

            # Add a new dimension at position 1 -> (B, 1, N)
            key_mask = key_mask.unsqueeze(1)
            query_mask = query_mask.unsqueeze(1)

            # The transpose produces the shape -> (B, N, 1)
            # The & operator is broadcasted along dimension 1 for the first
            # operand and along dimension 2 for the second. This replicates the
            # binary mask along the rows for the first operand and along the
            # columns for the second one, which virtually creates two batches
            # of matrices of size (B, N, N) where the second one is the
            # transpose of the first one. By 'logically-and' them together we
            # obtain the correct mask for each sequence in the batch
            mask = key_mask & query_mask.transpose(1, 2)
            mask = torch.where(~mask, fill_value, 0.0)

            # Add new 'heads' dimension for broadcasting -> (B, 1, N, N)
            # the attention matrix is (B, H, N, N) so the mask is broadcasted H
            # times along that dimension
            mask = mask.unsqueeze(1)

        if causal:
            # By masking the elements of the preactivation attention matrix to
            # -inf, the softmax automatically drops them to zero while
            # preserving the sum-to-one constraint. We can use a single
            # attention mask for this since it's shared among every sequence
            # (because of padding they all have the same length)
            n = query.shape[1]
            causal_mask = torch.full((n, n), fill_value, device=query.device).triu(diagonal=1)
            if mask is not None:
                mask = mask + causal_mask
            else:
                mask = causal_mask

        # attn = scaled_dot_product_attention(q, k, v, attention_mask=mask, dim=self.inner_dim)
        attn = F.scaled_dot_product_attention(q, k, v)

        # Concatenate heads
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        # Project back to dim (unnecessary if self.inner_dim * n_heads == dim)
        if self.inner_dim * self.n_heads != self.dim:
            attn = self.out_proj(attn)

        # assert attn.shape[-1] == self.dim
        return attn

class LinformerAttention(MultiHeadAttention):
    """ Multi head attention with linear projections on K and V
    """
    # TODO: rename k and sequence_length
    def __init__(self, *args, k: int, sequence_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_dim = k
        self.max_length = sequence_length

        # Using Linear so that it automatically handles initialization
        self.E = nn.Linear(self.max_length, self.proj_dim, bias=False)
        self.F = nn.Linear(self.max_length, self.proj_dim, bias=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        causal: bool = False, # here just for compatibility reasons
        key_mask: Tensor = None,
        query_mask: Tensor = None,
        full = False,
    ) -> Tensor:
        if (key_mask is None) != (query_mask is None):
            raise ValueError('Either both key_mask and query_mask must be None, or both must be provided.')

        if causal:
            raise ValueError('Warning: causal masking is not supported by the Linformer attention')

        q, k, v = self._qkv_proj(query, key, value)

        # TODO: mask before projecting on (query, key, value)
        if query_mask is not None:
            q = q.masked_fill(~query_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)

        # Share same mask for K and V
        if key_mask is not None:
            k = k.masked_fill(~key_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)
            v = v.masked_fill(~key_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)

        # Broadcast E @ K and F @ V over batch and head dimensions
        if not full:
            proj_k = self.E.weight
            proj_v = self.F.weight

            if key.shape[1] < self.max_length:
                proj_k = proj_k[:, :key.shape[1]]
                proj_v = proj_v[:, :key.shape[1]]

            k = torch.matmul(proj_k, k)
            v = torch.matmul(proj_v, v)

        # attn = scaled_dot_product_attention(q, k, v, dim = self.inner_dim)
        attn = F.scaled_dot_product_attention(q, k, v)

        # Concatenate heads
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        # Project back to dim (unnecessary if self.inner_dim * n_heads == dim)
        if self.inner_dim * self.n_heads != self.dim:
            attn = self.out_proj(attn)

        assert attn.shape[-1] == self.dim
        return attn

class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        """ TODO """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dim = self.encoder.dim

    def forward(
        self,
        enc_in: Tensor,
        dec_in: Optional[Tensor] = None,
        enc_mask: Optional[Tensor] = None,
        dec_mask: Optional[Tensor] = None,
        return_enc_output: bool = False,
    ):
        enc_out = self.encoder(enc_in, mask = enc_mask)

        if self.decoder is not None:
            dec_out = self.decoder(dec_in, enc_out, dec_mask = dec_mask, enc_mask = enc_mask)

            if return_enc_output:
                return enc_out, dec_out
            else:
                return dec_out
        return enc_out

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

# TODO: dropout, gelu, 
class TransformerEncoder(Replicated):
    def __init__(self,*args, causal: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.causal = causal
        self.dim = self.stacked[0].dim

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, causal = self.causal, mask = mask)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, attention: nn.Module):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.attention = attention
        if not self.attention:
            self.attention = MultiHeadAttention(dim, n_heads = 8, inner_dim = dim // 8)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias = False),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim, bias = False)
        )

    def forward(self, x, causal = False, mask = None):
        x = self.norm1(self.attention(x, x, x, causal = causal, query_mask = mask, key_mask = mask) + x) # Norm first = False (original paper)
        x = self.norm2(self.mlp(x) + x) # Norm first = False (original paper)
        return x

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
                 policy_head=False,
                 policy_depth=10,
                 policy_classes=4096,
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
        self.policy_head = policy_head
        self.policy_depth = policy_depth
        self.policy_classes = policy_classes

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

        attn = MultiHeadAttention(dim=dim, n_heads=heads)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(dim, mlp_dim, attn),
            n_layers=stages_depth[0] if hierarchical else depth
        )
                                                                      
        # self.transformer = nn.TransformerEncoder(
        #         nn.TransformerEncoderLayer(
        #             d_model=dim,
        #             nhead=heads,
        #             dim_feedforward=mlp_dim,
        #             dropout=dropout,
        #             activation='gelu',
        #             norm_first=True,
        #             batch_first=True
        #             ),
        #         num_layers=stages_depth[0] if hierarchical else depth,
        #         norm=nn.LayerNorm(dim)
        #         )

        out_dim=dim
        merging_layers = []
        if hierarchical:
            for l in range(merging_blocks):
                in_dim = out_dim
                out_dim *= 2
                attn = MultiHeadAttention(dim=out_dim, n_heads=heads)
                k = stages_depth[l + 1]
                merge = nn.Sequential(
                    PatchMerging(patch_size*(2**(l+1)), in_dim, out_dim, mode=merging_strategy),
                    TransformerEncoder(
                        TransformerEncoderLayer(
                            out_dim,
                            mlp_dim, # dim_feedforward=(out_dim * 4),
                            attn),
                        n_layers=k
                        # norm=nn.LayerNorm(out_dim)
                    )
                )
                merging_layers.append(merge)

        self.stages_stack = nn.ModuleList(merging_layers)
        self.mlp_head = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, num_classes)
        )

        if self.policy_head:
            self.policy_mlp = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, self.policy_depth * self.policy_classes),
                Rearrange('b (d c) -> b d c', d = self.policy_depth, c = self.policy_classes)
            )

    def forward(self, img, aux):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        aux = self.to_aux_embedding(aux) # Bx3 -> BxD
        aux = aux[:, None, :] # Add dummy (n) dimension: BxD -> BxNxD

        x = torch.cat((aux, x), dim=1) # Add aux as an additional token patch

        # TODO: make ViT pos embedding optional
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        # Hierarchical (pooling) stages
        for stage in self.stages_stack:
            x = stage(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        if self.policy_head:
            a = self.policy_mlp(x)
            return self.mlp_head(x), a

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
                 policy_head=False,
                 policy_depth=10,
                 policy_classes=4096,
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
                    emb_dropout=emb_dropout,
                    policy_head=policy_head,
                    policy_depth=policy_depth,
                    policy_classes=policy_classes
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
