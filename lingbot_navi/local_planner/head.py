"""Checkpoint-compatible local planning head for external world routes."""
import torch
from torch import nn
from torch.nn import functional as F

from ..spatial_encoding import METRIC_FOURIER_DIM, metric_xyz_fourier_features

METRIC_XYZ_FEATURE_DIM = 3 + METRIC_FOURIER_DIM


def metric_xyz_features(xyz):
    return torch.cat((xyz.float(), metric_xyz_fourier_features(xyz)), dim=-1)


def local_attention_mask(padding_mask, *, query_start, dtype):
    """Conditions attend to conditions; trajectory queries attend to all valid tokens."""
    if (padding_mask.ndim != 2 or padding_mask.dtype != torch.bool
            or not all(padding_mask.shape) or not dtype.is_floating_point):
        raise ValueError("attention mask needs nonempty boolean [B,N] padding and floating dtype")
    batch, length = padding_mask.shape
    if type(query_start) is not int or not 0 < query_start < length:
        raise ValueError("prefix attention requires an explicit condition/query boundary inside the sequence")
    if padding_mask[:, :query_start].all(dim=1).any():
        raise ValueError("each sample requires at least one valid condition token")
    mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=padding_mask.device)
    mask[:, :, :query_start, query_start:] = float("-inf")
    return mask.masked_fill_(padding_mask[:, None, None, :], float("-inf"))


class EncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, *, attention_mask):
        # Same Pre-LN encoder equations as 0926. Direct SDPA keeps [B,1,N,N]
        # masks broadcast over heads instead of allocating [B*H,N,N].
        normalized = self.norm1(src)
        qkv = F.linear(normalized, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        batch, length, width = src.shape
        heads = self.self_attn.num_heads
        q, k, v = (x.reshape(batch, length, heads, width // heads).transpose(1, 2)
                   for x in qkv.chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.self_attn.dropout if self.training else 0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        src = src + self.dropout1(self.self_attn.out_proj(attended))
        hidden = self.norm2(src)
        return src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(hidden)))))


class LocalEncoderStyleSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.hidden_dim
        self.layers = nn.ModuleList([
            EncoderLayer(config.hidden_dim, config.heads, config.ffn_dim, config.dropout,
                         activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, tokens, *, padding_mask, query_start):
        if (tokens.ndim != 3 or tokens.shape[-1] != self.width or not tokens.is_floating_point()
                or not tokens.shape[0] or not tokens.shape[1] or padding_mask.shape != tokens.shape[:2]
                or padding_mask.dtype != torch.bool or padding_mask.device != tokens.device):
            raise ValueError("encoder requires [B,N,D] tokens and boolean [B,N] padding on the same device")
        hidden = tokens.masked_fill(padding_mask[..., None], 0)
        if not torch.isfinite(hidden).all():
            raise ValueError("valid encoder tokens must be finite")
        mask = local_attention_mask(padding_mask, query_start=query_start, dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask)
        return self.norm(hidden).float()


def build_input_projection(input_width, config):
    """Normalize frozen features, then optionally learn a channel mapping.

    Equal widths with auto/identity do NOT allocate a square Linear. Prompt
    and all reference/history/current image patches use the same policy, with
    their actual source widths verified independently. No cache is rewritten.
    """
    from torch import nn
    width, mode = config.hidden_dim, config.input_projection_mode
    if input_width is None or width is None:
        raise ValueError("resolve frozen input and head widths before building projections")
    if mode == "identity" and input_width != width:
        raise ValueError("identity input projection requires equal source/head widths")
    projection = (nn.Identity() if mode == "identity" or mode == "auto" and input_width == width
                  else nn.Linear(input_width, width))
    return nn.Sequential(nn.LayerNorm(input_width), projection)


class RouteTokenEncoder(nn.Module):
    """[B,R,3] -> [B,R,D]; raw XYZ + shared 60D metric Fourier + arc + order.

    Arc is measured along the SUPPLIED (possibly noisy) route, never local GT.
    Legacy 5D features exist only to load explicitly versioned old checkpoints.
    """
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(METRIC_XYZ_FEATURE_DIM + 2, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, route, padding_mask):
        xyz = route.detach().float().masked_fill(padding_mask[..., None], 0.)
        if not torch.isfinite(xyz).all():
            raise ValueError("valid route points must be finite")
        arc = torch.nn.functional.pad(torch.linalg.vector_norm(xyz.diff(dim=1), dim=-1).cumsum(1), (1, 0))
        order = torch.linspace(0., 1., xyz.shape[1], device=xyz.device).expand(xyz.shape[0], -1)
        # Same FP32 physical spectrum as goal_condition_features: 10 frequencies
        # per axis, sin/cos = 60 channels. No bbox normalization or extra 2*pi.
        position = metric_xyz_features(xyz)
        return self.projection(torch.cat([position, arc[..., None], order[..., None]], -1))

class RoutePlanningHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.visual_projection = build_input_projection(config.vision_width, config)
        # Normalized patch XY, seconds before current, history/reference/current multi-hot.
        self.visual_metadata = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.route_encoder = RouteTokenEncoder(width)
        self.modality = nn.Parameter(torch.randn(4, width) * .02)  # navigation, image, prompt, query
        self.queries = nn.Parameter(torch.randn(config.horizon, width) * .02)
        self.slot_projection = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.mode_embedding = nn.Embedding(2, width)
        nn.init.zeros_(self.mode_embedding.weight)
        self.register_buffer("query_slot_progress", torch.arange(1, config.horizon + 1).float() / config.horizon)
        self.transformer = LocalEncoderStyleSelfAttention(config)
        self.readout = nn.Linear(width, 3)
        self.action_readout = nn.Linear(width, 3)
        self.arrival_readout = nn.Linear(width, 1)
        with torch.no_grad():
            self.action_readout.bias.copy_(torch.tensor([0., 0., 1.]))
            self.arrival_readout.bias.fill_(-2.)

    def forward(self, route, *, visual_tokens, visual_metadata, visual_padding_mask):
        batch = visual_tokens.shape[0]
        device = visual_tokens.device
        visual = self._condition(visual_tokens, visual_padding_mask, batch, self.config.vision_width, "visual")
        metadata = self._condition(visual_metadata, visual_padding_mask, batch, 6, "image XY/time/roles")
        route_padding = torch.zeros(route.shape[:2], dtype=torch.bool, device=device)
        navigation = self.route_encoder(route, route_padding) + self.modality[0]
        images = self.visual_projection(visual) + self.visual_metadata(metadata) + self.modality[1]
        query = (self.queries + self.slot_projection(self.query_slot_progress[:, None]) + self.modality[3])[None]
        query = query + self.mode_embedding(torch.zeros(batch, dtype=torch.long, device=device))[:, None]
        tokens = torch.cat((images, navigation, query), dim=1)
        padding = torch.cat((visual_padding_mask, route_padding,
                             torch.zeros(batch, self.config.horizon, dtype=torch.bool, device=device)), dim=1)
        hidden = self.transformer(tokens, padding_mask=padding, query_start=tokens.shape[1] - self.config.horizon)
        future = hidden[:, -self.config.horizon:]
        return dict(xyz=self.readout(future), action_logits=self.action_readout(future),
                    arrival_logits=self.arrival_readout(future).squeeze(-1))

    @staticmethod
    def _condition(features, padding, batch, width, name):
        """Validate before projection; masked NaN/Inf must never reach LayerNorm/QKV."""
        if (features.ndim != 3 or features.shape[0] != batch or features.shape[1] == 0
                or features.shape[-1] != width or padding.shape != features.shape[:2]
                or padding.dtype != torch.bool or padding.device != features.device):
            raise ValueError(f"{name} requires [B,N,{width}] features and a boolean [B,N] padding mask")
        if padding.all(dim=1).any():
            raise ValueError(f"each sample requires valid {name} tokens")
        # Upstream features are frozen, but the following projections are trainable.
        clean = features.detach().float().masked_fill(padding[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError(f"valid {name} features must be finite")
        return clean
