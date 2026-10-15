import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lwctrans.network_architecture.Coformer.Coformer import BasicBlock
from lwctrans.network_architecture.Coformer.light_transformer_block import ProjectConvBlock, LightTransformerBlock
from lwctrans.network_architecture.LWCTrans import InitWeights_He

MODES = ('ratio', 'dynratio', 'gate')


def _p(module, default=0.0):
    return getattr(module, 'p', getattr(module, 'drop_prob', default))


def _branch_args(ref: BasicBlock):
    c0 = ref.conv_blocks[0]
    t0 = ref.trans_blocks[0]
    return dict(
        half=c0.d_conv.conv.in_channels,
        n_conv=len(ref.conv_blocks),
        n_trans=len(ref.trans_blocks),
        conv_drop=_p(c0.d_conv.dropout),
        mlp_ratio=t0.mlp_ratio,
        mlp_drop=_p(t0.mlp.drop),
        drop_path=_p(t0.drop_path),
        norm_layer=type(t0.norm1),
        act_layer=type(t0.mlp.act),
    )


def _build_branches(a: dict, width: int):
    conv = nn.Sequential(*[ProjectConvBlock(dim=width, drop=a['conv_drop']) for _ in range(a['n_conv'])])
    trans = nn.Sequential(*[
        LightTransformerBlock(dim=width, mlp_ratio=a['mlp_ratio'], drop=a['mlp_drop'], drop_path=a['drop_path'],
                              norm_layer=a['norm_layer'], act_layer=a['act_layer'])
        for _ in range(a['n_trans'])
    ])
    return conv, trans


class RatioSplitBlock(nn.Module):
    """Learnable CNN/Transformer channel split. Static (one ratio per block) or dynamic (one ratio per sample)."""

    def __init__(self, ref: BasicBlock, dynamic: bool = False, hi: float = 0.625,
                 tau_start_frac: float = 0.0625, tau_end: float = 0.25, router_ratio: float = 0.25):
        super().__init__()
        a = _branch_args(ref)
        dim = 2 * a['half']
        self.dim = dim
        self.hi = hi
        self.lo = 1.0 - hi
        self.width = int(round(hi * dim))
        self.off = dim - self.width
        self.dynamic = dynamic
        self.conv_blocks, self.trans_blocks = _build_branches(a, self.width)
        if dynamic:
            hid = max(8, int(dim * router_ratio))
            self.router = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, 1))
        else:
            self.split_logit = nn.Parameter(torch.zeros(1))
        self.tau_start = tau_start_frac * dim
        self.tau_end = tau_end
        self.register_buffer('tau', torch.tensor(float(self.tau_start)))
        idx = torch.arange(dim, dtype=torch.float32) + 0.5
        self.register_buffer('idx', idx, persistent=False)
        self.register_buffer('only_conv', idx < self.off, persistent=False)
        self.register_buffer('only_trans', idx > self.width, persistent=False)
        self.last_stat = None

    def reset_adaptive(self):
        if self.dynamic:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        else:
            nn.init.zeros_(self.split_logit)

    def set_tau_progress(self, frac: float):
        frac = min(max(frac, 0.0), 1.0)
        t = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1.0 + math.cos(math.pi * frac))
        self.tau.fill_(t)

    def rho(self, x):
        if self.dynamic:
            z = self.router(x.mean(dim=(2, 3, 4)).float()).squeeze(1)
        else:
            z = self.split_logit.expand(x.shape[0])
        return self.lo + (self.hi - self.lo) * torch.sigmoid(z.float())

    def forward(self, x):
        b, c = x.shape[:2]
        rho = self.rho(x)
        m = torch.sigmoid((rho[:, None] * c - self.idx[None]) / self.tau)
        m = torch.where(self.only_conv[None], torch.ones_like(m), torch.where(self.only_trans[None], torch.zeros_like(m), m))
        hard = (m > 0.5).float()
        m_in = (hard + m - m.detach()).to(x.dtype)
        m_out = m.to(x.dtype)
        sh = (b, -1, 1, 1, 1)
        w, off = self.width, self.off
        yc = self.conv_blocks(x[:, :w] * m_in[:, :w].reshape(sh)) * m_out[:, :w].reshape(sh)
        yt = self.trans_blocks(x[:, off:] * (1 - m_in[:, off:]).reshape(sh)) * (1 - m_out[:, off:]).reshape(sh)
        y = F.pad(yc, (0, 0, 0, 0, 0, 0, 0, off)) + F.pad(yt, (0, 0, 0, 0, 0, 0, off, 0))
        if not self.training:
            self.last_stat = rho.detach().float().cpu()
        return y

    def summary(self):
        if self.dynamic:
            return {'rho_bias': float(self.lo + (self.hi - self.lo) * torch.sigmoid(self.router[-1].bias.detach().float()).item()),
                    'tau': float(self.tau)}
        rho = float(self.lo + (self.hi - self.lo) * torch.sigmoid(self.split_logit.detach().float()).item())
        return {'rho': rho, 'conv_channels': rho * self.dim, 'dim': self.dim, 'tau': float(self.tau)}


class GateSplitBlock(nn.Module):
    """Fixed 50:50 split plus an input-dependent per-channel gate that reallocates weight between the two branches."""

    def __init__(self, ref: BasicBlock, reduction: int = 4):
        super().__init__()
        half = ref.conv_blocks[0].d_conv.conv.in_channels
        self.dim = 2 * half
        self.conv_blocks = ref.conv_blocks
        self.trans_blocks = ref.trans_blocks
        hid = max(8, half // reduction)
        self.gate = nn.Sequential(nn.Linear(half, hid), nn.GELU(), nn.Linear(hid, half))
        self.last_stat = None

    def reset_adaptive(self):
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = self.conv_blocks(x1)
        y2 = self.trans_blocks(x2)
        a = torch.sigmoid(self.gate((y1 + y2).mean(dim=(2, 3, 4)).float()))
        if not self.training:
            self.last_stat = a.detach().cpu()
        a = a.to(y1.dtype)[:, :, None, None, None]
        return torch.cat([2 * a * y1, 2 * (1 - a) * y2], dim=1)

    def summary(self):
        return {'dim': self.dim}


def _make(mode: str, ref: BasicBlock):
    if mode == 'ratio':
        return RatioSplitBlock(ref, dynamic=False)
    if mode == 'dynratio':
        return RatioSplitBlock(ref, dynamic=True)
    if mode == 'gate':
        return GateSplitBlock(ref)
    raise ValueError(f'unknown split mode {mode}, expected one of {MODES}')


def _backbone(network):
    net = network
    for attr in ('module', '_orig_mod'):
        net = getattr(net, attr, net)
    return net.network.backbone


def convert_basic_blocks(network, mode: str):
    backbone = _backbone(network)
    init = InitWeights_He()
    for layer in backbone.layers:
        for i, blk in enumerate(layer.blocks):
            if isinstance(blk, BasicBlock):
                new = _make(mode, blk)
                if mode == 'gate':
                    new.gate.apply(init)
                else:
                    new.apply(init)
                new.reset_adaptive()
                layer.blocks[i] = new
    return network


def adaptive_modules(network):
    out = []
    for si, layer in enumerate(_backbone(network).layers):
        for bi, blk in enumerate(layer.blocks):
            if isinstance(blk, (RatioSplitBlock, GateSplitBlock)):
                out.append((f'stage{si + 1}' if len(layer.blocks) == 1 else f'stage{si + 1}.{bi}', blk))
    return out
