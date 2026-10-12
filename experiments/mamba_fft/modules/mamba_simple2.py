import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class Mamba(nn.Module):
    """Standard Mamba block (copied from mamba_ssm). Used in Variate Path."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None, None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class TLE(nn.Module):
    """Multi-scale Dilated Temporal Enhancement (adaptive to L).

    3 parallel depthwise convs with dilation ratios L/24, L/12, L/6.
    Each branch captures different time-range patterns. Zero-init.
    """

    def __init__(self, n_vars):
        super().__init__()
        self.n_vars = n_vars
        self.weight = nn.Parameter(torch.randn(n_vars, 1, 3) * 0.02)
        self.proj = nn.Linear(n_vars * 3, n_vars, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, L, V)
        L = x.shape[1]
        d1 = 1
        d2 = max(1, L // 24)
        d3 = max(1, L // 12)
        x_t = x.transpose(1, 2)                              # (B, V, L)
        x_t = F.layer_norm(x_t, (L,))                         # per-var norm
        c1 = F.conv1d(x_t, self.weight, groups=self.n_vars, dilation=d1, padding=d1)
        c2 = F.conv1d(x_t, self.weight, groups=self.n_vars, dilation=d2, padding=d2)
        c3 = F.conv1d(x_t, self.weight, groups=self.n_vars, dilation=d3, padding=d3)
        x_t = torch.cat([c1, c2, c3], dim=1)                  # (B, 3V, L)
        x_t = F.silu(x_t).transpose(1, 2)                    # (B, L, 3V)
        x_t = self.proj(x_t)                                  # (B, L, V)
        return x + self.alpha * x_t                            # zero-init


class TrendSeasonal(nn.Module):
    """Multi-scale Learnable Trend Decomposition (adaptive to L).

    3 learnable alphas × 3 kernel sizes scaled by L.
    Model learns optimal combination. Zero-init.
    """

    def __init__(self, n_vars):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, L, V)
        L = x.shape[1]
        scales = [
            max(3,  L // 32 + 1 - (L // 32) % 2),   # ~3%  of L
            max(7,  L // 14 + 1 - (L // 14) % 2),   # ~7%  of L
            max(15, L // 6  + 1 - (L // 6)  % 2),   # ~16% of L
        ]
        x_t = x.transpose(1, 2)                              # (B, V, L)
        seasonal_total = 0
        for k, alpha in zip(scales, [self.alpha1, self.alpha2, self.alpha3]):
            trend = F.avg_pool1d(x_t, int(k), stride=1, padding=int(k) // 2)
            seasonal_total = seasonal_total + alpha * (x_t - trend)
        x_t = x_t + seasonal_total
        return x_t.transpose(1, 2)                            # (B, L, V)


class SpectralMod(nn.Module):
    """Adaptive Spectral Modulation with input-dependent gating.

    FFT → amplitude spectrum → MLP predicts dynamic gain → modulate freq_filter.
    Model learns which frequencies matter based on current input. Zero-init.
    """

    def __init__(self, seq_len, n_vars, hidden=16):
        super().__init__()
        self.n_freqs = seq_len // 2 + 1
        self.freq_filter = nn.Parameter(torch.ones(self.n_freqs, n_vars, dtype=torch.cfloat) * 0.1)
        # MLP: amplitude spectrum → per-freq weight
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.n_freqs, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_freqs),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, L, V)
        xf = torch.fft.rfft(x.float(), dim=1)                # (B, n_freqs, V) complex
        # Adaptive gating from amplitude spectrum
        amp = torch.abs(xf).mean(dim=-1)                      # (B, n_freqs)
        gate = self.gate_mlp(amp).unsqueeze(-1)               # (B, n_freqs, 1)
        xf = xf * self.freq_filter * gate                     # dynamic modulation
        x_mod = torch.fft.irfft(xf, dim=1, n=x.shape[1])     # (B, L, V)
        return x + self.alpha * x_mod.to(dtype=x.dtype)       # zero-init


class DVR(nn.Module):
    """Dynamic Variate Routing: N-dim inverted attention with learnable gate."""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, E) — self-attention on the variate (N) dimension
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        gate = torch.sigmoid(self.gate(residual))
        x = residual + gate * attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CSMamba(nn.Module):
    """CS-Mamba: Cross-Scale Mamba with Dynamic Variate Routing.

    Ablation flags (set via configs):
      use_tle       : enable TLE (default True). False -> skip.
      use_trend_seasonal: enable TrendSeasonal (default True). False -> skip.
      use_spectral_mod : enable SpectralMod (default True). False -> skip.
      use_dvr       : enable DVR cross-variate routing (default True). False -> skip.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        self.use_dvr = getattr(configs, 'use_dvr', True)
        self.use_tle = bool(getattr(configs, 'use_tle', True))
        self.use_trend_seasonal = bool(getattr(configs, 'use_trend_seasonal', True))
        self.use_spectral_mod = bool(getattr(configs, 'use_spectral_mod', True))

        d_state = configs.d_state if hasattr(configs, 'd_state') else 16

        self.blocks = nn.ModuleList([
            Mamba(d_model=configs.d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(self.e_layers)
        ])
        # Zero-initialized skip scales: at initialization the Mamba path is
        # close to identity, and each scale learns how much pre-Mamba
        # representation to preserve without post-LN changing the DVR input
        # distribution at initialization.
        self.mamba_skip_scales = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.e_layers)
        ])

        if self.use_dvr:
            self.dvr_layers = nn.ModuleList([
                DVR(d_model=configs.d_model, n_heads=self._get_dvr_heads(configs))
                for _ in range(self.e_layers)
            ])
        else:
            self.dvr_layers = None

        # Pre-embedding modules (operate on (B, L, V), time dimension)
        enc_in = getattr(configs, 'enc_in', 7)
        self.tle = TLE(enc_in) if self.use_tle else None
        self.trend = TrendSeasonal(enc_in) if self.use_trend_seasonal else None
        self.spectral = SpectralMod(configs.seq_len, enc_in) if self.use_spectral_mod else None

        from layers.Embed import DataEmbedding_inverted
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    @staticmethod
    def _get_dvr_heads(configs):
        d_model = configs.d_model
        max_heads = min(8, configs.enc_in)
        valid = [h for h in range(1, max_heads + 1) if d_model % h == 0]
        return max(valid) if valid else 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape

        # Serial pre-embedding modules, each operating on the L dimension.
        x = x_enc
        if self.tle is not None:
            x = self.tle(x)
        if self.trend is not None:
            x = self.trend(x)
        if self.spectral is not None:
            x = self.spectral(x)
        x = self.enc_embedding(x, None)

        if self.use_dvr:
            for mamba_block, skip_scale, dvr in zip(self.blocks, self.mamba_skip_scales, self.dvr_layers):
                x = mamba_block(x) + skip_scale * x
                x = dvr(x)
        else:
            for mamba_block, skip_scale in zip(self.blocks, self.mamba_skip_scales):
                x = mamba_block(x) + skip_scale * x

        dec_out = self.projector(x).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out


