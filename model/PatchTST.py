import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*N, 1, L]  -> unfold patches along time
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*N, 1, n_patch, patch_len]
        x = x.squeeze(1)  # [B*N, n_patch, patch_len]
        n_patch = x.shape[1]
        x = self.value_embedding(x)  # [B*N, n_patch, d_model]
        x = x + self.position_embedding[:, :n_patch, :]
        return self.dropout(x)


class Model(nn.Module):
    """
    PatchTST (channel-independent patched Transformer).
    Paper: https://arxiv.org/abs/2211.14730
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = bool(getattr(configs, 'use_norm', True))
        self.output_attention = configs.output_attention

        patch_len = int(getattr(configs, 'patch_len', 16))
        stride = int(getattr(configs, 'stride', 8))
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, configs.dropout
        )
        # pad so unfold covers the full lookback
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # head: flatten patch tokens -> pred_len
        # n_patch depends on seq_len; compute for the configured lookback
        n_patch = int((configs.seq_len + padding - patch_len) / stride + 1)
        self.head_nf = configs.d_model * n_patch
        self.head = nn.Linear(self.head_nf, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        bsz, seq_len, n_vars = x_enc.shape
        # channel independence: [B, L, N] -> [B*N, 1, L]
        x = x_enc.permute(0, 2, 1).reshape(bsz * n_vars, 1, seq_len)
        x = self.padding_patch_layer(x)
        enc_out = self.patch_embedding(x)  # [B*N, n_patch, d_model]
        enc_out, attns = self.encoder(enc_out)
        # [B*N, n_patch * d_model]
        enc_out = enc_out.reshape(bsz * n_vars, -1)
        # [B*N, pred_len] -> [B, pred_len, N]
        dec_out = self.head(enc_out).reshape(bsz, n_vars, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
