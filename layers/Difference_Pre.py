import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:  
    try:
        from mamba_plus import Mamba
    except ImportError:
        Mamba = None

class DifferenceDataEmb(nn.Module):

    def __init__(self, res_num, enc_in, d_model):
        super(DifferenceDataEmb, self).__init__()
        self.res_num = res_num
        self.enc_in = enc_in
        self.d_model = d_model
        
        self.embeddings = nn.ModuleList([nn.Linear(self.enc_in, self.d_model) for _ in range(self.res_num)])

    def forward(self, multi_res_data):

        x_diff_list = []
        x_padding_list = []
        
        for l in range(self.res_num):
            x = multi_res_data[l].permute(0, 2, 1)
            
            x_padding = torch.concatenate([x[:, 0:1, :], x], dim=1)
            
            x_diff = torch.diff(x_padding, dim=1)
            
            x_diff_emb = self.embeddings[l](x_diff)
            
            x_diff_list.append(x_diff_emb)
            x_padding_list.append(x_padding)

        return x_diff_list, x_padding_list

class DataRestoration(nn.Module):
    def __init__(self, res_num, enc_in, d_model):
        super(DataRestoration, self).__init__()
        self.res_num = res_num
        self.enc_in = enc_in
        self.d_model = d_model
        
        self.projections = nn.ModuleList([nn.Linear(self.d_model, self.enc_in) for _ in range(self.res_num)])

    def forward(self, x_diff_list, x_padding_list):
        x_out_list = []
        
        for l in range(self.res_num):
            x_diff = self.projections[l](x_diff_list[l])
            
            _x_out = x_diff + x_padding_list[l][:, :-1, :]
            _x_out = _x_out.permute(0, 2, 1)
            
            x_out_list.append(_x_out)

        return x_out_list


class DifferenceAddNorm(nn.Module):
    def __init__(self, d_model, dropout, residual=True):
        super(DifferenceAddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual

    def forward(self, new, old):
        new = self.dropout(new)
        return self.norm(old + new) if self.residual else self.norm(new)


class DifferenceSequenceMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(DifferenceSequenceMixer, self).__init__()
        self.use_mamba = Mamba is not None
        self.dropout = nn.Dropout(dropout)

        if self.use_mamba:
            self.mixer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        if self.use_mamba:
            out = self.mixer(x)
        else:
            out, _ = self.mixer(x)
            out = self.proj(out)
        return self.dropout(out)


class DifferenceBiMambaLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        residual=True,
    ):
        super(DifferenceBiMambaLayer, self).__init__()
        self.bidirectional = bidirectional

        self.forward_mixer = DifferenceSequenceMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        if self.bidirectional:
            self.backward_mixer = DifferenceSequenceMixer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        self.addnorm_mamba = DifferenceAddNorm(d_model, dropout, residual=residual)
        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1),
        )
        self.addnorm_ffn = DifferenceAddNorm(d_model, dropout, residual=residual)

    def forward(self, x):
        output_forward = self.forward_mixer(x)

        if self.bidirectional:
            output_backward = self.backward_mixer(x.flip(dims=[1])).flip(dims=[1])
            mixed = output_forward + output_backward
        else:
            mixed = output_forward

        output = self.addnorm_mamba(mixed, x)
        residual = output
        output = self.ffn(output.transpose(1, 2)).transpose(1, 2)
        output = self.addnorm_ffn(output, residual)
        return output


class DifferenceBiMambaEncoder(nn.Module):
    def __init__(
        self,
        res_num,
        d_model,
        num_layers=1,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        residual=True,
    ):
        super(DifferenceBiMambaEncoder, self).__init__()
        self.res_num = res_num
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DifferenceBiMambaLayer(
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout,
                            activation=activation,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            bidirectional=bidirectional,
                            residual=residual,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(res_num)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(res_num)])

    def forward(self, x_diff_list, attn_mask=None, tau=None, delta=None):
        outputs = []
        for idx, x in enumerate(x_diff_list):
            for layer in self.layers[idx]:
                x = layer(x)
            outputs.append(self.norms[idx](x))
        return outputs, None


class DifferenceEVSBlock(nn.Module):
    def __init__(
        self,
        d_model,
        small_kernel=3,
        large_kernel=9,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        ffn_expansion=2.0,
    ):
        super(DifferenceEVSBlock, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm or mamba_plus is required for DifferenceEVSBlock. Install it in the runtime environment."
            )

        hidden_features = int(d_model * ffn_expansion)

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.local_conv = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=small_kernel,
                padding=small_kernel // 2,
                groups=d_model,
            ),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_res_scale = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_model, d_model)
        self.ssm_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.project_in = nn.Linear(d_model, hidden_features * 2)
        self.dwconv = nn.Sequential(
            nn.Conv1d(
                hidden_features * 2,
                hidden_features * 2,
                kernel_size=large_kernel,
                padding=large_kernel // 2,
                groups=hidden_features * 2,
            ),
            nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=1),
        )
        self.project_out = nn.Linear(hidden_features, d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        ssm_input = self.norm1(x)
        u, z = self.in_proj(ssm_input).chunk(2, dim=-1)
        local_feat = self.local_conv(u.transpose(1, 2)).transpose(1, 2)
        ssm_feat = self.ssm(local_feat)
        z_gate = F.gelu(z)
        mixed_feat = local_feat + self.ssm_res_scale * ssm_feat
        mixed_feat = self.out_proj(mixed_feat * z_gate)
        x = x + self.ssm_dropout(mixed_feat)

        ffn_input = self.norm2(x)
        ffn_hidden = self.project_in(ffn_input)
        ffn_hidden = self.dwconv(ffn_hidden.transpose(1, 2)).transpose(1, 2)
        ffn_left, ffn_right = ffn_hidden.chunk(2, dim=-1)
        ffn_hidden = F.gelu(ffn_left) * ffn_right
        ffn_hidden = self.project_out(ffn_hidden)
        x = x + self.ffn_dropout(ffn_hidden)
        return x


class DifferenceTimeExpert(nn.Module):
    def __init__(
        self,
        res_num,
        d_model,
        small_kernel=3,
        large_kernel=9,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        ffn_expansion=2.0,
    ):
        super(DifferenceTimeExpert, self).__init__()
        self.blocks = nn.ModuleList(
            [
                DifferenceEVSBlock(
                    d_model=d_model,
                    small_kernel=small_kernel,
                    large_kernel=large_kernel,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(res_num)
            ]
        )

    def forward(self, x_diff_list, attn_mask=None, tau=None, delta=None):
        return [block(x_diff) for block, x_diff in zip(self.blocks, x_diff_list)], None
