import torch
import torch.nn as nn
import torch.nn.functional as F


class  Spat_Cau_Encoder(nn.Module):
    def __init__(self,n_sub, n_nodes,in_dim,hid_dim):
        super().__init__()
        if in_dim < 2:
            raise ValueError("Spat_Cau_Encoder requires in_dim >= 2 for lagged cause/effect windows")
        self.num_nodes = n_nodes
        self.num_sub = n_sub
        self.A_c_out = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        self.A_c_in = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        torch.nn.init.kaiming_uniform_(self.A_c_out)
        self.A_s_out = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        self.A_s_in = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        torch.nn.init.kaiming_uniform_(self.A_s_out)
        with torch.no_grad():
            self.A_c_in.copy_(self.A_c_out)
            self.A_s_in.copy_(self.A_s_out)
        self.common= nn.Conv2d(in_channels=n_sub, out_channels=1, kernel_size=1, stride=1, padding=0)
        lagged_in_dim = in_dim - 1
        self.fc_sin = nn.ModuleList([nn.Linear(lagged_in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_sout = nn.ModuleList([nn.Linear(lagged_in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_cin = nn.ModuleList([nn.Linear(lagged_in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_cout = nn.ModuleList([nn.Linear(lagged_in_dim, hid_dim) for _ in range(n_sub)])

    def forward(self, inputs, i):
        all_c = []
        all_s = []
        A_c_out = self.A_c_out
        A_c_in = self.A_c_in
        A_s_out = self.A_s_out
        A_s_in = self.A_s_in
        for j in range(self.num_sub):
            batch_data_j = inputs[j][i]
            x, y, t = batch_data_j
            x = x.squeeze(-1)
            cause_c = A_c_out[j] @ x[..., :-1]
            effect_c = A_c_in[j] @ x[..., 1:]
            cause_s = A_s_out[j] @ x[..., :-1]
            effect_s = A_s_in[j] @ x[..., 1:]
            causal_j_c = F.tanh(self.fc_cout[j](cause_c)) @ F.tanh(self.fc_cin[j](effect_c)).permute(0, 2, 1)
            causal_j_s = F.tanh(self.fc_sout[j](cause_s)) @ F.tanh(self.fc_sin[j](effect_s)).permute(0, 2, 1)
            all_c.append(causal_j_c)
            all_s.append(causal_j_s)
        causal_c_all = torch.stack(all_c, dim=1)
        causal_c = self.common(causal_c_all)
        causal_s = torch.stack(all_s, dim=1)
        return causal_s, causal_c.mean(dim=1).squeeze(1), causal_c_all
