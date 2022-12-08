import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import torchdiffeq

import controldiffeq


class VectorFieldGDE(torch.nn.Module):
    def __init__(self, X, func_f, func_g):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.X = X
        self.func_f = func_f
        self.func_g = func_g

    def __call__(self, t, hz):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0]  # h: torch.Size([64, 207, 32])
        z = hz[1]  # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h)  # vector_field_f: torch.Size([64, 207, 32, 2])
        vector_field_g = self.func_g(z)  # vector_field_g: torch.Size([64, 207, 32, 2])

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        return tuple([dh, out])


class VectorField_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(VectorField_f, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linear = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                    for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)  # 32,32*4  -> # 32,32,4

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linear:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k,
                 embed_dim):
        super(VectorField_g, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                          hidden_channels * hidden_channels)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
        # torch.nn.init.xavier_uniform_(self.weights_pool)
        # torch.nn.init.xavier_uniform_(self.bias_pool)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        z = self.agc(z)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z  # torch.Size([64, 307, 64, 1])

    def agc(self, z):
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return z


# class NeuralGCDE(nn.Module):
#     def __init__(self, num_nodes, func_f, func_g, input_channels, hidden_channels, output_channels, atol,
#                  rtol, solver, horizon, num_layers, default_graph, embed_dim):
#         super(NeuralGCDE, self).__init__()
#         self.num_node = num_nodes
#         self.input_dim = input_channels
#         self.hidden_dim = hidden_channels
#         self.output_dim = output_channels
#         self.horizon = horizon
#         self.num_layers = num_layers
#
#         self.default_graph = default_graph
#         # self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
#
#         self.func_f = func_f
#         self.func_g = func_g
#         self.solver = solver
#         self.atol = atol
#         self.rtol = rtol
#
#         # predictor
#         self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#
#         self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
#         self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
#
#     def forward(self, coeffs, times):
#         # source: B, T_1, N, D
#         # target: B, T_2, N, D
#         spline = torchcde.CubicSpline(coeffs, times)
#         h0 = self.initial_h(spline.evaluate(times[0]))
#         z0 = self.initial_z(spline.evaluate(times[0]))
#
#         z_t = cdeint_gde(X=spline,  # dh_dt
#                          h0=h0,
#                          z0=z0,
#                          func_f=self.func_f,
#                          func_g=self.func_g,
#                          t=times,
#                          method=self.solver,
#                          atol=self.atol,
#                          rtol=self.rtol)
#
#         z_T = z_t[-1:, ...].transpose(0, 1)
#
#         # CNN based predictor
#         output = self.end_conv(z_T)  # B, T*C, N, 1
#         output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
#         output = output.permute(0, 1, 3, 2)  # B, T, N, C
#
#         return output


class NeuralGCDE(nn.Module):
    def __init__(self, num_nodes, horizon, num_layers, func_f, func_g, input_channels, hidden_channels, output_channels,
                 atol,
                 rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = horizon
        self.num_layers = num_layers

        # self.default_graph = args.default_graph
        # self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, coeffs, times):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        h0 = self.initial_h(spline.evaluate(times[0]))
        z0 = self.initial_z(spline.evaluate(times[0]))

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative,  # dh_dt
                                           h0=h0,
                                           z0=z0,
                                           func_f=self.func_f,
                                           func_g=self.func_g,
                                           t=times,
                                           method=self.solver,
                                           atol=self.atol,
                                           rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:, ...].transpose(0, 1)

        # CNN based predictor
        output = self.end_conv(z_T)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output


def make_model(num_nodes, input_dim, hid_dim, hid_hid_dim, output_dim, embed_dim, num_layers, cheb_k, horizon, solver,
               default_graph):
    vector_field_f = VectorField_f(input_channels=input_dim, hidden_channels=hid_dim,
                                   hidden_hidden_channels=hid_hid_dim,
                                   num_hidden_layers=num_layers)
    vector_field_g = VectorField_g(input_channels=input_dim, hidden_channels=hid_dim,
                                   hidden_hidden_channels=hid_hid_dim,
                                   num_hidden_layers=num_layers, num_nodes=num_nodes, cheb_k=cheb_k,
                                   embed_dim=embed_dim)
    model = NeuralGCDE(num_nodes=num_nodes, func_f=vector_field_f, func_g=vector_field_g, input_channels=input_dim,
                       hidden_channels=hid_dim,
                       output_channels=output_dim,
                       atol=1e-9, rtol=1e-7, solver=solver, horizon=horizon, num_layers=num_layers, )

    return model, vector_field_f, vector_field_g
