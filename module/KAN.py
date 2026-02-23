import torch
import torch.nn.functional as F
import math

"""
Notes on this implementation (memory-efficient KAN):
1) Memory efficiency:
   Instead of explicitly expanding large intermediate tensors for multiple basis activations,
   the computation is reformulated as: compute basis functions first, then linearly combine them.
   This reduces memory footprint and can improve speed.

2) Regularization change:
   The original paper includes sample-based L1 regularization that depends on expanded tensors.
   Here we use L1 on spline weights + an entropy-like term (proxy) that is compatible with F.linear.

3) Optional learnable scaling:
   Some KAN implementations include a learnable scaling per spline; this code supports enabling/disabling it.

4) Initialization:
   Kaiming initialization is used to improve optimization stability (e.g. on MNIST).
"""

class KANLinear(torch.nn.Module):
    """
    KANLinear: a linear layer augmented with B-spline basis functions.

    Output = Linear(base_activation(x)) + Linear(BSplineBases(x))

    Args:
        in_features (int): input dimension
        out_features (int): output dimension
        grid_size (int): number of grid intervals (default 5)
        spline_order (int): spline order (default 3)
        scale_noise (float): noise scale for spline initialization
        scale_base (float): base weight scale
        scale_spline (float): spline weight scale
        enable_standalone_scale_spline (bool): whether to use per-(out,in) spline scaler
        base_activation (nn.Module): activation for base branch (default SiLU)
        grid_eps (float): interpolation factor between uniform grid and adaptive grid
        grid_range (list): input range for initial grid, e.g. [-1, 1]
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build grid: shape (in_features, grid_size + 2*spline_order + 1)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Parameters:
        # base_weight: (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # spline_weight: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # optional per-(out,in) spline scaler: (out_features, in_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init for base weights
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            # Noise used to initialize spline curve values at grid points
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )

            # Convert sampled curve values into spline coefficients
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis values for input x.

        Args:
            x: (B, in_features)

        Returns:
            bases: (B, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2*spline_order + 1)

        # x: (B, in_features, 1)
        x = x.unsqueeze(-1)

        # Initialize bases as piecewise-constant indicator functions
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Cox–de Boor recursion for B-spline bases
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit spline coefficients that interpolate given points (x, y).

        Args:
            x: (batch_size, in_features)
            y: (batch_size, in_features, out_features)

        Returns:
            coeff: (out_features, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # A: (in_features, batch, coeff)
        A = self.b_splines(x).transpose(0, 1)
        # B: (in_features, batch, out_features)
        B = y.transpose(0, 1)

        # Least squares solve: A @ solution ≈ B
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, coeff, out_features)
        result = solution.permute(2, 0, 1)            # (out_features, in_features, coeff)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        # Apply optional per-(out,in) scaling to spline weights
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: (B, in_features)

        Returns:
            y: (B, out_features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Base branch: activation + linear
        base_output = F.linear(self.base_activation(x), self.base_weight)  # (B, out_features)

        # Spline branch:
        # bases: (B, in_features, coeff) -> flatten to (B, in_features*coeff)
        # spline_weight: (out_features, in_features, coeff) -> flatten to (out_features, in_features*coeff)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        Update grid locations adaptively based on data distribution.

        Args:
            x: (B, in_features)
            margin: padding margin for grid boundaries
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Current spline bases and original coefficients
        splines = self.b_splines(x).permute(1, 0, 2)          # (in, B, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in, coeff, out)

        # Compute current unreduced spline output: (B, in, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        # Sort per feature to estimate quantiles
        x_sorted = torch.sort(x, dim=0)[0]

        # Adaptive grid by sampling quantiles
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        # Uniform grid for stability
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Blend adaptive and uniform grids
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # Extend grid by spline_order points on both sides
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # Update buffers and re-fit spline coefficients on new grid
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Regularization proxy:
        - L1 on spline weights (mean abs)
        - an entropy-like term computed from normalized l1 per (out,in)

        Returns:
            scalar tensor
        """
        l1_fake = self.spline_weight.abs().mean(-1)  # (out, in)
        reg_act = l1_fake.sum()
        p = l1_fake / reg_act
        reg_ent = -torch.sum(p * p.log())
        return regularize_activation * reg_act + regularize_entropy * reg_ent


class KAN(torch.nn.Module):
    """
    Multi-layer KAN: stack multiple KANLinear layers.

    Args:
        layers_hidden (list): e.g. [in_dim, h1, h2, out_dim]
        other args: passed to each KANLinear
    """
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()

        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        Args:
            x: (B, in_dim)
            update_grid: if True, update grid before each layer forward

        Returns:
            (B, out_dim)
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)