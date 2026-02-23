import torch
import torch.nn as nn

class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module

    This module enhances feature representations by computing
    neuron importance using an energy function derived from
    spatial statistics (mean & variance).

    Reference:
    "SimAM: A Simple, Parameter-Free Attention Module for CNNs"
    """

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        # Sigmoid activation used to generate attention weights
        self.activation = nn.Sigmoid()

        # Stability constant to avoid division by zero
        self.e_lambda = e_lambda

    def __repr__(self):
        """Return a readable module description."""
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"

    @staticmethod
    def get_module_name():
        """Return module name (useful for logging or model summaries)."""
        return "simam"

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            Tensor of same shape with attention applied
        """

        b, c, h, w = x.size()

        # Number of spatial elements per channel (used for normalization)
        n = h * w - 1

        # Compute squared deviation from channel mean
        # Measures how different each neuron is from the average response
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Compute energy function:
        # larger deviation -> higher importance
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        # Generate attention weights and rescale features
        return x * self.activation(y)


if __name__ == '__main__':
    # Example usage
    input_tensor = torch.randn(3, 64, 7, 7)
    model = SimAM()
    output = model(input_tensor)

    print(output.shape)  # should match input shape