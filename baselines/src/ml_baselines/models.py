from __future__ import annotations

from ml_baselines.torch_utils import TorchUnavailableError, require_torch, resolve_device

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if torch is not None and nn is not None:

    def scatter_add_messages(x, edge_index, *, num_nodes: int | None = None):
        """Manual edge_index aggregation without torch_geometric."""

        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        source = edge_index[0].long()
        target = edge_index[1].long()
        resolved_nodes = int(num_nodes or x.shape[0])
        output = torch.zeros((resolved_nodes, x.shape[-1]), dtype=x.dtype, device=x.device)
        output.index_add_(0, target, x[source])
        return output


    def scatter_mean_messages(x, edge_index, *, num_nodes: int | None = None):
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        target = edge_index[1].long()
        output = scatter_add_messages(x, edge_index, num_nodes=num_nodes)
        degree = torch.zeros((output.shape[0], 1), dtype=x.dtype, device=x.device)
        degree.index_add_(0, target, torch.ones((target.shape[0], 1), dtype=x.dtype, device=x.device))
        return output / degree.clamp_min(1.0)


    class ManualMessagePassing(nn.Module):
        """Small GraphSAGE-style stack using index_add aggregation."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, layers: int = 3) -> None:
            super().__init__()
            if layers <= 0:
                raise ValueError("layers must be positive.")
            self.input = nn.Linear(input_dim, hidden_dim)
            self.layers = nn.ModuleList(
                nn.Linear(hidden_dim * 2, hidden_dim)
                for _ in range(layers)
            )
            self.output = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()

        def forward(self, node_features, edge_index):
            hidden = self.activation(self.input(node_features))
            for layer in self.layers:
                messages = scatter_mean_messages(hidden, edge_index, num_nodes=hidden.shape[0])
                hidden = self.activation(layer(torch.cat([hidden, messages], dim=-1)))
            return self.output(hidden)


    class ItemResourceMLP(nn.Module):
        def __init__(self, item_dim: int, hidden_dim: int, output_dim: int = 1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(item_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, item_features):
            return self.net(item_features)


else:

    def scatter_add_messages(*args, **kwargs):
        raise TorchUnavailableError("scatter_add_messages requires PyTorch.")


    def scatter_mean_messages(*args, **kwargs):
        raise TorchUnavailableError("scatter_mean_messages requires PyTorch.")


    class ManualMessagePassing:
        def __init__(self, *args, **kwargs) -> None:
            raise TorchUnavailableError("ManualMessagePassing requires PyTorch.")


    class ItemResourceMLP:
        def __init__(self, *args, **kwargs) -> None:
            raise TorchUnavailableError("ItemResourceMLP requires PyTorch.")


def build_manual_message_passing(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    device: str = "cpu",
):
    require_torch()
    model = ManualMessagePassing(input_dim, hidden_dim, output_dim, layers=layers)
    return model.to(resolve_device(device))
