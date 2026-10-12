"""Shared point sampling without backbone/data import dependencies."""

import torch


@torch.no_grad()
def farthest_point_indices(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Sample point indices using random-start batched FPS.

    Args:
        points: XYZ clouds [B,N,3].
        num_points: Number of points to select.

    Returns:
        Selected indices [B,num_points].
    """
    batch, count, _ = points.shape
    distances = torch.full((batch, count), float("inf"), device=points.device)
    farthest = torch.randint(count, (batch,), device=points.device)
    indices = torch.empty(batch, num_points, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for index in range(num_points):
        indices[:, index] = farthest
        center = points[rows, farthest]
        distances = torch.minimum(
            distances, (points - center[:, None]).square().sum(-1)
        )
        farthest = distances.argmax(-1)
    return indices
