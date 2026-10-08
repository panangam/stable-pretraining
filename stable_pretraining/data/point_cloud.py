"""Fixed-size XYZ point-cloud datasets and preprocessing for Point-JEPA."""

import io
from collections.abc import Callable
from pathlib import Path
import zipfile

import numpy as np
import torch

from .datasets import Dataset


class PointCloudTransform:
    """Subsample, scale, center, normalize, and optionally rotate XYZ points.

    Args:
        num_points: Number of points returned per object.
        training: Enable random scaling and vertical-axis rotation.
    """

    def __init__(self, num_points: int = 1024, training: bool = True) -> None:
        self.num_points = num_points
        self.training = training

    def __call__(self, sample: dict) -> dict:
        """Preprocess a point-cloud sample.

        Args:
            sample: Dictionary containing XYZ ``points``.

        Returns:
            Sample with normalized ``points`` [num_points,3].
        """
        points = sample["points"]
        if self.training:
            indices = torch.randperm(len(points))[: self.num_points]
            scale = torch.empty(2).uniform_(0.8, 1.2)
            points = points[indices] * torch.stack([scale[0], torch.ones(()), scale[1]])
        else:
            points = points[: self.num_points]
        points = points - points.mean(0, keepdim=True)
        points = points / points.norm(dim=-1).max().clamp_min(1e-8)
        if self.training:
            angle = torch.rand(()) * (2 * torch.pi)
            c, s = torch.cos(angle), torch.sin(angle)
            rotation = torch.stack(
                [
                    torch.stack([c, torch.zeros_like(c), s]),
                    torch.tensor([0.0, 1.0, 0.0]),
                    torch.stack([-s, torch.zeros_like(c), c]),
                ]
            )
            points = points @ rotation
        sample["points"] = points
        return sample


class ShapeNet55(Dataset):
    """Read ShapeNet55 NPY objects directly from its official ZIP archive.

    Args:
        archive: Path to ShapeNet55.zip.
        split_file: Optional official split file containing object IDs.
        transform: Sample transform, usually ``PointCloudTransform``.
        limit: Optional maximum object count for development runs.
    """

    def __init__(
        self,
        archive: str | Path,
        split_file: str | Path | None = None,
        transform: Callable | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.archive = str(archive)
        self._zip = None
        with zipfile.ZipFile(self.archive) as handle:
            names = sorted(
                name
                for name in handle.namelist()
                if name.endswith(".npy") and "/shapenet_pc/" in name
            )
        if split_file is not None:
            ids = {Path(name).stem for name in Path(split_file).read_text().split()}
            names = [name for name in names if Path(name).stem in ids]
        self.names = names[:limit]

    def __len__(self) -> int:
        """Return the number of selected objects.

        Returns:
            Dataset length.
        """
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        """Read one ShapeNet object.

        Args:
            index: Dataset position.

        Returns:
            SPT sample containing XYZ coordinates.
        """
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive)
        with self._zip.open(self.names[index]) as source:
            points = torch.from_numpy(
                np.load(io.BytesIO(source.read()))[:, :3].copy()
            ).float()
        return self.process_sample({"points": points, "sample_idx": index})

    def __getstate__(self) -> dict:
        """Leave ZIP handles out of DataLoader worker serialization.

        Returns:
            Pickleable dataset state.
        """
        state = super().__getstate__()
        state["_zip"] = None
        return state


class ModelNet40H5(Dataset):
    """Read the standard ModelNet40 HDF5 split for frozen-feature evaluation.

    Args:
        root: Directory containing ``ply_data_train*.h5`` and ``ply_data_test*.h5``.
        split: ``train`` or ``test``.
        transform: Point-cloud preprocessing transform.
    """

    def __init__(
        self, root: str | Path, split: str, transform: Callable | None = None
    ) -> None:
        super().__init__(transform=transform)
        import h5py

        points, labels = [], []
        for path in sorted(Path(root).glob(f"ply_data_{split}*.h5")):
            with h5py.File(path) as source:
                points.append(torch.from_numpy(source["data"][:].copy()))
                labels.append(torch.from_numpy(source["label"][:].reshape(-1).copy()))
        self.points = torch.cat(points)
        self.labels = torch.cat(labels)

    def __len__(self) -> int:
        """Return the split size.

        Returns:
            Dataset length.
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        """Read one ModelNet object.

        Args:
            index: Dataset position.

        Returns:
            SPT sample containing XYZ points and class label.
        """
        return self.process_sample(
            {
                "points": self.points[index],
                "label": self.labels[index],
                "sample_idx": index,
            }
        )
