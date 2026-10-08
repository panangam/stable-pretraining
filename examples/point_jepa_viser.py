"""Explore ShapeNet55 point clouds colored by Point-JEPA student latent PCA.

Run from the repository root::

    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/point_jepa_viser.py

Open the viser URL printed in the terminal. The button loads another random
shape, recomputes its patch latents, and updates the point cloud in place.
"""

import argparse
from pathlib import Path
import threading
import time

import numpy as np
import torch
import viser

from stable_pretraining.data.point_cloud import ShapeNet55
from stable_pretraining.methods.point_jepa import PointJEPA


def load_model(checkpoint_path: Path, device: torch.device) -> PointJEPA:
    """Restore the Point-JEPA student from an SPT Lightning checkpoint.

    Args:
        checkpoint_path: Path to a Point-JEPA checkpoint.
        device: Inference device.

    Returns:
        Point-JEPA model in evaluation mode on the selected device.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    model = PointJEPA(
        **{
            name: hparams[name]
            for name in (
                "num_groups",
                "group_size",
                "embed_dim",
                "encoder_depth",
                "encoder_heads",
                "predictor_dim",
                "predictor_depth",
                "predictor_heads",
                "lr",
                "weight_decay",
                "ema_decay_start",
                "ema_decay_end",
                "accumulate_steps",
            )
        }
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def pca_rgb(latents: np.ndarray) -> np.ndarray:
    """Map the first three per-shape latent principal components to RGB.

    Args:
        latents: Student patch features [G,D].

    Returns:
        Patch colors [G,3] as uint8 RGB.
    """
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right[:3].T
    lower, upper = np.percentile(scores, (2, 98), axis=0)
    normalized = np.clip((scores - lower) / np.maximum(upper - lower, 1e-8), 0, 1)
    return np.round(normalized * 255).astype(np.uint8)


@torch.inference_mode()
def infer_shape(
    dataset: ShapeNet55,
    index: int,
    model: PointJEPA,
    device: torch.device,
    rng: np.random.Generator,
    num_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Infer point-patch latents and color the full cloud by nearest center.

    Args:
        dataset: ShapeNet55 archive reader without a transform.
        index: Object index in the archive.
        model: Trained Point-JEPA model.
        device: Inference device.
        rng: Generator for choosing the model input points.
        num_points: Point count expected by the pretrained encoder.

    Returns:
        Normalized full-cloud positions [N,3], uint8 colors [N,3], and object ID.
    """
    raw = dataset[index]["points"]
    selected = torch.from_numpy(rng.choice(len(raw), size=num_points, replace=False))
    sample = raw[selected]
    center = sample.mean(dim=0)
    radius = (sample - center).norm(dim=-1).max().clamp_min(1e-8)
    points = (raw - center) / radius
    sample = points[selected].unsqueeze(0).to(device)

    tokens, centers = model.tokenizer(sample)
    tokens, centers = model.sequencer(tokens, centers)
    latents = model.encoder.forward_student(tokens, centers)[0].float().cpu().numpy()
    patch_colors = pca_rgb(latents)

    centers_cpu = centers[0].float().cpu()
    nearest = torch.cdist(points.unsqueeze(0), centers_cpu.unsqueeze(0))[0].argmin(
        dim=1
    )
    colors = patch_colors[nearest.numpy()]
    return points.numpy(), colors, Path(dataset.names[index]).stem


def main() -> None:
    """Serve an interactive point cloud with a random-shape button."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("POINT_JEPA_DEV/data/ShapeNet55.zip")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("POINT_JEPA_DEV/mini_run.ckpt")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_model(args.checkpoint, device)
    dataset = ShapeNet55(args.archive)
    rng = np.random.default_rng(args.seed)
    current_index = int(rng.integers(len(dataset)))
    points, colors, object_id = infer_shape(dataset, current_index, model, device, rng)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    cloud = server.scene.add_point_cloud(
        "/point_jepa/shape",
        points=points,
        colors=colors,
        point_size=args.point_size,
        point_shape="circle",
    )
    button = server.gui.add_button("Random shape")
    info = server.gui.add_markdown(
        f"**Shape:** `{object_id}`  ·  {len(points):,} points  ·  PCA of {model.hparams.num_groups} student patch latents"
    )
    inference_lock = threading.Lock()

    @server.on_client_connect
    def set_camera(client: viser.ClientHandle) -> None:
        """Frame the normalized cloud for each new browser client."""
        client.camera.position = (2.5, 2.0, 2.5)
        client.camera.look_at = (0.0, 0.0, 0.0)

    @button.on_click
    def randomize(_: viser.GuiEvent) -> None:
        """Infer and display a different random object on each click."""
        nonlocal current_index
        with inference_lock:
            button.disabled = True
            try:
                next_index = int(rng.integers(len(dataset) - 1))
                if next_index >= current_index:
                    next_index += 1
                next_points, next_colors, next_id = infer_shape(
                    dataset, next_index, model, device, rng
                )
                cloud.points = next_points
                cloud.colors = next_colors
                info.content = f"**Shape:** `{next_id}`  ·  {len(next_points):,} points  ·  PCA of {model.hparams.num_groups} student patch latents"
                current_index = next_index
            finally:
                button.disabled = False

    print(f"Point-JEPA PCA viewer: http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
