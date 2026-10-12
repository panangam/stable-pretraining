# Point-JEPA

`spt.PointJEPA` implements the 64 group, 32 neighbor, 384 width Point-JEPA architecture with a shared PointNet tokenizer, spatial sequencer, EMA teacher encoder, and 192 width predictor. It uses PyTorch FPS/KNN grouping and SPT's `TeacherStudentCallback` and manual optimizer accumulation.

The teacher encodes all patch tokens, while the student encodes only selected context tokens outside the union of targets. Each of the four target blocks is predicted in a separate predictor call using student context features and learned mask tokens with target XYZ positions; queries from different blocks do not attend to each other. Target blocks share a sampled length within each batch, have independent start positions, and may overlap. Predictions are compared with the teacher's encoded target features using Smooth-L1 with beta 2. Downstream evaluation uses all student tokens and concatenates their max and mean pools.

The shared target length matches the official implementation. See [the reference comparison](point_jepa_reference_comparison.md) for verified matches, SPT integration changes, and remaining numerical differences, including encoder position ownership, teacher target normalization, preprocessing, and EMA scheduling. This implementation is not yet numerically equivalent to the released code.

## Short ShapeNet smoke run

The development script reads the ZIP archive without extraction and uses 16 objects by default; it does not require the official split metadata.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/point_jepa_mini.py --archive POINT_JEPA_DEV/data/ShapeNet55.zip --batch-size 2 --steps 2 --limit 16
```

The checkpoint is saved in `POINT_JEPA_DEV/mini_run.ckpt`. The script uses the paper architecture, FP16 mixed precision, and optimizer `frequency=2` for accumulation. SPT uses manual optimization, so set `PointJEPA(accumulate_steps=...)` instead of Lightning's `Trainer(accumulate_grad_batches=...)`.

## Full pretraining configuration

Download the official `train.txt` split from the [Point-JEPA metadata](https://github.com/Ayumu-J-S/Point-JEPA/tree/main/.metadata/ShapeNet55), then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/point_jepa_mini.py --full --split-file train.txt --archive POINT_JEPA_DEV/data/ShapeNet55.zip --batch-size 16 --accumulate-steps 32 --num-workers 8
```

This selects the 500 epoch released configuration: AdamW at 1e-3 with weight decay 0.05, a 30/500 linear warmup from 1e-5, cosine decay to 1e-6, and EMA 0.9995 to 0.99999. The supplement describes a different EMA range of roughly 0.995 to 1.0. The full path uses `spt.Manager`; it was not run during the short smoke validation.

## Frozen ModelNet40 linear evaluation

Install the optional HDF5 dependency with `pip install -e '.[point-cloud]'` and place the standard `modelnet40_ply_hdf5_2048` files under one directory. The evaluation reads train and test HDF5 files, preprocesses each object to 1024 points, concatenates max and mean pooled student token features, and fits a linear SVC with `C=0.012`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/point_jepa_modelnet40_svm.py data/modelnet40_ply_hdf5_2048 POINT_JEPA_DEV/mini_run.ckpt
```

The two-step smoke checkpoint is only for validating the pipeline and cannot establish representation quality. ModelNet40 data is not bundled with this repository.

## Interactive latent PCA viewer

Install the optional viewer dependency with `pip install -e '.[point-cloud-viz]'` if viser is not already installed, then run the script from the repository root and open `http://127.0.0.1:8080`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/point_jepa_viser.py --checkpoint POINT_JEPA_DEV/mini_run.ckpt
```

The viewer selects a random ShapeNet55 object, infers the student's 64 patch latents from 1024 sampled points, fits a three-component PCA for that object, and colors every point in the full cloud by its nearest patch center. Click **Random shape** to load and infer a different object. PCA colors are recalculated for each shape, so their RGB values are not directly comparable across objects. The short smoke checkpoint validates the display path; meaningful learned structure requires a trained checkpoint. Pass `--device cpu` to run without a GPU, or use `--host` and `--port` to change the viewer address.
