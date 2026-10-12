"""Short ShapeNet55 Point-JEPA smoke run, with paper architecture by default."""

import argparse
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader

import stable_pretraining as spt


def main() -> None:
    """Run a bounded mixed-precision ShapeNet55 fit and save a checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("POINT_JEPA_DEV/data/ShapeNet55.zip"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--accumulate-steps", type=int, default=2)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("POINT_JEPA_DEV/mini_run.ckpt")
    )
    args = parser.parse_args()
    if args.full and args.split_file is None:
        parser.error(
            "--full requires the official ShapeNet55 train split via --split-file"
        )
    dataset = spt.data.ShapeNet55(
        args.archive,
        split_file=args.split_file,
        limit=None if args.full else args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.full,
        collate_fn=spt.data.PointCloudTransform().collate,
    )
    model = spt.PointJEPA(
        accumulate_steps=args.accumulate_steps,
        **(
            {
                "embed_dim": 96,
                "encoder_depth": 2,
                "encoder_heads": 3,
                "predictor_dim": 48,
                "predictor_depth": 1,
                "predictor_heads": 3,
            }
            if args.small
            else {}
        ),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        sync_batchnorm=True,
        max_epochs=500 if args.full else None,
        max_steps=-1 if args.full else args.steps,
        callbacks=[spt.TeacherStudentCallback()],
        logger=False,
        enable_checkpointing=args.full,
        enable_model_summary=False,
        limit_val_batches=0,
        limit_train_batches=(len(loader) // args.accumulate_steps)
        * args.accumulate_steps
        if args.full
        else 1.0,
    )
    if args.full:
        spt.Manager(
            trainer=trainer,
            module=model,
            data=spt.data.DataModule(train=loader),
            seed=1,
        )()
    else:
        trainer.fit(model, loader)
        trainer.save_checkpoint(args.output)
        metrics = {
            key: round(value.item(), 4)
            for key, value in trainer.callback_metrics.items()
            if key.startswith("train/")
        }
        print(
            f"steps={trainer.global_step} peak_vram_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f} metrics={metrics} checkpoint={args.output}"
        )


if __name__ == "__main__":
    main()
