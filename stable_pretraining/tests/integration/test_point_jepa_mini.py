"""Small Lightning integration test for Point-JEPA manual optimization and EMA."""

import lightning as pl
import torch
from torch.utils.data import DataLoader

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.point_jepa import PointJEPA


def test_point_jepa_two_steps_and_checkpoint(tmp_path):
    """Two accumulated optimizer steps update EMA and survive a checkpoint."""
    model = PointJEPA(
        num_groups=8,
        group_size=8,
        embed_dim=48,
        encoder_depth=1,
        encoder_heads=3,
        predictor_dim=24,
        predictor_depth=1,
        predictor_heads=3,
        accumulate_steps=2,
    )
    initial = model.encoder.teacher.blocks[0].attn.qkv.weight.detach().clone()
    loader = DataLoader(
        [{"points": torch.randn(64, 3)} for _ in range(4)], batch_size=1
    )
    callback = TeacherStudentCallback()
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.fit(model, loader)
    assert trainer.global_step == 2
    assert callback._last_global_step == 2
    assert not torch.equal(model.encoder.teacher.blocks[0].attn.qkv.weight, initial)
    path = tmp_path / "point_jepa.ckpt"
    trainer.save_checkpoint(path)
    assert torch.load(path, map_location="cpu", weights_only=False)["global_step"] == 2
