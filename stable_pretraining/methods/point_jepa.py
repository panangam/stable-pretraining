"""Point-JEPA self-supervised pretraining on XYZ point clouds."""

from functools import partial
import math

import torch
from torch import nn

from stable_pretraining import Module
from stable_pretraining.backbone import FlexibleTransformer, TeacherStudentWrapper
from stable_pretraining.backbone.point_cloud import (
    PointJEPAMasking,
    PointSequencer,
    PointTokenizer,
    PointTokenTransformer,
)
from stable_pretraining.forward import point_jepa


class PointJEPA(Module):
    """Point-JEPA with shared tokenizer and EMA teacher transformer.

    Args:
        num_groups: FPS patch centers.
        group_size: Neighbors in each patch.
        embed_dim: Encoder token width.
        encoder_depth: Encoder transformer depth.
        encoder_heads: Encoder attention heads.
        predictor_dim: Predictor hidden width.
        predictor_depth: Predictor transformer depth.
        predictor_heads: Predictor attention heads.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        ema_decay_start: Initial EMA coefficient from the released config.
        ema_decay_end: Final EMA coefficient from the released config.
        accumulate_steps: Number of microbatches per optimizer step in SPT manual optimization.

    Note:
        Use ``TeacherStudentCallback`` in the Trainer to update the EMA teacher.
        The released configuration uses EMA 0.9995 to 0.99999; the paper
        supplement instead describes approximately 0.995 to 1.0.
    """

    def __init__(
        self,
        num_groups: int = 64,
        group_size: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        ema_decay_start: float = 0.9995,
        ema_decay_end: float = 0.99999,
        accumulate_steps: int = 1,
    ) -> None:
        encoder = PointTokenTransformer(embed_dim, encoder_depth, encoder_heads)
        super().__init__(
            forward=point_jepa,
            tokenizer=PointTokenizer(num_groups, group_size, embed_dim),
            sequencer=PointSequencer(),
            masking=PointJEPAMasking(),
            encoder=TeacherStudentWrapper(
                encoder,
                base_ema_coefficient=ema_decay_start,
                final_ema_coefficient=ema_decay_end,
            ),
            predictor_position=nn.Sequential(
                nn.Linear(3, 128), nn.GELU(), nn.Linear(128, predictor_dim)
            ),
            predictor=FlexibleTransformer(
                input_dim=embed_dim,
                hidden_dim=predictor_dim,
                output_dim=embed_dim,
                num_patches=num_groups,
                depth=predictor_depth,
                num_heads=predictor_heads,
                mlp_ratio=4,
                self_attn=True,
                cross_attn=False,
                use_adaln=False,
                pos_embed_type="none",
                drop_path_rate=0.25,
                attn_drop=0.05,
                proj_drop=0.1,
                zero_init_output=False,
                num_prefix_tokens=0,
                add_mask_token=True,
            ),
            optim={
                "optimizer": partial(
                    torch.optim.AdamW, lr=lr, weight_decay=weight_decay
                ),
                "frequency": accumulate_steps,
            },
            hparams={
                "num_groups": num_groups,
                "group_size": group_size,
                "embed_dim": embed_dim,
                "encoder_depth": encoder_depth,
                "encoder_heads": encoder_heads,
                "predictor_dim": predictor_dim,
                "predictor_depth": predictor_depth,
                "predictor_heads": predictor_heads,
                "lr": lr,
                "weight_decay": weight_decay,
                "ema_decay_start": ema_decay_start,
                "ema_decay_end": ema_decay_end,
                "accumulate_steps": accumulate_steps,
            },
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Use the released 30/500 warmup fraction and cosine LR schedule.

        Returns:
            SPT optimizer and scheduler configurations.
        """
        steps = self.trainer.max_steps
        if steps < 1:
            steps = math.ceil(
                self.trainer.estimated_stepping_batches / self.hparams.accumulate_steps
            )
        self.optim["scheduler"] = {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": 0
            if steps == 1
            else max(1, min(steps - 1, round(steps * 30 / 500))),
            "max_steps": steps,
            "warmup_start_lr": 1e-5,
            "eta_min": 1e-6,
        }
        return super().configure_optimizers()
