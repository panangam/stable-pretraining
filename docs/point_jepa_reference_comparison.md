# Point-JEPA reference comparison

Compared against the local official checkout at `/home/oras/codes/brown_cs/cs2952x/Point-JEPA/`, HEAD `c36b51ca20a07f3fa3c75f6b9d3bc0e29b45401c`. Configuration values below come from `configs/Point-JEPA/pretraining/shapenet.yaml`, rather than constructor defaults that the YAML overrides. The implementation has been aligned with the released configuration and execution path. The tables distinguish matching computation from the remaining SPT and hardware adaptations.

## Confirmed matches

| Component | Official behavior and current SPT behavior | Official source |
| --- | --- | --- |
| Target length | One ratio is sampled per `TargetSampler.sample()` call, before looping over all four blocks. All blocks therefore share one length, `max(1, int(T * ratio))`: 9–12 tokens at T=64. | `modules/target_sampler.py`: `sample`, `select_embed_contiguous` |
| Target locations | Each block has an independently sampled start. Indices are shared across the batch; blocks may overlap. | `modules/target_sampler.py`: `select_embed_contiguous` |
| Context | Exclude the union of target indices, sample a ratio in 0.40–0.75 of the remaining indices, then take a contiguous slice of that filtered list. | `modules/context_sampler.py`: `sample`, `get_context_contiguous_idx` |
| Sequencing | Start at the center with minimum XYZ coordinate sum, then repeatedly visit the nearest unvisited center. | `modules/point_sequencer.py`: `sort_iterative_nearest`, `min_start=True` |
| Tokenizer ownership | Run one shared tokenizer before context/target branching. Only the encoder Transformer is teacher/student duplicated in the reference. | `models/point_jepa.py`: `_perform_step`, `setup` |
| Encoder/predictor sizes | Encoder 12 layers, width 384, 6 heads; predictor 6 layers, width 192, 6 heads. Encoder position embeddings are added at every layer; predictor positions are added once. | Pretraining YAML; `modules/transformer.py` |
| Teacher/context inputs | Teacher encodes the complete sequence without gradients. Student encodes only selected context. Targets are selected from teacher Transformer representations. | `models/point_jepa.py`: `get_target_block`, `forward`, `generate_targets` |
| Predictor information | Context features plus context positions; learned mask token plus target positions. Hidden patch PointNet features are not predictor inputs. Each target block gets its own predictor call. | `modules/transformer.py`: `TransformerPredictor.forward`; `models/point_jepa.py`: `forward` |
| Loss | Mean Smooth-L1 with beta 2 over predictions and targets, including repeated contributions for overlapping blocks. SPT stores tensors as [B,M,K,D], reference as [M,B,K,D]; this layout difference does not change the loss. | `models/point_jepa.py`: `training_step`, `forward` |
| Downstream | Full student sequence, concatenate max pool followed by mean pool, then linear SVC with C=0.012. | `models/point_jepa.py`: `svm_validation` |
| Optimizer/configuration | AdamW LR 1e-3, weight decay 0.05; 500 epochs, warmup 30 epochs from 1e-5, cosine minimum 1e-6; EMA endpoint values 0.9995 and 0.99999. The EMA ramp and epoch LR values now also match the reference. | Pretraining YAML |

The shared target length is confirmed reference behavior, not a memory-saving assumption.

## Changes made for SPT integration

| Area | Official implementation | SPT integration and effect |
| --- | --- | --- |
| Module contract | Raw point/label tuples, custom Lightning training and validation steps. | Dictionary batches and a reusable `point_jepa` forward function return loss, embeddings, predictions, targets, and diagnostics. `PointJEPA` delegates optimization and logging to `spt.Module`. |
| Teacher lifecycle | Custom `EMA` object and `optimizer_step()` override. | `TeacherStudentWrapper` and `TeacherStudentCallback` update after optimizer steps. Accumulation uses SPT optimizer `frequency`; Lightning `accumulate_grad_batches` is not used. |
| EMA schedule | Linear ramp over optimizer updates; initial updates copy student weights before moving averaging begins. | A Point-JEPA-specific TeacherStudentWrapper subclass reproduces the linear ramp and first two initialization copies. The existing TeacherStudentCallback still triggers updates after optimizer steps; update count and ramp length are checkpointed. |
| LR schedule | Bolts warmup/cosine scheduler stepped by epoch. | A LambdaLR maps optimizer-update count to epoch boundaries, reproducing the reference warmup denominator (30−1), peak at epochs 29/30, and cosine endpoint. SPT continues stepping it after optimizer updates, but LR remains constant inside each epoch. Bounded max_steps runs compress the schedule into their update budget; they are pipeline checks, not the 500-epoch schedule. |
| Grouping backend | PyTorch3D FPS/KNN. | Portable PyTorch FPS and `torch.cdist`/top-k avoid a PyTorch3D dependency and support small CPU checks. Both now use random-start FPS. KNN ordering is immaterial to PointNet pooling. Backend tie-breaking, RNG streams, performance, and floating-point details can still differ. |
| Data storage | Extracted NPY or preloaded NPZ datasets, official split metadata and labels. | Direct ZIP reading, optional split-file filtering, sample dictionaries, and a development-only object limit. ShapeNet class labels are omitted because pretraining does not consume them. |
| Preprocessing location | Batched transforms on the training device inside Lightning steps. | Training and SVM examples collate clouds before applying batched FPS/augmentation on CPU. Scaling and rotation draws are shared across each batch, matching the reference distribution. Device/RNG streams differ. Direct per-object Dataset transforms remain available for callers that explicitly choose them. |
| Evaluation orchestration | Periodic ModelNet40 and ScanObjectNN SVM evaluations during validation. | Separate ModelNet40 frozen-feature script. The mini/full training example disables validation and does not reproduce the reference's periodic evaluations. |
| Runtime | Released batch size 512, synchronized BatchNorm, FP16, 500 epochs. | Single-device FP16 example with sync_batchnorm enabled and smaller microbatches and accumulation for 8GB VRAM. Batch 16 × accumulation 32 gives optimizer batch 512, but tokenizer BatchNorm still sees microbatches, so the runs are not numerically equivalent. `--small` explicitly selects a reduced architecture for development. Full runs use `spt.Manager`; the bounded smoke script uses Trainer directly. |

## Numerical differences corrected

| Area | Current matching behavior | Official source |
| --- | --- | --- |
| Encoder XYZ position MLP | One trainable MLP outside the EMA wrapper; teacher uses it under no-grad and student trains it. Encoders receive position embeddings explicitly. | `models/point_jepa.py`: `positional_encoding`, `get_target_block`, `forward` |
| Teacher target normalization | Return the final block output before learned encoder output normalization, then apply non-affine layer normalization. Student/downstream features retain learned output normalization. | `models/point_jepa.py`: `generate_targets`; `modules/transformer.py`: hidden states |
| Patch FPS start | Random starting point per cloud. | `modules/tokenizer.py`: `PointcloudGrouping.forward` |
| Input subsampling | Random-start FPS to 1024 points for training, evaluation, and visualization; do nothing when input already has the requested point count. | `utils/transforms.py`: `PointcloudSubsampling` |
| Scaling and rotation | Scale all XYZ axes in [0.8,1.2], center, normalize to unit sphere, and rotate around Y. The official transform builder does not forward the YAML's symmetry argument, so no reflections are enabled in either implementation. | `models/point_jepa.py`: `build_transformation`; `utils/transforms.py` |
| Local PointNet details | Disable biases on convolutions immediately before BatchNorm; concatenate pooled features before local features. Widths and shared-MLP/pooling structure match. | `modules/Pointnet.py`: `PointNetSmall` |
| Predictor stochastic depth | Constant 0.25 at every predictor layer, matching the actual scalar passed by the reference; encoder stochastic depth still ramps from 0 to 0.25. | `models/point_jepa.py`: predictor construction; `modules/transformer.py` |
| Mask-token initialization | Active predictor mask token initialized to zero. The reference's separate unused model-level mask token is omitted. | `modules/transformer.py`: `TransformerPredictor` |
| Collapse diagnostics | Flatten batch/block/token axes, then channel-wise sample std with correction=1. | `models/point_jepa.py`: `token_std` |

## Remaining differences and scope

The generic FlexibleTransformer retains its query projection, but its output is fully replaced by the learned mask token for every target query; it has no effect or gradient in this path. Its generic implementation, attention backend and RNG ordering prevent a claim of bit-for-bit reproduction, even when evaluation outputs match under mapped weights. Downstream evaluation and visualization skip spatial sequencing, matching the official SVM path; sequencing is used for pretraining masking.

The integration implements the released XYZ/KNN/greedy/contiguous/final-layer path. Alternate reference settings such as radius grouping, different sequencing/sampling modes, intermediate teacher layers and other target normalization policies are not exposed. Official checkpoint parameter names are not imported automatically.

Checkpoint loading uses the current architecture directly, without migration of earlier SPT parameter names, positional MLPs, convolution biases, or fusion channels. Checkpoints from the earlier development implementation must be regenerated. Current checkpoints preserve EMA update progress.

## Validation

`POINT_JEPA_DEV/compare_reference.py` directly loads the local official PointNet, Transformer and EMA classes on CPU. Under mapped weights it checks matching mini PointNet outputs, student outputs, pre-normalization teacher targets, and predictor outputs. It also compares initialization copies, linear EMA coefficients and weights over ten updates, and checks reference LR values at epochs 0, 29, 30 and 500. These checks pass. Stochastic training masks are not expected to be bit-identical across the implementations.

The focused tests exercise masking, forward/backward gradient ownership, shared positional encodings, teacher target normalization, accumulation/EMA/checkpointing, batched transforms and schedule boundaries. `POINT_JEPA_DEV/validate_alignment.py` runs two optimizer steps on real ShapeNet55 data using a reduced model on CPU, saves/reloads a checkpoint, performs downstream inference, and checks PCA inference from the current checkpoint. No full pretraining, GPU program or accuracy reproduction is part of this lightweight validation.
