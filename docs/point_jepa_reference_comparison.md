# Point-JEPA reference comparison

Compared against the local official checkout at `/home/oras/codes/brown_cs/cs2952x/Point-JEPA/`, HEAD `c36b51ca20a07f3fa3c75f6b9d3bc0e29b45401c`. Configuration values below come from `configs/Point-JEPA/pretraining/shapenet.yaml`, rather than constructor defaults that the YAML overrides. This audit changes documentation only; the current SPT implementation is an integration of the core method, with the numerical differences listed below.

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
| Optimizer/configuration | AdamW LR 1e-3, weight decay 0.05; 500 epochs, warmup 30 epochs from 1e-5, cosine minimum 1e-6; EMA endpoint values 0.9995 and 0.99999. Matching endpoint values do not imply matching schedule shapes. | Pretraining YAML |

The shared target length is confirmed reference behavior, not a memory-saving assumption. SPT caps target length at T−1 for tiny custom configurations; that cap does not affect the default ratio and T=64.

## Changes made for SPT integration

| Area | Official implementation | SPT integration and effect |
| --- | --- | --- |
| Module contract | Raw point/label tuples, custom Lightning training and validation steps. | Dictionary batches and a reusable `point_jepa` forward function return loss, embeddings, predictions, targets, and diagnostics. `PointJEPA` delegates optimization and logging to `spt.Module`. |
| Teacher lifecycle | Custom `EMA` object and `optimizer_step()` override. | `TeacherStudentWrapper` and `TeacherStudentCallback` update after optimizer steps. Accumulation uses SPT optimizer `frequency`; Lightning `accumulate_grad_batches` is not used. |
| EMA schedule | Linear ramp over optimizer updates; initial updates copy student weights before moving averaging begins. | SPT uses its existing cosine coefficient schedule based on epoch and warm-initializes the teacher at construction. Updates occur only at optimizer-step boundaries. This is a training difference, not an identical schedule expressed through another API. |
| LR schedule | Bolts warmup/cosine scheduler stepped by epoch. | SPT warmup/cosine scheduler stepped by optimizer update, preserving the 30/500 warmup fraction and LR endpoints. Tiny runs round the warmup length and special-case a one-step run. |
| Grouping backend | PyTorch3D FPS/KNN. | Portable PyTorch FPS and `torch.cdist`/top-k avoid a PyTorch3D dependency and support small CPU checks. KNN ordering is immaterial to the PointNet pooling, but FPS start behavior differs as listed below. |
| Data storage | Extracted NPY or preloaded NPZ datasets, official split metadata and labels. | Direct ZIP reading, optional split-file filtering, sample dictionaries, and a development-only object limit. ShapeNet class labels are omitted because pretraining does not consume them. |
| Preprocessing location | Batched transforms on the training device inside Lightning steps. | Per-object transforms in the Dataset, normally on CPU. Augmentation draws are independent per object rather than shared across a batch. |
| Evaluation orchestration | Periodic ModelNet40 and ScanObjectNN SVM evaluations during validation. | Separate ModelNet40 frozen-feature script. The mini/full training example disables validation and does not reproduce the reference's periodic evaluations. |
| Runtime | Released batch size 512, synchronized BatchNorm, FP16, 500 epochs. | Single-device FP16 example with smaller microbatches and accumulation for 8GB VRAM. Batch 16 × accumulation 32 gives optimizer batch 512, but tokenizer BatchNorm still sees microbatches, so the runs are not numerically equivalent. `--small` explicitly selects a reduced architecture for development. Full runs use `spt.Manager`; the bounded smoke script uses Trainer directly. |

## Remaining numerical differences and unsupported options

These choices are not required by SPT and should not be presented as faithful reproduction of the released code.

| Area | Official implementation | Current SPT implementation / consequence |
| --- | --- | --- |
| Encoder XYZ position MLP | A single `positional_encoding` outside the EMA student/teacher Transformers. The teacher uses it under no-grad, while the student trains it. | `PointTokenTransformer.position` is inside the EMA wrapper, giving separate student and teacher position MLPs with EMA teacher weights. |
| Teacher target normalization | Select the final block's hidden state before the encoder's learned output LayerNorm, then apply non-affine `F.layer_norm`. | Apply the encoder's learned output LayerNorm, then non-affine `F.layer_norm`. Initially similar, but learned affine parameters can make the targets differ. |
| Patch FPS start | Random starting point for each cloud. | Start from the point furthest from the origin. Both perform FPS, but neighborhoods need not match. |
| Input subsampling | Random-start FPS to 1024 points for both train and validation. | Random subset for training; first 1024 points for evaluation. This changes spatial coverage, including downstream evaluation. |
| Scaling | Scaling on all three axes in [0.8,1.2]. The model's transform builder does not pass the YAML's `transformation_scale_symmetries` argument, so configured reflections are not actually enabled in this checkout. | Scale X/Z only and leave Y fixed; no reflections. Centering, unit-sphere normalization, and random Y-axis rotation have the same intended operations. |
| Local PointNet details | Bias disabled on the two convolutions immediately before BatchNorm; concatenate pooled features before local features. | Those convolutions have biases; concatenate local features before pooled features. Widths, shared-MLP/pooling architecture and permutation invariance match. Concatenation order is equivalent under a corresponding fusion-weight permutation, but raw reference weights cannot be copied directly. |
| Predictor stochastic depth | Pass scalar 0.25 into the predictor encoder, applying 0.25 at every layer. The computed `dpr_predictor` list in the reference model is unused. | `FlexibleTransformer` ramps drop-path probabilities from 0 to 0.25. The student encoder ramp matches the reference. Dropout 0.1 and attention dropout 0.05 match the released YAML for both stacks. |
| Mask-token initialization | Predictor mask token starts at zero. The separately initialized model-level mask token is unused by `TransformerPredictor.forward`. | FlexibleTransformer initializes its active mask token from a normal distribution with std 0.02. |
| Predictor query projection | No query projection; hidden query features begin at zero and receive mask/position embeddings. | Generic FlexibleTransformer has a query projection, whose output is replaced entirely by the mask token when all query positions are masked. It has no effect or gradient in this Point-JEPA path. |
| Collapse diagnostics | Flatten batch/block/token axes, then channel-wise sample std (PyTorch default correction=1). | Same axes but population std (`unbiased=False`). This affects reported diagnostics, not the Smooth-L1 objective. |
| Alternate reference settings | Supports radius grouping, different sequencing/sampling modes, selected/intermediate teacher layers and several target normalization policies. | Implements the default XYZ/KNN/greedy/contiguous/final-layer path, rather than exposing every reference option. Official checkpoint parameter names also differ; the existing rename compatibility only covers earlier SPT checkpoints. |

## Validation

The existing six Point-JEPA tests passed after switching to separate target-block predictor calls, including a two-optimizer-step CPU Lightning run with accumulation, EMA updates, and checkpoint saving. This comparison itself did not launch training or GPU programs. Source inspection establishes the differences above; it does not establish matching full-training accuracy or numerical equivalence.
