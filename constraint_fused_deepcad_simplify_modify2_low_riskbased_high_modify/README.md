# Constraint-Fused DeepCAD High Modify

This is the independent High Modify package. Run all commands from the repository root so `python -m` imports resolve consistently.

## Architecture Invariants

- The main decoder path is `decoder(z)`; no `constraint_memory` is accepted by the generation adapter.
- Default latent shape is `(1, batch, 512)`.
- Encoder-side constraint information is fused through `EncoderFused` and `SegmentSeparatedPooling`.
- Unary/pair constraint reconstruction is computed from decoder line features, not directly from `z`.
- Constraint prediction loss is line-only by default.

## Experiment Layout

- Project root: `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify`
- Experiment: `{proj_dir}/cf_high_modify`
- Checkpoints: `{exp_dir}/model`
- TensorBoard logs: `{exp_dir}/log`
- Metrics and manifests: `{exp_dir}/artifacts`
- Config snapshot: `{exp_dir}/config.txt`

## Formal Training

The default formal run follows the repository convention: `batch_size=256` and `nr_epochs=100`.

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify ^
  --batch_size 256 ^
  --nr_epochs 100 ^
  -g 0
```

Use `--continue` to resume an existing run. Use `--force_overwrite` only when intentionally replacing an experiment directory.

## Smoke Checks

Smoke checks are for quick pipeline validation only; they do not replace formal training or full evaluation.

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.verify_p1
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.verify_p2
```

## Evaluation

Full evaluation defaults to the whole `test` split (`sample_count=0`) and writes unified artifacts under `{exp_dir}/artifacts`:

- Reconstruction vectors: `{exp_dir}/artifacts/reconstruction_test_latest`
- Aggregate output: `{exp_dir}/artifacts/test_eval_latest/summary.json`
- Per-sample output: `{exp_dir}/artifacts/test_eval_latest/per_sample_counts.csv`

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.evaluate ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify ^
  --ckpt latest ^
  --eval_split test
```

For sub-split debugging only, pass a small `--sample_count`. If reconstruction vectors already exist, use `--skip_reconstruct` to recompute metrics without decoding again.
