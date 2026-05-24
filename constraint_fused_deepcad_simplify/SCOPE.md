# Constraint-Fused DeepCAD Simplify Scope

This package is intentionally independent from `constraint_fused_deepcad`.

## In Scope

- Horizontal / vertical unary constraints only.
- Command-level axis tags injected into the encoder.
- Unary axis reconstruction head and loss.
- Independent train / evaluate entrypoints under `constraint_fused_deepcad_simplify/`.

## Out of Scope

- Pair constraints such as parallel, perpendicular, or collinear.
- Joint command + constraint token encoding.
- Decoder-side cross attention.
- Latent GAN and downstream point-cloud pipelines.
