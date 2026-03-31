# Yaw-Shift / Seam-Relocation Robustness Protocol

This document describes the recommended **supplementary** evaluation protocol
for ERP rotation consistency.

It is intentionally **not** released as a standalone scored QA task in the v1
benchmark.

## Why this is a supplementary protocol instead of a QA task

The goal is not to ask whether the model can solve another wording variant.
The goal is to test whether the model has learned ERP panoramas as
**yaw-reparameterizable spherical representations** rather than as fixed-width
rectangular images with column-specific shortcuts.

Standalone QA formulations are not clean enough:

- if the prompt states that two ERP panoramas come from the same capture point,
  it leaks the key equivalence relation
- if the prompt does not state that, the task becomes a mixture of scene
  matching, seam relocation, and viewpoint-change interpretation
- if the prompt simply asks for ordinary localization before and after
  rotation, strong task skill can hide whether representation consistency was
  truly learned

For this reason, the clean design is:

- start from already valid benchmark items
- apply a known yaw shift to the ERP input
- rerun the same task
- check whether predictions obey the expected transformation law

## What this protocol tests

This protocol does **not** test a new spatial skill.

It tests whether strong benchmark performance is grounded in:

- spherical / camera-centered geometry

instead of:

- shortcuts tied to the ERP image coordinate system
- seam placement bias
- fixed column-position priors

So the core scientific question is:

> Does the model treat yaw-shifted ERP panoramas as equivalent
> reparameterizations of the same omnidirectional scene?

## Protocol construction

For a benchmark scene:

1. take the original ERP panorama
2. synthesize one or more yaw-shifted ERP panoramas
3. rerun selected benchmark tasks on the shifted inputs
4. compare predictions before and after the shift

The protocol is especially useful because yaw shifting also relocates the ERP
seam. This makes it a natural **seam-relocation robustness** stress test.

## Recommended task families

### Equivariant tasks

These tasks should change in a predictable way after a known yaw shift:

- `referring_grounding_bfov`
- `absolute_direction_mc`

Expected behavior:

- `referring_grounding_bfov`
  - the predicted BFOV yaw should shift by the known offset
  - pitch and angular extent should remain stable
- `absolute_direction_mc`
  - the predicted sector should update according to the known yaw shift

### Invariant tasks

These tasks should remain semantically unchanged after a yaw shift:

- `relative_direction_mc`
- `camera_rotation_transform_mc`
- `object_conditioned_reorientation_mc`
- `observer_distance_choice`
- `polar_shape_recovery_mc`

Expected behavior:

- the answer should remain the same
- failures after yaw shift indicate dependence on ERP image coordinates rather
  than on the underlying spherical or 3D scene structure

### Seam-relocation stress tasks

Seam-sensitive items are not part of the current executable protocol set.
They are best treated as a future stress slice because seam relocation changes
the seam-sensitive question construction itself.

Recommended future evaluation:

- regenerate seam-sensitive items after yaw shift
- compare seam-task accuracy before and after seam relocation
- report the accuracy drop as a seam-relocation robustness measure

## Suggested metrics

Report at least:

1. base accuracy on original ERP items
2. shifted accuracy on yaw-shifted ERP items
3. robustness gap = shifted accuracy - base accuracy

Additionally:

- for `referring_grounding_bfov`
  - transformed BFOV IoU
  - transformed center error
- for invariant tasks
  - answer-consistency rate
- for seam-sensitive tasks
  - seam-relocation drop

## Recommended interpretation

If a model truly understands ERP as a spherical representation:

- equivariant tasks should update correctly under yaw shift
- invariant tasks should stay stable
- seam-sensitive tasks should remain robust when the seam moves

If performance collapses after yaw shift while the original benchmark score is
high, then the original score may rely too much on:

- fixed ERP column statistics
- seam placement priors
- non-geometric image shortcuts

This is why the protocol is valuable.

It does not replace the core benchmark, but it provides a stronger diagnosis of
**what kind of representation the model has actually learned**.
