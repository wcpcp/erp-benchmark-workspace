# Annotation Guidelines

## General rule

This benchmark is stricter than the training data.

Every official benchmark item should be checked for:

- visual answerability from the ERP image
- unique answer
- no hidden assumptions
- no language-only shortcut
- no unresolved ambiguity near task thresholds

## Task-specific checks

### `entity_identify`

- confirm the referring phrase uniquely matches the target
- confirm distractor labels are plausible but wrong
- reject if target identity is too generic or uncertain

### `attribute_understanding`

- only use normalized attribute vocab
- reject if the attribute is not clearly visible
- reject if multiple options could reasonably fit

### `referring_grounding`

- all candidate regions must be visible and valid
- target region must be uniquely grounded
- distractors should be spatially or semantically confusable

### `existence` and `counting`

- reject scenes with unverified tiny or heavily occluded instances
- ensure negative categories are realistic but absent
- count only verified instances

### `absolute_direction` and `relative_direction`

- reject low-margin directional cases unless intentionally in a hard slice
- ensure seam behavior is labeled if relevant
- check that the answer requires panorama-wide direction, not local crop only

### `seam_continuity`

- manual verification is mandatory
- only keep items where seam continuity is clearly supported by mask topology or
  cross-view evidence

### `rotation_consistency`

- verify the transformed answer using the exact applied ERP yaw rotation
- keep question wording explicit about the rotation amount

### `polar_distortion_awareness`

- manual verification is mandatory
- reject if the real object shape is itself highly elongated or deformable

### `depth_ordering` and `distance_bucket`

- reject if depth source is noisy or missing
- reject if both objects lie too close to the bucket or comparison threshold

### `relative_3d_position`

- only use when `entity_xyz_camera` is trusted
- use coarse relations only

### `object_search_initial_turn`

- the target must be unambiguous
- current forward heading convention must be fixed across the benchmark

### `path_search_initial_action`

- only use when free-space or opening metadata is trusted
- manual verification is mandatory
