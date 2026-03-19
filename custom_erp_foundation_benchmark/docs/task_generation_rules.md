# Task Generation Rules

This document turns the benchmark blueprint into concrete generation rules.

## 1. Core V1 tasks

### `entity_identify`

Construction:

- choose one verified anchor entity
- build a referring phrase from `reground_query` or `caption_brief`
- create 3 distractor labels from scene objects first, then fallback label pool

Filtering:

- reject if target label is too generic
- reject if the referring phrase is ambiguous
- prefer same-scene distractors

### `attribute_understanding`

Construction:

- choose one anchor entity with normalized attribute values
- prefer `color`, `material`, `shape`, `condition` in that order
- create 3 distractor values from attribute-specific vocab pools

Filtering:

- reject unsupported free-form attributes
- reject if the attribute is not visible from ERP evidence

### `referring_grounding`

Construction:

- choose one target anchor and 3 distractor anchors
- expose 4 candidate regions as multiple choice
- question uses the referring phrase only

Filtering:

- target must be uniquely referable
- distractors should be semantically or spatially close enough to matter

### `existence`

Construction:

- one positive item from an actually present category
- one negative item from a realistic absent category

Filtering:

- reject if positive category instances are uncertain
- reject if negative category is too implausible for the scene

### `counting`

Construction:

- choose a category with verified instances
- answers use `0/1/2/3/4/5+`

Filtering:

- reject if count depends on tiny or heavily occluded instances

### `scene_composition`

Construction:

- create one true statement from top anchors or scene tags
- create 3 distractor statements by swapping scene type or object arrangement

Filtering:

- true statement must be grounded in visible objects
- avoid text-only prior shortcuts

### `absolute_direction`

Construction:

- derive coarse direction from `lon_lat`
- for v1, collapse to `front/right/back/left`

Filtering:

- reject borderline bins unless intentionally hard
- keep yaw coverage balanced over the whole pool

### `relative_direction`

Construction:

- choose anchor-support pairs with sufficient angular margin
- derive coarse relation from pairwise angular delta

Filtering:

- reject low-margin pairs
- balance left/right/front/back

### `seam_continuity`

Construction:

- only use seam-adjacent or seam-crossing entities
- ask whether the apparent split still belongs to one object

Filtering:

- manual verification is mandatory
- reject if seam evidence is weak

### `rotation_consistency`

Construction:

- synthetically rotate the ERP panorama in yaw
- ask how the target direction should transform

Filtering:

- manual spot-check every rotation rule
- prefer 90-degree and 180-degree rotations

### `polar_distortion_awareness`

Construction:

- only use pole-proximal entities
- ask whether a visible deformation is a projection artifact

Filtering:

- manual verification is mandatory
- reject naturally elongated real objects

### `depth_ordering`

Construction:

- choose pairs with valid depth and sufficient depth gap
- ask which object is closer/farther

Filtering:

- reject noisy or low-margin depth pairs

### `distance_bucket`

Construction:

- choose one anchor with valid depth
- bucket into `near/medium/far`

Filtering:

- reject threshold-near cases unless intentionally hard
- keep bucket balance across the whole benchmark

## 2. Advanced extension tasks

### `relative_3d_position`

Construction:

- requires trusted `entity_xyz_camera`
- choose strongest coarse relation among horizontal, vertical, and front-back

Filtering:

- reject low-margin cases
- manual review recommended

### `object_search_initial_turn`

Construction:

- use target yaw relative to current heading
- answer space: `turn_left / turn_right / go_forward / turn_back`

Filtering:

- target must be unique
- forward heading convention must be globally fixed

### `path_search_initial_action`

Construction:

- requires opening or free-space metadata
- ask for the first plausible action only

Filtering:

- manual verification is mandatory
- reject if path plausibility depends on hidden geometry
