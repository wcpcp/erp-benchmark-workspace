# ERP Spatial Benchmark: Appendix Task Table

This document summarizes the **current official ERP Spatial Benchmark v1** as implemented in:

- `/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/build_benchmark.py`
- `/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/evaluate_predictions.py`

It is intended as a paper-ready appendix reference for:

- question schema
- ground-truth rule
- filtering rule
- scoring rule

Rotation consistency is **not** a scored standalone benchmark task in v1. Its supplementary protocol is documented separately in:

- `/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/ROTATION_PROTOCOL.md`

## Global Benchmark Policy

| Item | Current Rule |
|---|---|
| Benchmark scope | ERP-only spatial benchmark for omnidirectional scene understanding |
| Core ability groups | `spherical_localization_and_panoramic_topology`, `viewpoint_conditioned_spatial_updating`, `observer_centered_3d_layout_understanding`, `erp_representation_understanding` |
| Current scored tasks | `referring_grounding_bfov`, `absolute_direction_mc`, `relative_direction_mc`, `camera_rotation_transform_mc`, `object_conditioned_reorientation_mc`, `observer_distance_choice`, `relative_3d_position_mc`, `seam_continuity_mc`, `polar_shape_recovery_mc` |
| Release style | Single public benchmark with public answers |
| Candidate generation | Per-scene candidate pool, then public-pool selection |
| Final pool balancing | Controlled by `target_public_per_task` and `max_per_scene_per_task` |
| Default per-scene cap | `max_per_scene_per_task = 1` |
| Manual review queue | Fragile tasks are exported to `review_queue.jsonl` |
| Manual review tasks | `seam_continuity_mc`, `polar_shape_recovery_mc`, `relative_3d_position_mc` |
| Primary metric | `ability_group_macro_accuracy` |
| Secondary metrics | `task_macro_score`, `overall.accuracy`, `by_task`, `by_ability_group`, `by_diagnostic_slice` |
| Grounding-specific metrics | `mean_bfov_iou`, `mean_center_error_deg` |

## Official Scoring Policy

| Scope | Rule |
|---|---|
| Main headline score | Macro average over the four `ability_group` accuracies |
| Item-level metric | Closed-form exact match for all multiple-choice tasks; seam-aware spherical BFOV IoU for grounding |
| Accepted prediction forms for MC tasks | Correct option key such as `A/B/C/D`, exact option text, or normalized `answer_text` |
| Grounding prediction format | A BFOV tuple or list in the form `[yaw, pitch, x_fov, y_fov]` |
| Grounding correctness threshold | Correct iff seam-aware spherical BFOV IoU `>= 0.5` |
| Overall accuracy | Micro accuracy over all scored items |
| Task macro score | Macro average over task-level accuracies |
| Diagnostic slices | Slice-level accuracy over tags such as `seam`, `pole`, `rotation`, `rotation_protocol` |

## Core Task Table

| Task ID | Ability Group | Question Schema | Ground-Truth Rule | Filtering Rule | Scoring Rule |
|---|---|---|---|---|---|
| `referring_grounding_bfov` | `spherical_localization_and_panoramic_topology` | `Provide the BFOV [yaw, pitch, x_fov, y_fov] of {target_ref} in the full ERP panorama.` or `What is the BFOV [yaw, pitch, x_fov, y_fov] for {target_ref} in this ERP image?` | Use the target entity's resolved BFOV directly as the reference answer: `[yaw, pitch, x_fov, y_fov]`. `answer_text` is the same BFOV rendered in text form. | Entity must have a valid `resolved_bfov`; otherwise no item is built. | Parse predicted BFOV and compute seam-aware spherical BFOV IoU against GT. Item is correct iff IoU `>= 0.5`. Extra regression metrics: mean BFOV IoU and mean BFOV center error in degrees. |
| `absolute_direction_mc` | `spherical_localization_and_panoramic_topology` | `In the complete 360 panorama, which direction sector best contains {target_ref}?` or `Which absolute panorama sector best matches {target_ref} in the ERP image?` | Convert target yaw to one of 8 absolute sectors: `front`, `front-right`, `right`, `back-right`, `back`, `back-left`, `left`, `front-left`. Distractors come from neighboring and opposite sectors. | Drop items whose yaw is too close to sector boundaries: `absolute_sector_margin < 8°`. | Exact match on correct option key or option text. |
| `relative_direction_mc` | `spherical_localization_and_panoramic_topology` | `On the panoramic ring, where does {target_ref} fall relative to {reference_ref}?` or `Around the full panorama ring, what is the angular relation of {target_ref} to {reference_ref}?` | Compute `delta_yaw = wrap(yaw_target - yaw_reference)` and assign one of 5 panoramic relation labels: `right`, `back-right`, `opposite`, `back-left`, `left`. | Filter if `abs(delta_yaw) < 15°`. Also filter if the angle is too close to relation boundaries: margin to `{15°, 90°, 150°, 180°}` must be at least `8°`. | Exact match on correct option key or option text. |
| `camera_rotation_transform_mc` | `viewpoint_conditioned_spatial_updating` | `If the observer turns {angle_deg} degrees to the {turn_direction}, where would {target_ref} appear in the new view?` or `After turning {angle_deg} degrees to the {turn_direction}, where does {target_ref} appear in the updated view?` | Sample one explicit observer turn from `{right/left} × {90°, 135°, 180°}`. Recompute target direction in the rotated observer frame and classify into `right`, `back-right`, `behind`, `back-left`, `left`. | No explicit hard exclusion beyond relation validity. Difficulty is derived from distance to the reoriented sector boundaries. | Exact match on correct option key or option text. |
| `object_conditioned_reorientation_mc` | `viewpoint_conditioned_spatial_updating` | `Once {facing_ref} is centered as the new front direction, where does {target_ref} lie?` or `If you turn to face {facing_ref}, where would {target_ref} appear in the reoriented view?` | Treat the facing object's yaw as the new forward direction. Compute `delta_yaw = wrap(yaw_target - yaw_facing)` and classify into `right`, `back-right`, `behind`, `back-left`, `left`. | Filter if `abs(delta_yaw) < 15°`. Also require at least `8°` margin from relation boundaries `{15°, 90°, 150°, 180°}`. | Exact match on correct option key or option text. |
| `observer_distance_choice` | `observer_centered_3d_layout_understanding` | `Which of these objects is closest to the current observer in the full panorama?` or `From the current camera position, which listed object is nearest?` | Select 4 depth-valid candidate entities and choose the one with minimum observer-centered depth (`entity_center_depth`). Options are referring expressions, not coarse labels. | Need at least 4 depth-valid entities. Builder takes the nearest 6 by depth, then uses the first 4. Adjacent depths among the selected 4 must differ by at least `0.35m`; otherwise drop the item. | Exact match on correct option key or option text. |
| `relative_3d_position_mc` | `observer_centered_3d_layout_understanding` | `In the current camera-centered 3D frame, which relation best describes {entity_a_ref} relative to {entity_b_ref}?` or `From the current camera viewpoint, which camera-centered 3D relation best matches {entity_a_ref} relative to {entity_b_ref}?` | Use `erp_consistent_xyz_camera` for both entities. Compute `dx = x_A - x_B`, `dy = y_A - y_B`, `dz = z_A - z_B`. Add `right of/left of` if the x-axis separation clears a size-aware threshold; add `above/below` if the y-axis separation clears a size-aware threshold; add `in front of/behind` if `abs(dz) >= 0.6`. Only 1-axis or 2-axis relations are kept. Distractors are generated by flipping active axes and then filling from a fallback relation pool. | Both entities must be compact: `x_fov <= 45°`, `y_fov <= 45°`, and `x_fov * y_fov <= 1800`. Drop if no axis is sufficiently clear. Drop if more than two relation axes become active. | Exact match on correct option key or option text. This task is always marked for manual review due to geometry fragility. |
| `seam_continuity_mc` | `erp_representation_understanding` | One of five seam-aware question families: nearest-neighbor (`Which listed object is actually nearest to {target_ref} near the {target_side} image edge?`), relative-direction (`What is the relation of {neighbor_ref} relative to {target_ref} across the left-right image boundary?`), dedup-count (`The left-edge and right-edge visible parts of {target_ref} should be counted as:`), structure-continuity (`For the {target_ref} touching both image sides, which explanation is more reasonable?`), and same-entity judgement (`The left-edge and right-edge appearances of {target_ref} are best described as:`). | The correct answer depends on `seam_subtype`. `nearest_neighbor` uses the opposite-boundary entity with the smallest wrap-around yaw gap; `relative_direction` uses the same close wrap-around pair but expects the relation `adjacent across the boundary`; `dedup_count`, `structure_continuity`, and `same_entity_judgement` use seam-crossing entities and label them respectively as `one continuous object`, `one continuous structure`, and `same object at different image positions`. Metadata stores subtype-specific traces such as wrap gap, flat-gap lure fields, seam side, and seam-crossing flags. | `nearest_neighbor` requires a boundary-touching target, an opposite-boundary correct candidate with wrap gap `<= 15°`, flat x-gap `>= 0.65 * erp_width`, one flat-image lure with `flat_x_gap <= 0.25 * erp_width`, and two additional larger-gap distractors. `relative_direction` reuses the same close wrap-around pair. `dedup_count`, `structure_continuity`, and `same_entity_judgement` require `seam_crossing_flag = true`, and `structure_continuity` additionally requires a structure-like semantic label. During final public selection, seam items are pooled under one task and sampled in stable random order. | Exact match on correct option key or option text. Always included in manual review. |
| `polar_shape_recovery_mc` | `erp_representation_understanding` | `What is the true shape of {target_ref} in this ERP panorama?` or `Which shape best matches the real object geometry of {target_ref} in this high-latitude ERP region?` | Use the entity's semantic `shape` attribute as the correct answer. Distractors are drawn from a fallback shape vocabulary excluding the ground-truth shape. | Entity must have a valid shape label. It must be a high-latitude or pole-proximal case: `abs(lat) >= 60°` or `infer_pole_proximity(entity)`. Need at least 3 distractor shapes. | Exact match on correct option key or option text. Always included in manual review. |

## Relative 3D Task: Detailed GT Rules

Because `relative_3d_position_mc` is the most geometry-sensitive task, its rule is restated more explicitly here.

| Component | Current Rule |
|---|---|
| Coordinate source | `entity.erp_consistent_xyz_camera` |
| Reference frame | Current camera-centered 3D frame |
| X relation | `right of` if `dx > 0` and `abs(dx) >= max(0.35, radius_x(A) + radius_x(B))`; otherwise `left of` if `dx < 0` and the same threshold is satisfied |
| Y relation | `above` if `dy > 0` and `abs(dy) >= max(0.25, radius_y(A) + radius_y(B))`; otherwise `below` if `dy < 0` and the same threshold is satisfied |
| Z relation | `in front of` if `dz > 0` and `abs(dz) >= 0.6`; `behind` if `dz < 0` and `abs(dz) >= 0.6` |
| Radius approximation | `depth * tan(fov / 2)` along the relevant BFOV axis |
| Allowed label cardinality | Only 1-axis and 2-axis relations are retained |
| Rejection cases | Missing geometry, large objects, zero active axes, or more than two active axes |

## Seam Task Design Notes

The current v1 seam design intentionally avoids trivial yes/no wording. The two scored seam tasks cover different seam-specific failure modes:

| Task | What it diagnoses |
|---|---|
| `seam_continuity_mc` | Whether the model treats the left-right seam as a true wrap-around topology: nearest neighbors can lie across the boundary, cross-boundary pairs can still be adjacent, seam-crossing fragments should not be double-counted, and boundary-touching structure/entity fragments may remain one continuous instance |

## Supplementary Rotation Robustness Protocol

Rotation consistency is currently released as a **supplementary protocol**, not as a scored v1 QA task.

| Protocol | Current Status | What it measures | Where it is defined |
|---|---|---|---|
| `Yaw-shift / Seam-relocation Robustness Protocol` | Supplementary only | Whether benchmark performance is grounded in spherical geometry rather than fixed ERP column statistics or seam placement priors | `/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/ROTATION_PROTOCOL.md` and `/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/rotation_protocol.py` |

### Protocol Coverage

| Task Family | Expected Behavior Under Yaw Shift |
|---|---|
| `referring_grounding_bfov` | Equivariant: predicted BFOV yaw should shift with the known offset while pitch and extent stay stable |
| `absolute_direction_mc` | Equivariant: the absolute sector should update according to the known yaw shift |
| `relative_direction_mc` | Invariant: the pairwise panoramic relation should remain unchanged |
| `camera_rotation_transform_mc` | Invariant under the current protocol design |
| `object_conditioned_reorientation_mc` | Invariant under the current protocol design |

### Protocol Reporting

| Metric | Meaning |
|---|---|
| `base_overall.accuracy` | Accuracy on the original benchmark items participating in the protocol |
| `shifted_overall.accuracy` | Accuracy on the yaw-shifted protocol items |
| `headline_gap` | `shifted_accuracy - base_accuracy` |
| `pair_consistency.overall_consistency` | Whether paired predictions obey the expected invariant/equivariant transformation law |
| `grounding_transformed_mean_iou` | Mean IoU between transformed base grounding prediction and shifted grounding prediction |

## Notes for Paper Writing

If you want a compact appendix presentation, the most concise paper version is:

1. one short table for global scoring policy
2. one main task table for the 10 scored tasks
3. one short supplementary table for the yaw-shift protocol

This file is already structured in exactly that order.
