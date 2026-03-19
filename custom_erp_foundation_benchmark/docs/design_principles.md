# Benchmark Design Principles

## 1. Benchmark-only mindset

A benchmark is not a training set.

The benchmark should optimize for:

- interpretability
- answer uniqueness
- coverage across ability slices
- ERP specificity
- hidden-test reliability

It should not optimize for:

- maximum sample count
- language diversity at the cost of ambiguity
- easy automatic generation with weak ground truth

## 2. Principles adapted for ERP-native VLM evaluation

### Principle A: capability-first coverage

Inspired by HELM, the benchmark should report scores over explicit capability
slices rather than one blended number.

For ERP evaluation, that means at least:

- language alignment
- ERP-native spatial understanding
- 3D/search preconditions
- ERP robustness slices such as seam, pole, and rotation

Source:

- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)

### Principle B: include hard and stress slices, not just average cases

Inspired by Dynabench, the benchmark should include difficult, failure-prone,
and shortcut-resistant cases rather than only naturally easy samples.

For ERP, the important stress slices are:

- seam-adjacent objects
- high-latitude pole-region objects
- same-class distractors
- low-margin depth comparisons
- rotation-transform cases

Source:

- [Dynabench: Rethinking Benchmarking in NLP](https://arxiv.org/abs/2104.14337)

### Principle C: make official answers closed-form whenever possible

Benchmark answers should be easy to score consistently.

For this benchmark, core answers should mainly be:

- multiple choice
- yes/no
- count from a small integer set
- discrete direction labels
- relation labels
- structured bins such as `near / medium / far`

Long-form explanations may still exist, but they should not dominate the
official score.

### Principle D: enforce ERP necessity

A benchmark for an ERP-native model should not be solvable in the same way by a
generic narrow-FoV crop benchmark.

Therefore, the ERP-specific branch should prefer items whose truth depends on:

- panorama-wide direction
- seam continuity
- wrap-around topology
- pole distortion handling
- rotation consistency
- multi-object arrangement over a wide field

### Principle E: benchmark answers must be more reliable than training labels

Because this is a test set, the verification standard should be much stricter
than the training-data pipeline.

Recommended rule:

- all official items must be manually verified
- ambiguous items must be rejected, not softened
- if two independent annotators disagree, the item should be escalated or
  discarded

### Principle F: hold out scenes, not only prompts

For a visual benchmark, scene leakage is more dangerous than prompt leakage.

Therefore:

- official benchmark scenes should not appear in SFT or pretraining curation
- near-duplicate ERP panoramas should be deduplicated at the scene level
- public dev and hidden test should not share near-duplicate scenes

## 3. Implications for your ERP benchmark

These principles lead to a different design than a generic VQA benchmark:

- keep task families small and interpretable
- use fixed answer spaces
- report both aggregate and slice scores
- preserve a benchmark-only hidden split
- separate `core_v1` from harder geometry/search extensions

## 4. Why some public datasets are only secondary references

### `OSR-Bench`

Useful because it is close to omnidirectional spatial reasoning and language
question answering.

### `PanoEnv`

Useful because it adds ERP-side 3D reasoning and seam/view-composition effects.

### `H*Bench`

Useful as a downstream search-oriented reference. It is especially relevant if
you adapt it from perspective observations to direct ERP input.

Source:

- [Thinking in 360°: Humanoid Visual Search in the Wild](https://arxiv.org/abs/2511.20351)

### `OmniSpatial`

Useful as an external spatial-language control benchmark, but not ERP-native.
Therefore it should be treated as a secondary comparison set rather than the
main benchmark target.

Source:

- [OmniSpatial project page](https://qizekun.github.io/omnispatial/)

## 5. Practical acceptance rules

An item should enter the official hidden test only if all of the following hold:

- the required metadata fields are present
- the ground truth is rule-verifiable or manually confirmed
- the question is visually grounded
- the answer is unique
- the item passes ambiguity checks
- the item does not duplicate another scene/target pair
- the item fits the target task quota and slice balance

## 6. Benchmark reporting format

The final benchmark should report:

- overall core score
- per-task score
- per-ability score
- seam slice score
- pole slice score
- rotation slice score
- distractor slice score
- optional search extension score

This is much more useful than a single blended benchmark number because it
shows whether the model truly understands ERP-specific structure.
