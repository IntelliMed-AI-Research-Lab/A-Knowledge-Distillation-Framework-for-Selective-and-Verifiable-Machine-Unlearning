<div align="center">

# A Reference-Based Audit of Retain-Only Distillation for Class-Level Machine Unlearning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Preprint-orange?style=flat-square)](.)

**Umme Sara · MST Moriom Akter Bithee · Md Irfanul Kabir Hira · Mohammad Shorif Uddin**

</div>

---

This repository contains the code and results for an **audit** of retain-only knowledge distillation used as a class-level unlearning mechanism. It is **not** a new unlearning algorithm, and it does **not** show that distillation forgets. The question it asks is what an auditor can conclude about an updated model from black-box access to the original model, the updated model, and a model retrained on the retained classes.

> **The results are empirical evidence, not a certificate of deletion.** All numbers come from one seed, one architecture (ResNet-18) and one draw of deleted classes.

## Contents

- [Why an audit](#why-an-audit)
- [Method](#method)
- [Models compared](#models-compared)
- [Datasets](#datasets)
- [Training recipes](#training-recipes)
- [Audit metrics](#audit-metrics)
- [Results](#results)
- [What the results support, and what they do not](#what-the-results-support-and-what-they-do-not)
- [Repository structure](#repository-structure)
- [Installation and usage](#installation-and-usage)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations and future work](#limitations-and-future-work)
- [Citation](#citation)

---

## Why an audit

Class-level unlearning is often judged by how far the updated model moves away from the original model. Divergence from the original is necessary for forgetting but not sufficient. The goal is closeness to a model that never saw the deleted data, so this audit anchors every measurement to a **retrained reference** wherever one is available, and reports common indicators (logit distance, feature similarity, prediction entropy, transferred membership attacks) only as descriptions when no reference value exists.

## Method

**Retain-only distillation.** A teacher is trained on all classes. After a deletion request, a student is trained from random initialization on the retain set `D_r` only, using teacher logits and cross-entropy on the retained labels. No deleted image or label enters the student's loss, and no explicit forgetting term is applied.

```
L_KD    = T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )     # logits over all C classes
L_total = lambda * L_KD + (1 - lambda) * L_CE                  # CE on retained labels only
```

Fixed settings: `T = 4`, `lambda = 0.7`. Neither was searched.

**A leakage channel.** The teacher's logits on retained images still carry probability mass on the deleted classes, so soft labels can pass deleted-class information to a student that never sees a deleted image. A masked variant (KD-M), which removes the deleted-class logits from the distillation target, is specified in the paper but **was not trained**. Only the unmasked variant (KD-U) is evaluated.

## Models compared

| Model | Role | Initialization | Trained on | Objective |
|---|---|---|---|---|
| Teacher `f_T` | Original model | ImageNet | All classes | CE |
| Student `g_φ` | Updated model | Random | Retained classes | CE + KD |
| Retrained, accuracy reference `f_R^acc` | Retained-class accuracy comparison | Random | Retained classes | CE |
| Retrained, audit reference `f_R` | Deleted-class diagnostics (JS, entropy, probes, retained-class attacks) | ImageNet | Retained classes | CE |

The two references differ in initialization **and** recipe, and neither uses the student's recipe. Comparisons of the student with `f_R` therefore mix the effect of initialization with any effect of distillation. See [Limitations](#limitations-and-future-work).

## Datasets

Datasets are **not** included. Download them from the links below.

| Dataset | Classes used | Train images per class | Link |
|---|---|---|---|
| Caltech-101 | 87 (balanced subset; includes the background class and related classes such as *Faces* and *Faces_easy*) | 32 | [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) |
| Caltech-256 | 257 (including clutter) | 60 | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/caltech256) |
| CIFAR-100 | 100 | 500 | [Kaggle](https://www.kaggle.com/datasets/melikechan/cifar100) |
| Stanford Dogs | 120 | 70 | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) |

Because the Caltech subsets are class-balanced subsets, results are not directly comparable with studies on the standard splits.

**Deletion.** 5, 10 and 15 classes are deleted per dataset (12 configurations). Classes are drawn at random with seed 42, and the three scopes are nested prefixes of one draw. The test set is unchanged, so retained and deleted classes are evaluated on the same test set.

**Preprocessing.** Images are resized to 224×224.
- Original notebooks: normalization with mean 0.5 and std 0.5; training augmentation of horizontal flip (p = 0.5), rotation up to 10°, brightness/contrast jitter 0.2, and translation up to 10%.
- Audit notebooks: ImageNet mean/std and horizontal flip only.

## Training recipes

All runs use ResNet-18, a fixed number of epochs, no learning-rate schedule, no weight decay (except the accuracy reference), no early stopping, and the final-epoch weights. Test accuracy is logged each epoch but is not used for model selection.

| Parameter | Teacher | Student | Retrained (accuracy) | Retrained (audit) |
|---|---|---|---|---|
| Optimizer | Adam | Adam | AdamW | Adam |
| Learning rate | 1e-4 | 3e-4 | 1e-3 | 1e-4 |
| Epochs | 30 | 30 | 30 | 30 |
| Batch size | 16 | 16 | 32 | 16 |
| Weight decay | 0 | 0 | 1e-4 | 0 |
| LR scheduler | None | None | None | None |
| Initialization | ImageNet | Random | Random | ImageNet |
| Objective | CE | CE + KD | CE | CE |
| KD temperature / weight | n/a | 4 / 0.7 | n/a | n/a |
| Base seed | 42 | 42 | 42 | 42 |

**[TODO before release: some earlier notebook versions for Caltech-256 and Stanford Dogs use 35 epochs. State here which count produced the reported numbers.]**

## Audit metrics

| Question | Measurement | Reference |
|---|---|---|
| Utility | Retained-class accuracy `A_r`; deleted-class accuracy `A_f`; overall accuracy is reported but is dominated by how many deleted classes are in the test set | Random-init retrained model (`f_R^acc`) |
| Output behavior on deleted classes | Jensen–Shannon divergence between student and retrained outputs, restricted to retained classes and renormalized (upper bound ln 2 = 0.693 nats) | Audit reference `f_R` |
| Uncertainty | Prediction entropy on deleted-class samples | Audit reference `f_R` |
| Residual information | Linear probe on frozen penultimate features (ridge classifier), trained on deleted-class training samples and tested on deleted-class test samples | Teacher and audit reference `f_R` |
| Membership exposure, deleted classes | Logistic-regression attack on 5 output features (max probability, entropy, top-3 probabilities): teacher-own, transfer (teacher-fitted, applied to the student), student-own | None available |
| Membership exposure, retained classes | Own-output attacks on 7 label-aware features, balanced members and non-members (base rate 0.5) | Audit reference `f_R` |
| Descriptive only | Teacher–student logit distance and penultimate-feature cosine similarity | **No reference value** |

No composite score is computed. AUC intervals are approximate (Hanley–McNeil), assume independent samples, and exclude variation across seeds and class draws.

## Results

All results are from a single run per configuration.

### Accuracy (%)

Teacher accuracy is over all classes. Student and retrained accuracy are on retained-class test samples. Gap is retrained minus student, so positive means the retrained model is more accurate.

| Dataset | Deleted | Teacher (overall) | Student (retained) | Retrained, random init (retained) | Gap (pp) | Student (deleted) |
|---|---|---|---|---|---|---|
| Caltech-101 | 5 | 93.53 | 59.30 | 63.09 | +3.79 | 0.00 |
| | 10 | 93.53 | 63.96 | 58.49 | −5.47 | 0.00 |
| | 15 | 93.53 | 61.28 | 60.32 | −0.96 | 0.00 |
| Caltech-256 | 5 | 73.13 | 44.59 | 46.03 | +1.44 | 0.00 |
| | 10 | 72.07 | 46.61 | 45.21 | −1.40 | 0.00 |
| | 15 | 71.24 | 47.03 | 45.37 | −1.66 | 0.00 |
| Stanford Dogs | 5 | 61.67 | 28.35 | 32.14 | +3.79 | 1.33 |
| | 10 | 60.28 | 31.52 | 34.97 | +3.45 | 0.00 |
| | 15 | 61.06 | 30.19 | 33.59 | +3.40 | 0.00 |
| CIFAR-100 | 5 | 77.55 | 71.85 | 62.99 | −8.86 | 2.60 |
| | 10 | 77.87 | 71.40 | 62.60 | −8.80 | 1.10 |
| | 15 | 77.87 | 72.08 | 64.28 | −7.80 | 3.67 |

- Student accuracy on deleted classes is 0.00–3.67%. This is the expected consequence of training without those labels and is not evidence of forgetting by itself.
- The decline of overall student accuracy with the deletion scope is largely an accounting effect of deleted classes in the test set. On CIFAR-100 retained-class accuracy is flat (71.4–72.1%) while overall accuracy falls from 68.4% to 61.8%.
- Test-set sampling error alone gives an approximate 95% half-width of about 5 points for a difference between two models on Caltech-101, 2 points on Caltech-256 and Stanford Dogs, and 1.5 points on CIFAR-100.
- The student and the accuracy reference use different recipes, so these differences mix recipe and distillation effects.

### Behavior on deleted-class samples

JS is between student and the ImageNet-initialized audit reference. Probe is accuracy (%) of a linear probe for the teacher (T), audit reference (R) and student (S). Chance is 20%, 10% and 6.7% for 5, 10 and 15 classes.

| Dataset | Deleted | JS (nats) | Probe T | Probe R | Probe S |
|---|---|---|---|---|---|
| Caltech-101 | 5 | 0.4276 | 100.00 | 98.94 | 95.74 |
| | 10 | 0.4859 | 98.72 | 92.31 | 87.82 |
| | 15 | 0.4111 | 99.45 | 97.25 | 92.58 |
| Caltech-256 | 5 | 0.5356 | 85.16 | 75.00 | 60.16 |
| | 10 | 0.4831 | 90.75 | 82.38 | 73.13 |
| | 15 | 0.5049 | 89.57 | 83.44 | 69.94 |
| Stanford Dogs | 5 | 0.5025 | 87.66 | 87.01 | 64.94 |
| | 10 | 0.4813 | 92.48 | 90.91 | 73.98 |
| | 15 | 0.4545 | 90.45 | 86.59 | 73.98 |
| CIFAR-100 | 5 | 0.3490 | 98.40 | 97.80 | 96.40 |
| | 10 | 0.3466 | 93.30 | 88.10 | 87.60 |
| | 15 | 0.3565 | 89.80 | 82.73 | 80.07 |

- **JS divergence is 0.35–0.54 nats, 50–77% of its upper bound.** The student's outputs are far from the reference's in absolute terms. This cannot be interpreted further because the teacher–retrained divergence and a noise floor from two independently trained retrained models were not computed.
- **The retrained model is as confident as the student** on deleted-class samples (mean entropy 0.65–1.27 nats against uniform references of 4.3–5.5 nats), so low entropy is expected of a model that never saw the class and is not evidence of forgetting.
- **Deleted-class information is linearly decodable from every model**, including the retrained one. The ordering teacher ≥ retrained ≥ student holds in all 12 configurations, so the student retains no more linearly decodable information than the reference. The audit reference starts from ImageNet weights and the student does not, so this ordering is expected regardless of forgetting, and the probe separates the teacher from the retrained model only weakly.

### Membership inference

**Deleted-class samples (AUC).** Members are deleted-class training samples seen by the teacher; non-members are held-out samples of the same classes. The 10-class Caltech-101 run was not executed.

| Dataset | Deleted | Teacher-own | Transfer to student | Student-own |
|---|---|---|---|---|
| Caltech-101 | 5 | 0.742 | 0.571 | 0.594 |
| | 15 | 0.861 | 0.499 | 0.537 |
| Caltech-256 | 5 | 0.815 | 0.559 | 0.457 |
| | 10 | 0.820 | 0.512 | 0.584 |
| | 15 | 0.810 | 0.489 | 0.461 |
| Stanford Dogs | 5 | 0.845 | 0.534 | 0.527 |
| | 10 | 0.803 | 0.513 | 0.500 |
| | 15 | 0.869 | 0.494 | 0.443 |
| CIFAR-100 | 5 | 0.680 | 0.490 | 0.502 |
| | 10 | 0.698 | 0.499 | 0.476 |
| | 15 | 0.693 | 0.495 | 0.468 |

- The attack finds membership signal in the teacher (AUC 0.68–0.87).
- **Transfer to the student fails (AUC 0.49–0.57), but this reflects a calibration difference, not privacy.** Transfer accuracy (0.23–0.51) is below the base rate in every case because the student is less confident than the teacher.
- Attacks fitted to the student itself give AUC 0.44–0.59, and no interval lies entirely above 0.5. The intervals are wide, and chance-level AUC is expected for samples that neither the student nor a retrained model trained on, so this result cannot separate a small residual signal from none. The retrained model's deleted-class attack AUC was not measured.

**Retained-class samples.** The student exposes more membership signal than the ImageNet-initialized retrained model in **11 of 12** configurations: by 0.08–0.15 AUC on Caltech-101 and Caltech-256, and by 0.04 or less on Stanford Dogs and CIFAR-100. No intervals were computed. This may reflect distillation or the difference in initialization, and the two were not separated. It measures exposure of retained data, not forgetting of deleted data.

## What the results support, and what they do not

**Supported by this audit:**
1. Student accuracy on deleted classes is 0–3.7%.
2. Overall accuracy, logit distance, feature cosine similarity, prediction entropy and transferred-attack AUC are insufficient as evidence of forgetting when no reference is available.
3. The student's retained-class accuracy is comparable to a randomly initialized retrained model on the two Caltech datasets, 3.4–3.8 points lower on Stanford Dogs and 7.8–8.9 points higher on CIFAR-100. The recipes differ.
4. The student retains no more linearly decodable deleted-class information than the audit reference (low-power probe, confounded by initialization).
5. The student exposes more membership signal on retained samples than the audit reference in 11 of 12 configurations.

**Not shown:**
- That the student's outputs on deleted-class samples are close to those of a model that never saw them.
- Whether the teacher's soft labels transferred deleted-class information. This requires the masked-distillation student, which was not trained.
- How retain-only distillation compares with fine-tuning, gradient ascent, SCRUB, SalUn or DELETE. None were run.
- Anything about seed-to-seed variability.

## Repository structure

```
.
├── notebooks/
│   ├── caltech_balanced_dataset1.ipynb         # Caltech-101 (87-class subset): teacher, student, deleted-class
│   │                                           #   evaluation, teacher-own / transfer / student-own attacks
│   ├── Caltech_256_balanced_dataset.ipynb      # same pipeline, Caltech-256
│   ├── Standford_dog_balanced_dataset.ipynb    # same pipeline, Stanford Dogs
│   ├── cifar100.ipynb                          # same pipeline, CIFAR-100
│   ├── part_1_for_Caltech_101_...resum.ipynb   # audit: ImageNet-init retrained reference, JS, entropy,
│   │                                           #   linear probes, retained-class attacks
│   └── part_2_stanford-dogs-forget-10-forget-15.ipynb   # audit for Stanford Dogs, 10 and 15 deleted classes
├── requirements.txt
└── README.md
```

**[TODO before release:]**
- Keep **one** version of each notebook. The repository currently holds several copies that differ in epoch counts.
- Add the script that trains the randomly initialized accuracy reference (AdamW, lr 1e-3, batch size 32). It is required to reproduce the retained-class accuracy comparison above.
- Replace hard-coded local paths (`E:\...`) in the original notebooks with a configurable path.
- Update the tree above to match the final file names.

## Installation and usage

```bash
git clone https://github.com/irfanulkabirhira/A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning.git
cd A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning
pip install -r requirements.txt
```

`requirements.txt` (pin the exact versions you used):

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-learn
matplotlib
tqdm
opencv-python
timm
```

The experiments run as notebooks on a single GPU (Kaggle was used for the audit notebooks).

1. **Per dataset, run the original notebook.** It trains the ResNet-18 teacher on all classes, draws the deleted classes with seed 42, trains the retain-only distillation student, and computes accuracy, teacher–student distances, entropy, and the deleted-class membership attacks. Set the dataset path at the top of the notebook.
2. **Run the audit notebooks (`part_1`, `part_2`).** They train or load the ImageNet-initialized reference and compute JS divergence, entropy, linear probes and retained-class attacks for the 12 configurations. If you supply your own checkpoints, the notebook requires provenance metadata and rejects unverified weights unless you explicitly opt in. Without checkpoints, the models are retrained from scratch.
3. **Train the random-init accuracy reference** with the script listed under *Repository structure*.

## Reproducibility notes

- All runs use base seed 42. The audit notebooks add fixed per-run offsets to the seed.
- Deleted classes are the first 5, first 10 and all 15 of one seed-42 draw per dataset, so scopes are nested.
- The audit notebooks reconstruct models unless checkpoints are supplied, so their numbers pertain to the models they train.
- The original and audit notebooks use different preprocessing (see [Datasets](#datasets)).
- Teacher accuracy is identical across deletion scopes on Caltech-101 and (for 10 and 15 classes) CIFAR-100, but varies on Caltech-256 and Stanford Dogs. **[TODO: state whether the teacher is trained once per dataset or once per run.]**
- Results come from a single run per configuration. Differences of 1–3 points between scopes are not interpreted.

## Limitations and future work

**Limitations**
- **Missing reference quantities.** The audit criterion compares the student's distance to the retrained model with the teacher's distance to it. Only the first was computed. The teacher–retrained divergence and a noise floor from two retrained models are missing, so the JS values cannot be interpreted.
- **Initialization and recipe confounds.** The teacher and the audit reference start from ImageNet weights, the student and the accuracy reference start from random weights, and the two references use different recipes. No experiment separates initialization or recipe from distillation.
- **Untested leakage channel.** The masked-distillation variant was specified but not trained.
- **Single run.** One seed, one architecture, one nested class draw.
- **Metric limits.** The linear probe has low power and no control on retained classes. Membership attacks are logistic regressions on five (deleted-class, label-free) or seven (retained-class, label-aware) output features, with approximate intervals. Threshold, shadow-model and low-false-positive-rate attacks were not run.
- **No baselines.** No other unlearning method was evaluated.
- **No efficiency claim.** The student trains from random initialization on the whole retain set and needs teacher forward passes, so its cost is at least that of retraining. Wall-clock time was not measured.
- **Scope.** Class-level deletion, ResNet-18 and image classification only.

**Future work, in order of how much it would strengthen the conclusions**
1. Compute the teacher–retrained JS divergence and a retrained-seed noise floor (inference only).
2. Add probe and cosine-similarity controls on retained classes, and threshold and low-FPR attacks.
3. Train the masked-distillation variant (KD-M) and compare it with KD-U.
4. Repeat each configuration with several seeds and independent class draws, with paired tests across seeds.
5. Run fine-tuning, gradient ascent, SCRUB, SalUn and DELETE under the same protocol.
6. Match initialization and recipe between the student and the references.
7. Test a second architecture, instance-level deletion, and non-image tasks.

## Citation

```bibtex
@article{sara2026reference,
  title   = {A Reference-Based Audit of Retain-Only Distillation for Class-Level Machine Unlearning},
  author  = {Sara, Umme and Bithee, MST Moriom Akter and Hira, Md Irfanul Kabir and Uddin, Mohammad Shorif},
  journal = {Preprint},
  year    = {2026},
  note    = {Code: https://github.com/irfanulkabirhira/A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning}
}
```

## License

MIT. See [LICENSE](LICENSE). The datasets are public benchmarks; refer to their own licenses before reuse.

## Acknowledgements

This work received no external funding. AI language-editing tools (ChatGPT, Gemini) were used for language correction of the manuscript.
