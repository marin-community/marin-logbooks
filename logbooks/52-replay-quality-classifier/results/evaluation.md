# Replay low-value classifier evaluation

The structural classifier improves low-value recall over junkgate2 and rules at
both frozen calibration operating points. The 5% calibration point catches 101
of 204 low-value test documents while dropping 18 of 687 natural documents.
Labels are model-assisted judgments, not human ground truth.

| Model | Calibration false-drop target | Test low-value recall | Test natural false-drop | Test precision |
| --- | --- | --- | --- | --- |
| Structural GBDT | 2% | 71/204 (34.8%) | 8/687 (1.16%) | 89.9% |
| Structural GBDT | 5% | 101/204 (49.5%) | 18/687 (2.62%) | 84.9% |
| Native rules | 2% | 28/204 (13.7%) | 1/687 (0.15%) | 96.6% |
| Native rules | 5% | 37/204 (18.1%) | 20/687 (2.91%) | 64.9% |
| Rules | 2% | 31/204 (15.2%) | 2/687 (0.29%) | 93.9% |
| Rules | 5% | 36/204 (17.6%) | 19/687 (2.77%) | 65.5% |
| Junkgate2 | 2% | 7/204 (3.4%) | 12/687 (1.75%) | 36.8% |
| Junkgate2 | 5% | 22/204 (10.8%) | 39/687 (5.68%) | 36.1% |

The recommended replay cutoff is `low_value_score >= 0.4220386490525259`, the
predeclared 5% calibration operating point. Observed test natural false-drop is
2.62% (Wilson 95% interval 1.66–4.10%) and recall is 49.51% (42.72–56.32%). The
conservative 2% calibration point uses `>= 0.5949392992426772`: observed natural
false-drop 1.16% (0.59–2.28%) and recall 34.80% (28.60–41.57%). These are sample
estimates under frozen thresholds, not population guarantees.

At the 5% point, the paired recall gain is 38.73 percentage points over junkgate2
(bootstrap 95% interval 31.37–46.57) and 31.37 points over native rules (24.02–38.73). The continuous total-share
rule variant gives a 31.86-point gain (24.51–39.22).
Thresholds remain fixed throughout the 10,000 paired document resamples.

## Protocol

The sample contains 3,600 distinct rows from 388 cluster/quality/length caches,
balanced with small strata capped at population. The original 400-document seed
and matching visible content are development-only. Fresh exact visible-content
groups are split by SHA256 before labeling. Planned splits are 2,038 training,
655 calibration and 907 test documents. There are 2,009/643/891 conclusive labels,
respectively; 42 documents have no usable final annotation and 15 abstain.

Long documents expose merged head/middle/tail windows, at most 2,048 Marin tokens
each. Raw document text and requests/responses remain private. The public label
manifest identifies sample IDs, split, SHA256 of the JSON list of decoded visible
windows, category and binary judgment. This fingerprint is not a full-document
hash. No rationales or source text are published.

GLM-5.3-Flash via Together labeled the sample at temperature .7, concurrency 8,
with no SDK retries and exclusive request-admission receipts. Uncertain requests
were not replayed. An independent GPT-6 Astra panel labeled 300 fresh documents,
150 long and 150 short. Of 296 usable pairs, 291 are conclusive in both: binary
agreement 93.81%, Cohen's kappa .7983; category agreement over all pairs is 89.19%.
Fresh-context Astra adjudicated 60 anonymous disagreement or tokenizer-wrapper
audit cases. The checker and adjudicator share a model family. GLM continuation
explicitly excluded universal BOS/EOS markers as evidence; Astra's initial panel
used the original rubric. The audit corrected wrapper-based rationales.

The rubric protects useful code, math, multilingual prose and legible OCR.
Low-value labels cover dumps, machine output, dominant raw markup/tables, damaged
OCR, benchmark/trajectory bundles, repetitive templates and explicit generated
QA/instruction artifacts. Polished style alone never establishes model authorship.
A short QA tail does not invalidate coherent primary content by itself.

## Selection and baselines

The final model is 160 gradient-boosted trees, depth 3, minimum leaf 10, learning
rate .05, on 146 deterministic structural features. Model choice used a grouped
inner validation slice of the training set; cutoffs used only calibration data.
The model, cutoffs and selection record were published at research commit
`90d61498a6` before test evaluation. Packaging later corrected the tokenizer
metadata to the exact frozen replay tokenizer: it renames two added tool-call IDs,
neither of which appears in any sampled view. Learned parameters, views, labels
and cutoffs are unchanged; final model and calibration JSON digests differ only
because the bundle now pins the tokenizer files. Test labels were never used for training,
model selection or cutoff fitting.

The predeclared grid compared structural features, bounded Harrier-0.6B vectors
and concatenation, using small GBDT and logistic heads. Structural and fusion tied
mean inner-validation recall at the two false-drop targets (.4945), with AP .7615
and .7693. Embedding-only reached .3626. The cost rule selected structural, avoiding
mandatory embedding inference. The existing Harrier store could not be safely
joined to consolidated replay cache rows, so the 3,600 sampled vectors were
recomputed; no full-corpus embedding recomputation occurred.

Junkgate2 uses its deployed Marin model and calibration over real complete-document
begin/middle/end 512-token windows. Both baseline cutoffs were fitted on exactly
the same 643 calibration documents. The rules baseline is the continuous junk-chunk
share from `dml/lce-junk-filter` at `70acdcfd6f08996e855e5ab7c1e805dcaa1f5558`.
The native-rule ranking also preserves its separate machine priority via
`max(total_share/.6, machine_share/.3)`; score 1 reproduces the published verdict.
This additional baseline was added after the first test opening, without changing
the learned model or its cutoffs. Both rule rankings are reported.

## Limits and use

Rates are unweighted on a balanced, capacity-capped cache sample; they do not
establish corpus prevalence, retained tokens or downstream training improvement.
Bounded windows can miss interior content. Exact visible-content grouping does
not eliminate near duplicates. Confidence intervals condition on the fitted model,
calibration and model-assisted judgments; they do not include annotation error or
all model-selection uncertainty. Recalibrate and validate on another corpus.

The Exp3 data owner receives the frozen model, cutoffs and sampled scores. Full
replay scoring and adoption must be recorded separately and must not delay its
data handoff. No full-corpus filter or training run has been applied here.
