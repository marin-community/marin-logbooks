<!-- TASK_ID: da-19-1 -->

# Task: BiomniBench-DA Task DA-19-1

## Question

To identify therapeutic targets downstream of CBFβ-SMMHC inhibition in inv(16) AML, which genes are most significantly downregulated upon AI-10-49 treatment?

## Data Files

Multi-omics cohort from the "CBFβ-SMMHC Inhibition Triggers Apoptosis by Disrupting MYC Chromatin Dynamics in inv(16) AML" paper (Cancer Cell; GSE101788/101789/101790). Human inv(16) leukemia cells (ME-1) treated with the CBFβ-SMMHC inhibitor AI-10-49 or DMSO control, profiled by RNA-seq, ChIP-seq (H3K27ac and RUNX1), and ATAC-seq.

**Sample-to-condition mapping** (used across all modalities):
- **RNA-seq**: `GSM2715529`-`GSM2715531` = DMSO rep1-3; `GSM2715532`-`GSM2715534` = AI-10-49 rep1-3.
- **ChIP-seq H3K27ac**: `GSM2715535` = DMSO, `GSM2715536` = AI-10-49. **Input DNA**: `GSM2715539` = DMSO, `GSM2715540` = AI-10-49.
- **ChIP-seq RUNX1**: `GSM2715537` = DMSO, `GSM2715538` = AI-10-49.
- **ATAC-seq**: `GSM2715541`-`GSM2715542` = DMSO rep1-2; `GSM2715543`-`GSM2715544` = AI-10-49 rep1-2.

- `processed_data/rnaseq/cuffdiff/gene_exp.diff`: Cuffdiff gene-level differential-expression output (DMSO vs AI-10-49). Columns: `test_id`, `gene_id`, `gene`, `locus`, `sample_1` (DMSO), `sample_2` (AI-10-49), `status`, `value_1`/`value_2` (FPKM), `log2(fold_change)`, `test_stat`, `p_value`, `q_value`, `significant` (yes/no).
- `processed_data/rnaseq/cuffdiff/run.info`: Cuffdiff run parameters (command line, version).
- `processed_data/rnaseq/rnaseq_alignment_manifest.tsv`: RNA-seq sample-to-BAM alignment manifest. Columns: `sample_id`, `run_accession`, `condition` (DMSO / AI-10-49), `fastq_r1/r2`, `accepted_hits_bam`.

## Required Outputs

You MUST create the following two output files:

### 1. Analysis Trace (`/app/trace.md`)

Document your complete analysis process in markdown format, following this structure. This trace is evaluated by an expert judge: thoroughness and clarity directly affect your score.

**Required sections (use these as headings in your trace):**

1. **Objective**: Restate the question; define what "success" looks like.
2. **Data Sources**: For each file: name, dimensions, key columns, example values for any column you'll filter or group by, data quality notes.
3. **Approach**: Numbered analytical steps. For each step, document:
   - **Description**: what was done.
   - **Decision and rationale**: which method, threshold, normalization, or filter was chosen, and why; what alternatives were considered or rejected; any assumption or judgment call.
   - **Code**: paste the actual code (Python / R / shell) used for each non-trivial operation in this step, filters, aggregations, joins, statistical tests, transformations, plotting calls. Use real, copy-pasteable code (not pseudocode and not a paraphrase). The trace must be reproducible from these snippets, so do not skip the code for steps that do real work.
   - **Quantitative intermediate result** (counts, p-values, dimensions after filtering).
4. **Results**: Specific quantitative findings (numbers, gene/protein names, statistics, summary tables); biological / clinical interpretation of what they mean; what the analysis cannot conclude (limitations).
5. **References**: Real, identifiable citations (author + year, DOI, PubMed ID, named database) for biological mechanisms invoked, methodology choices, and any external knowledge used. No vague "the literature shows" without a checkable reference.

**Example excerpt** (showing expected level of detail):

````md
## Data Sources
- `samples.csv`: 1,850 samples x 10 columns. Key columns: SampleID, Species, Site, CollectionDate, pH, Abundance. 5 missing pH values imputed with site median.

## Approach

### Step 1: Load and inspect data
**Description**: Read the sample table and check for missing values in the columns we will filter or aggregate on.

**Decision and rationale**: Imputed 5 missing pH values with the per-site median rather than dropping rows or using a global median, because pH varies systematically across sites and dropping would lose 5/1,850 ≈ 0.3% of samples in a non-random way. Considered mean imputation but rejected it (skewed by extreme acidic outliers).

```python
df = pd.read_csv('samples.csv')
print(f"Shape: {df.shape}")  # (1850, 10)
print(f"Missing pH: {df['pH'].isna().sum()}")  # 5
df['pH'] = df.groupby('Site')['pH'].transform(lambda s: s.fillna(s.median()))
```
**Quantitative result**: 1,850 samples across 12 sites; 5 missing pH values imputed; 0 missing values remain.
````

**Key principles:**
- Show intermediate result counts at every filtering step (e.g., "1,850 -> 1,720 -> 1,680")
- Name specific entities and quantities, avoid vague summaries
- Justify analytical choices (test selection, threshold, normalization) explicitly
- Ground biological / clinical claims in real references, do not assert from memory
- State what the analysis cannot conclude (limitations)

### 2. Final Answer (`/app/answer.txt`)

Write your final answer to the question above in plain text format.

## Environment

- Python 3 and R are pre-installed
- Install any additional packages you need (pandas, statsmodels, etc.)
- Internet access is available for package installation
- **Do not search for or read the specific source paper, figures, or supplementary materials** that the dataset comes from. Solve the task directly from the provided data and domain knowledge.

## Data Sources

- `processed_data/rnaseq/cuffdiff/gene_exp.diff`: Located at `/app/data/processed_data/rnaseq/cuffdiff/gene_exp.diff`
- `processed_data/rnaseq/cuffdiff/run.info`: Located at `/app/data/processed_data/rnaseq/cuffdiff/run.info`
- `processed_data/rnaseq/rnaseq_alignment_manifest.tsv`: Located at `/app/data/processed_data/rnaseq/rnaseq_alignment_manifest.tsv`
