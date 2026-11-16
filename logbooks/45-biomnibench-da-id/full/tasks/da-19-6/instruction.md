<!-- TASK_ID: da-19-6 -->

# Task: BiomniBench-DA Task DA-19-6

## Question

To understand how CBFβ-SMMHC inhibition affects chromatin landscape in inv(16) AML, does AI-10-49 treatment lead to global changes in chromatin accessibility?

## Data Files

Multi-omics cohort from the "CBFβ-SMMHC Inhibition Triggers Apoptosis by Disrupting MYC Chromatin Dynamics in inv(16) AML" paper (Cancer Cell; GSE101788/101789/101790). Human inv(16) leukemia cells (ME-1) treated with the CBFβ-SMMHC inhibitor AI-10-49 or DMSO control, profiled by RNA-seq, ChIP-seq (H3K27ac and RUNX1), and ATAC-seq.

**Sample-to-condition mapping** (used across all modalities):
- **RNA-seq**: `GSM2715529`-`GSM2715531` = DMSO rep1-3; `GSM2715532`-`GSM2715534` = AI-10-49 rep1-3.
- **ChIP-seq H3K27ac**: `GSM2715535` = DMSO, `GSM2715536` = AI-10-49. **Input DNA**: `GSM2715539` = DMSO, `GSM2715540` = AI-10-49.
- **ChIP-seq RUNX1**: `GSM2715537` = DMSO, `GSM2715538` = AI-10-49.
- **ATAC-seq**: `GSM2715541`-`GSM2715542` = DMSO rep1-2; `GSM2715543`-`GSM2715544` = AI-10-49 rep1-2.

- `data/sample_sheet_all.tsv`: Master sample sheet for the GSE101788/101789/101790 cohort. Columns: `gse`, `run_accession` (SRR), `experiment_accession` (SRX), `biosample_accession`, `geo_sample_accession` (GSM), `geo_sample_title`, `assay` (RNA-Seq / ChIP-Seq / ATAC-seq), `library_layout`, `condition` (DMSO / AI-10-49), `treated_with`, `chip_antibody` (e.g., `Anti-Histone H3 (acetyl K27) antibody`, `Anti-RUNX1/AML1 antibody`), `cell_type`.
- `processed_data/atacseq/peaks/GSM27155{41,42,43,44}_atac_peaks.narrowPeak`: MACS2 narrowPeak files for ATAC-seq. Also `_peaks.xls` (MACS2 verbose XLS) and `_summits.bed` (per-peak summit coordinates).
- `processed_data/atacseq/peaks/atacseq_peak_calls.tsv`: ATAC peak-calling manifest. Columns: `sample_id`, `run_accession`, `condition`, `filtered_bam_noMT`, `nucfree_bam`, `bigwig`, `peak_prefix`, `narrowpeak`.

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

- `data/sample_sheet_all.tsv`: Located at `/app/data/data/sample_sheet_all.tsv`
- `processed_data/atacseq/peaks/GSM2715541_atac_peaks.narrowPeak`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715541_atac_peaks.narrowPeak`
- `processed_data/atacseq/peaks/GSM2715541_atac_peaks.xls`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715541_atac_peaks.xls`
- `processed_data/atacseq/peaks/GSM2715541_atac_summits.bed`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715541_atac_summits.bed`
- `processed_data/atacseq/peaks/GSM2715542_atac_peaks.narrowPeak`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715542_atac_peaks.narrowPeak`
- `processed_data/atacseq/peaks/GSM2715542_atac_peaks.xls`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715542_atac_peaks.xls`
- `processed_data/atacseq/peaks/GSM2715542_atac_summits.bed`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715542_atac_summits.bed`
- `processed_data/atacseq/peaks/GSM2715543_atac_peaks.narrowPeak`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715543_atac_peaks.narrowPeak`
- `processed_data/atacseq/peaks/GSM2715543_atac_peaks.xls`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715543_atac_peaks.xls`
- `processed_data/atacseq/peaks/GSM2715543_atac_summits.bed`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715543_atac_summits.bed`
- `processed_data/atacseq/peaks/GSM2715544_atac_peaks.narrowPeak`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715544_atac_peaks.narrowPeak`
- `processed_data/atacseq/peaks/GSM2715544_atac_peaks.xls`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715544_atac_peaks.xls`
- `processed_data/atacseq/peaks/GSM2715544_atac_summits.bed`: Located at `/app/data/processed_data/atacseq/peaks/GSM2715544_atac_summits.bed`
- `processed_data/atacseq/peaks/atacseq_peak_calls.tsv`: Located at `/app/data/processed_data/atacseq/peaks/atacseq_peak_calls.tsv`
