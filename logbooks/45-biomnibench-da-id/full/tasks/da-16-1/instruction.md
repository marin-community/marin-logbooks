<!-- TASK_ID: da-16-1 -->

# Task: Molecular Subtypes of NAFLD by Unsupervised Clustering - DA-16-1

## Question

To identify molecular subtypes of NAFLD (non-alcoholic fatty liver disease) for clinical-trial enrichment, can unsupervised clustering of hepatic transcriptomes stratify patients by disease severity? Perform hierarchical clustering and/or PCA on appropriately normalized gene expression. Assess whether the resulting clusters differ by fibrosis stage, NAS score, NASH stage, or early/moderate disease grouping. Report the distribution of clinical features across clusters with appropriate statistical tests.

## Data Files

GSE135251 hepatic RNA-seq cohort spanning the NAFLD spectrum (controls + NAFL + NASH at varying fibrosis stage / NAS score). Files under `/app/data/`:

- `GSE135251/`: per-sample HTSeq count files (`GSM*.counts.txt.gz`, ~216 files). Each file is a two-column gene-level count table (Ensembl gene id + count). The directory contains both genuine gene rows and HTSeq summary rows (`__no_feature`, `__ambiguous`, `__too_low_aQual`, `__not_aligned`, `__alignment_not_unique`) at the bottom of each file — drop these before counting genes.
- `geo/GSE135251_family.soft.gz`: SOFT family file with full sample metadata (sample title, characteristics fields including fibrosis stage, NAS score, NASH group, disease label, sex, age).
- `geo/GSE135251_series_matrix.txt.gz`: series matrix form of the same metadata (often easier to parse with `GEOparse` or pandas).

The cohort includes healthy controls. The NAFLD-clustering question is about disease cases — exclude controls before clustering unless you have a justified reason to keep them.

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
   - **Code**: paste the actual code (Python / R / shell) used for each non-trivial operation in this step — filters, aggregations, joins, statistical tests, transformations, plotting calls. Use real, copy-pasteable code (not pseudocode and not a paraphrase). The trace must be reproducible from these snippets, so do not skip the code for steps that do real work.
   - **Quantitative intermediate result** (counts, p-values, dimensions after filtering).
4. **Results**: Specific quantitative findings (cluster sizes, cluster-vs-feature p-values, summary tables); biological / clinical interpretation of what the clusters represent; what the analysis cannot conclude (limitations).
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
- Show intermediate result counts at every filtering step (e.g., "216 -> 206 -> 198")
- Name specific entities and quantities; avoid vague summaries
- Justify analytical choices (test selection, threshold, normalization, number of PCs / clusters) explicitly
- Ground biological / clinical claims in real references; do not assert from memory
- State what the analysis cannot conclude (limitations)

### 2. Final Answer (`/app/answer.txt`)

Write your final answer to the question above in plain text format.

## Environment

- Python 3 and R are pre-installed
- Install any additional packages you need (pandas, scikit-learn, scipy, statsmodels, GEOparse, edgeR/limma/DESeq2 in R, etc.)
- Internet access is available for package installation
- **Do not search for or read the specific source paper, figures, or supplementary materials** that the dataset comes from. Solve the task directly from the provided data and domain knowledge.
