<!-- TASK_ID: da-1-3 -->

# Task: Spatiotemporal Single-Cell Analysis of Immunotherapy Response in CRC - DA-1-3

## Question

Characterize the distribution of cell subsets in tumor tissue, adjacent normal tissue, and peripheral blood. Identify which cell types are specifically and highly enriched only in the tumor microenvironment.

## Data Files

Single-cell cohort from the spatiotemporal CRC anti-PD-1 study (Cell 2024; GSE236581), pre- and on-treatment tumor, adjacent normal, and peripheral blood samples.

- `GSE236581_counts.mtx`: Raw UMI counts in Matrix Market sparse format (genes × cells).
- `GSE236581_barcodes.tsv`: Cell barcodes (one per line, ordered to match columns of the counts matrix). Format `{SampleID}_{CellBarcode}`.
- `GSE236581_features.tsv`: Gene index with columns `gene_id`, `gene_symbol`, `feature_type`; ordered to match rows of the counts matrix (no header row).
- `GSE236581_CRC-ICB_metadata.txt`: Space-separated cell-level metadata. Columns: `orig.ident`, `nCount_RNA`, `nFeature_RNA`, `Ident`, `Patient`, `Treatment` (I = pre-treatment, II-IV = on-treatment), `Tissue` (Tumor / Normal / Blood), `MajorCellType`, `SubCellType`.
- `clinical.xlsx`: Patient and sample metadata. Three sheets (header is row 2):
  - `scRNA-seq patient meta`: clinical variables per patient (Response CR/PR/SD/PD, Tumor Regression Ratio, TMB, MSI/MSS, Treatment Regimen, etc.).
  - `scRNA-seq sample meta`: per-sample info (Biopsy Site, Treatment Stage, Treatment point, Sampling approach).
  - `Validation patient meta`: independent validation cohort with Response and Sample Type.

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
