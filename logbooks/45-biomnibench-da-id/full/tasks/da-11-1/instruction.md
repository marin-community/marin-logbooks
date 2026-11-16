<!-- TASK_ID: da-11-1 -->

# Task: KIR+CD8+ T Cells in Autoimmune Diseases - DA-11-1

## Question

Analyze the single-cell RNA-seq data from the inflamed tissues of celiac disease (CeD) patients to infer cell-cell communication networks, specifically focusing on ligand-receptor interactions between KIR+CD8+ T cells and pathogenic gliadin-specific CD4+ T cells. Identify the top signaling pathways, beyond the established MHC-I interaction, that are predicted to mediate the suppressive function and cytotoxic targeting observed in the in vitro assays.

## Data Files

Single-cell RNA-seq cohort from the "KIR+CD8+ T cells suppress pathogenic T cells" study (Science; GSE193442), intestinal mucosal biopsies from 6 active-celiac-disease patients and 4 healthy controls.

- `GSE193442_RAW/`: Per-sample gene × cell count matrices. Contains 10 gzipped tab-separated text files, one per sample:
  - 6 celiac-disease samples: `GSM5820724_MS01092019_counts.txt.gz` through `GSM5820729_MS9020_counts.txt.gz` (MS-prefixed mucosa-sample IDs).
  - 4 healthy controls: `GSM5820730_HC11_counts.txt.gz` through `GSM5820733_HC14_counts.txt.gz`.
  - Format per file: header row lists cell barcodes (e.g., `AAACCTGAGACCTAGG.1`); subsequent rows are one gene per row with integer UMI counts. Orientation: genes × cells.

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
