<!-- TASK_ID: da-26-4 -->

# Task: Synthetic-Lethal Partner Discovery for BRCA

## Question

Based on the provided multi-omics data and patient-level dependency scores, predict synthetic-lethal partners for loss-of-function events in BRCA. Combine statistical evidence from lasso regression linking molecular features to dependency, with mutual-exclusivity testing against the mutation matrix, and with paralog and protein-protein interaction priors. Rank candidates by translational relevance.

## Data Files

Translational cancer dependency map data from the "Building a translational cancer dependency map for The Cancer Genome Atlas" paper (Nature Cancer). Pairs CCLE cancer cell-line DepMap CRISPR essentiality scores with TCGA patient expression to produce patient-level predicted dependency scores; subset (`sub_`) files focus on the BRCA cohort used here.

- `sub_CCLE_TCGA_ID_meta.csv`: Metadata and ID mapping table linking CCLE cell lines to their TCGA-matched cancer types (BRCA and others). Contains sample/patient IDs, cancer-type labels, and subtype annotations (e.g., PAM50 where available).
- `sub_CCLE_depmapscore.txt`: CCLE cell-line DepMap gene essentiality scores (tab-separated, genes × cell lines; more negative = stronger dependency).
- `sub_CCLE_exp.Rdata`: R data object with CCLE cell-line gene expression (genes × cell lines).
- `sub_TCGA_depmapscore.txt`: Patient-level PREDICTED DepMap scores for TCGA (tab-separated, genes × TCGA patients). Output of the translational-dependency model applied to TCGA expression.
- `sub_TCGA_exp.txt`: TCGA patient gene expression (tab-separated, genes × patients). Used both as model input and for downstream clustering/association.
- `BRCA_damaging_mut.csv`: Patient × gene binary matrix of damaging/loss-of-function mutations in the TCGA BRCA cohort (1 = damaging mutation, 0 = none).

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
