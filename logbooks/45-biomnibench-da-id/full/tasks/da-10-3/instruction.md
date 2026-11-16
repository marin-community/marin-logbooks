<!-- TASK_ID: da-10-3 -->

# Task: ML Screening of Membraneless Organelle Participants - DA-10-3

## Question

Using the provided dataframe containing (i) multiple dataset membership indicator columns (0/1) for MLO participant sets and membrane-bound organelle control sets, (ii) a fixed non-PS negative/background set (e.g., hNoPS-test), and (iii) multiple predictor score columns (e.g., PLAAC, PScore, catGRANULE, SaPS-8fea/10fea, PdPS-8fea/10fea), evaluate and compare screening performance across datasets. Determine whether predictors perform consistently better on MLO participant datasets than on membrane-bound organelle controls.

## Data Files

Phase-separation (PS) predictor benchmark dataset, annotated human proteome with IDR ranges and sequence-level features, multiple published PS-predictor scores (PScore, PLAAC, catGRANULE, DeepPhase, SaPS/PdPS 8/10-feature classifiers), and curated MLO (membraneless organelle) and BioID-interactome localization labels for evaluating how well each predictor discriminates self-assembling (SaPS) vs partner-dependent (PdPS) vs non-PS (NoPS) proteins.

- `h_combined_with_flags.xlsx`: Human proteome master table (single sheet, ~8,957 proteins × 18 columns). Columns: `UniprotEntry`, `Gene name`, `Organism`, `Organism ID`, `length`, `Hydropathy`, `FCR`, `IDR`, `LCR`, predictor scores (`PScore`, `PLAAC`, `catGRANULE`, `DeepCoil`, `DeepPhase`, `Phos freq`), and class-membership flags (`hSaPS`, `hPdPS`, `hNoPS`).
- `sequence_prediction.xlsx`: Predictor scores and ranks across ~116k sequences (single sheet). Columns: `Organism`, `Sequence`, raw scores (`catGRANULE`, `PLAAC`, `PScore`, `SaPS-8fea`, `PdPS-8fea`) and their percentile-rank counterparts (`*_rnk`).
- `List_of_proteins_for_comparing_MLO_enrichment.xlsx`: MLO-participant dataset membership indicators (row 1 is a title, actual header is row 2). Two sheets: `Original data` (~3,986 proteins) and `Training proteins removed` (~3,815 proteins). Columns: `UniprotEntry` followed by 0/1 flags for MLO/membrane-organelle datasets (OpenCell nuclear punctae, DACT1 particulate proteome, G3BP1 proximity labeling, PhaSepDB high-throughput, Mitochondrial proteomics, Amyloid fiber-forming, etc.).

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
