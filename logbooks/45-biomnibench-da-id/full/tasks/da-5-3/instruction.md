<!-- TASK_ID: da-5-3 -->

# Task: Identifying Activating Phosphosites on Druggable Proteins with Tumor Upregulation and CRISPR Dependency

## Question

Which activating phosphosites on druggable proteins are significantly elevated in tumors and supported by CRISPR dependency of the host gene in matched cancer cell lines?

## Data Files

CPTAC pan-cancer proteogenomic druggability dataset (Cell 2024; 8 cohorts: CCRCC, COAD, HNSCC, LSCC, LUAD, OV, PDAC, UCEC), supplementary tables covering drug-target tiers, tumor-vs-normal differential expression, phosphosite abundance, kinase-substrate enrichment, and CRISPR dependencies.

- `mmc2.xlsx`: Protein-coding therapeutic targets and their assigned druggability tiers (5 sheets):
  - **Table S2A**: Drug target tier assignments with columns: Ensembl Gene ID, Gene Symbol, Assigned Tier, Possible Tiers, Pharos Protein Family (2863 genes)
  - **Table S2B**: Drug-target mappings with columns: Drug ID, Drug Name, Target Gene ID, Target Gene Symbol, Approved, Cancer Indication, Tier, Database
  - **Table S2C**: DGIdb category annotations with columns: Ensembl Gene ID, Gene Symbol, Category Sources (DGIdb), Category (DGIdb), Tier
  - **Table S2D**: Surfaceome annotations with columns: Ensembl Gene ID, Gene Symbol, Surfaceome Label Source, Tier

- `mmc4.xlsx`: Phosphosite abundance and kinase activity altered in tumor vs. normal samples (3 sheets):
  - **Table S4A**: Phosphosite-level tumor upregulation × CRISPR dependency joint-indicator matrix with columns: Phosphosite ID, Ensembl Gene ID, GeneSymbol, EssentialGenes, Family, Gene Symbol with site, ActivatingSite (1 = known activating phosphorylation), Tier, Tier1-Tier5 (tier membership flags), CCRCC, COAD, HNSCC, LSCC, LUAD, OV, PDAC, UCEC, CancerTypeCount (18,169 phosphosites). **IMPORTANT — column semantics for the 8 cancer-type columns (CCRCC … UCEC):** each per-cancer flag is 1 only when the phosphosite is BOTH significantly upregulated in that cancer's tumor-vs-normal phosphoproteomic comparison AND the host gene is a CRISPR dependency in matched cancer-type cell lines (the cohort uses adjusted p ≤ 0.01 jointly). The flag therefore already encodes the joint tumor-up-AND-dependency requirement; do not re-apply `EssentialGenes` (which is a global / pan-cancer essentiality column) as a second AND-filter on top of the per-cancer flags. `CancerTypeCount` = sum of the 8 per-cancer flags. To answer the question, melt to long phosphosite × cancer-type pairs where the per-cancer flag = 1 (filter to ActivatingSite = 1 and a non-null Tier for druggability); each such pair is a candidate hit.
  - **Table S4B**: Kinase-substrate enrichment analysis (KSEA) results with columns: Kinase, KSEA Z Score, Cancer Type, p value, adjusted p value

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
