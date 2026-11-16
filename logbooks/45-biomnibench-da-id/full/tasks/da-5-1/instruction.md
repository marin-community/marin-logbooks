<!-- TASK_ID: da-5-1 -->

# Task: Prioritizing Druggable Protein Targets for Pancreatic Cancer (PDAC)

## Question

To prioritize druggable targets for pancreatic cancer (PDAC), which proteins are promising as drug targets by considering both significant overexpression in tumors compared to normal tissue AND dependency in cancer cell lines based on CRISPR knockout screens?

## Data Files

CPTAC pan-cancer proteogenomic druggability dataset (Cell 2024; 8 cohorts: CCRCC, COAD, HNSCC, LSCC, LUAD, OV, PDAC, UCEC), supplementary tables covering drug-target tiers, tumor-vs-normal differential expression, CNV/mutation drivers, and CRISPR dependencies.

- `mmc2.xlsx`: Protein-coding therapeutic targets and their assigned druggability tiers (5 sheets):
  - **Table S2A**: Drug target tier assignments with columns: Ensembl Gene ID, Gene Symbol, Assigned Tier, Possible Tiers, Pharos Protein Family (2863 genes)
  - **Table S2B**: Drug-target mappings with columns: Drug ID, Drug Name, Target Gene ID, Target Gene Symbol, Approved, Cancer Indication, Tier, Database
  - **Table S2C**: DGIdb category annotations with columns: Ensembl Gene ID, Gene Symbol, Category Sources (DGIdb), Category (DGIdb), Tier
  - **Table S2D**: Surfaceome annotations with columns: Ensembl Gene ID, Gene Symbol, Surfaceome Label Source, Tier

- `mmc3.xlsx`: Tumor-overexpressed proteins with genetic screen and genomic aberration evidence (5 sheets):
  - **Table S3A**: Integrated tumor overexpression and CRISPR dependency indicator matrix with columns: Ensembl Gene ID, GeneSymbol, EssentialGenes, Family, Tier, Tier1-Tier5 (tier membership flags), CCRCC, COAD, HNSCC, LSCC, LUAD, OV, PDAC, UCEC, CancerTypeCount (5262 genes)
  - **Table S3B**: Tumor vs. normal differential expression statistics per cancer type with columns: Cancer Type, Ensembl Gene ID, mean_difference.rna, pval.rna, mean_difference.protein, pval.protein, signed_p.rna, signed_p.protein, Tier, GeneSymbol
  - **Table S3C**: Mutation-driven expression with columns: Cancer Type, Ensembl Gene ID, correlation_coef.rna, pval.rna, correlation_coef.protein, pval.protein, signed_pval.rna, signed_pval.protein, Tier, GeneSymbol, overexpressed_protein
  - **Table S3D**: CNV-driven expression (cis effects) with columns: Cancer Type, Ensembl Gene ID, correlation_coef.rna, pval.rna, FDR.rna, correlation_coef.protein, pval.protein, fdr.prot, target, GeneSymbol, overexpressed_protein

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
