<!-- TASK_ID: da-3-5 -->

# Task: Mutation Enrichment Analysis in Anti-PD-1 Therapy Responders vs Non-Responders

## Question

Using Supplementary Table 1D, identify which specific mutations are significantly enriched in responders or non-responders. Report statistical significance.

## Data Files

WES cohort from Hugo et al. 2016 melanoma anti-PD-1 study (GSE78220), pre-treatment tumor biopsies with irRECIST response annotations. Auxiliary background-mutation datasets from the TCGA melanoma cohort (CGAN 2015) and the Hodis et al. 2012 melanoma exome study are also included.

- `supplementary_tables.xls`: Supplementary tables from Hugo et al. (Cell 2016) containing multiple sheets:
  - Sheet `S1A`: Patient clinical characteristics with columns including Patient ID, irRECIST, Gender, Age, Disease Status, Treatment, Biopsy Time, and mutation status (BRAF, NRAS, NF1). Header rows need to be skipped (skip=2)
  - Sheet `S1B`: WES summary per patient with columns including Patient ID, Response (R/NR), TotalNonSyn, Purity, Ploidy, and neoepitope predictions. Header rows need to be skipped (skip=2)
  - Sheet `S1D`: Per-mutation somatic variant calls for all patients with columns: Chr, Pos, NucMut, Sample (patient ID, e.g., Pt1), Gene (Hugo symbol), Transcript, Detail, MutType, Aamut, CodonChange, cDNAmut, ConsvScr, Database, NumCall, MUTECT, HapCaller, VarScan2. Header rows need to be skipped (skip=2). This is the primary data source for identifying per-patient gene mutations
  - Sheet `S1E`: Pre-computed gene-level mutation enrichment results with columns: Gene, RatioRtoNR, NumHitsR, PvalR, pAdjR, LorR, NumHitsNR, PvalNR, pAdjNR, LorNR, BGhit
- `CGAN_mmc3.xlsx`: TCGA melanoma mutation significance analysis (Cancer Genome Atlas Network, Cell 2015). Contains 5 supplemental table sheets. The key sheet is `Supplemental Table S2A` (header on row 2) with columns: rank, gene, description, N, n, npat, nsite, nsil, n1-n6, NS:S, p_clust, p_cons, p_joint, p_cv, p, q. Used to estimate background mutation frequency per gene
- `Hodis_mmc2.txt`: Raw mutation annotation file (MAF format) from Hodis et al. (2012) melanoma exome study. Tab-separated with an Oncotator header comment on line 1 (skip=1). Columns include: Hugo_Symbol, Chromosome, Start_position, End_position, Variant_Classification, Variant_Type, Reference_Allele, Tumor_Seq_Allele1/2, Tumor_Sample_Barcode, and other standard MAF fields. Contains mutations from approximately 121 melanoma samples, used to count unique patients (Tumor_Sample_Barcode) with non-synonymous mutations per gene for background frequency estimation

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
