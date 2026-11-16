<!-- TASK_ID: da-25-1 -->

# Task: DA-25-1

## Background
In real world setting, we would like to leverage as many datasets as possible to address biology questions. Combining datasets from different sources are always very challenging. We picked 4 mRNA datasets of different disease state (TCGA primary prostate cancer, GSE120741 primary prostate cancer, GSE126078 castration-resistant prostate cancer, GSE118435 castration-resistant prostate cancer) and 1 DNA-seq dataset from TCGA which are from the same samples in TCGA mRNA dataset.

## Question
For TCGA dataset, if we want to focus on only Pathogenic/Oncogenic alterations, what are the top mutated genes? Is there any way to visualize the results? At genomics level, is there any gene associates with clinical T stage?

## Data Files

Prostate-cancer transcriptome + mutation cohort from the "Dynamic prostate cancer transcriptome analysis delineates the trajectory to disease" paper (Nature Communications). Four independent mRNA datasets spanning primary prostate cancer and castration-resistant prostate cancer (CRPC) plus matched TCGA mutation calls.

- `GSE118435_RAW.tar`: GEO raw tarball for GSE118435 (castration-resistant prostate cancer bulk RNA-seq). Contains per-sample count/expression files.
- `GSE120741_Porto_ge_table.txt.gz`: GSE120741 primary prostate cancer gene-expression table (tab-separated, genes × samples).
- `GSE126078_CRPC.zip`: Zipped archive of GSE126078 CRPC gene-expression data.
- `data_clinical_patient_TCGA.txt`: TCGA prostate cancer (PRAD) patient-level clinical file in cBioPortal format. **Rows 1-4 are comment-prefixed header lines; actual column header is on row 5.** Columns include `PATIENT_ID`, demographics, clinical T/N/M stage, Gleason score, and outcome fields.
- `data_mrna_seq_v2_rsem_TCGA.txt`: TCGA PRAD mRNA expression (RSEM-normalized). Tab-separated gene × sample matrix. First two columns: `Hugo_Symbol`, `Entrez_Gene_Id`; remaining columns are TCGA sample IDs.
- `data_mutations_TCGA.txt`: TCGA PRAD somatic mutations in MAF format (tab-separated). Key columns: `Hugo_Symbol`, `Variant_Classification`, `Variant_Type`, `HGVSp_Short`, `Tumor_Sample_Barcode`, functional annotation columns (including pathogenicity/oncogenicity fields).

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
