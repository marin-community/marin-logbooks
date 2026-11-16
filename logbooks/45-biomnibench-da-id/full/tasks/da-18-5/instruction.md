<!-- TASK_ID: da-18-5 -->

# Task: BiomniBench-DA Task DA-18-5

## Question

To evaluate MAPK pathway activation as an endocrine resistance mechanism, what is the cumulative frequency of MAPK pathway alterations in post-hormonal therapy HR+HER2- tumors?

## Data Files

Targeted-panel (MSK-IMPACT) genomic profiling cohort from the "Genomic Landscape of Endocrine-Resistant Advanced Breast Cancers" paper (Cancer Cell; cBioPortal-format files). Primary and metastatic breast-cancer samples annotated with receptor subtype (HR+/HER2-, HR+/HER2+, HR-/HER2+, TNBC) and pre/post-hormonal-therapy exposure.

- `data_clinical_sample.txt`: Sample-level clinical table in cBioPortal format. **Rows 1-4 are comment-prefixed header lines (`#` display names, descriptions, types, priority), the actual column header is on row 5.** Key columns: `PATIENT_ID`, `SAMPLE_ID` (MSK-IMPACT format, e.g. `P-0000004-T01-IM3`), `SAMPLE_TYPE` (Primary / Metastasis), `SAMPLE_SITE`, `TUMOR_TISSUE_ORIGIN` (`Treatment Naive Primary` / `Post-Treatment Primary` / metastatic site), receptor-status flags for both the primary tumor (`RECEPTOR_STATUS_PRIMARY`, `ER/PR/HER2_*_PRIMARY`) and the sequenced sample (`HR_STATUS`, `ER_STATUS`, `PR_STATUS`, `OVERALL_HER2_STATUS`, `HER2_IHC_STATUS`, `HER2_FISH`), `ONCOTREE_CODE`, `CANCER_TYPE_DETAILED`, and `TMB_NONSYNONYMOUS` (mutations per Mb).
- `data_cna.txt`: Gene × sample discrete copy-number alteration matrix in cBioPortal format. Row index: `Hugo_Symbol`; column index: `SAMPLE_ID`. Values follow the cBioPortal convention (`-2` = deep deletion, `-1` = shallow deletion, `0` = neutral, `1` = gain, `2` = amplification).
- `data_mutations.txt`: Mutation Annotation Format (MAF) table. Key columns: `Hugo_Symbol`, `Entrez_Gene_Id`, `NCBI_Build` (GRCh37), `Chromosome`, `Start_Position`, `End_Position`, `Variant_Classification` (e.g., `Missense_Mutation`, `Nonsense_Mutation`, `Frame_Shift_*`, `Splice_Site`), `Variant_Type`, `Reference_Allele`, `Tumor_Seq_Allele2`, `Tumor_Sample_Barcode`, `HGVSc`, `HGVSp`, `HGVSp_Short` (e.g., `p.F373L`), `Protein_position`, `Hotspot` flag, plus read-count fields (`t_ref_count`, `t_alt_count`, `n_ref_count`, `n_alt_count`).

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

- `data_clinical_sample.txt`: Located at `/app/data/data_clinical_sample.txt`
- `data_cna.txt`: Located at `/app/data/data_cna.txt`
- `data_mutations.txt`: Located at `/app/data/data_mutations.txt`
