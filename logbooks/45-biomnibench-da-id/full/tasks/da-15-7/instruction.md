<!-- TASK_ID: da-15-7 -->

# Task: Integrative Transcriptomic Analysis of ALS Spinal Cord - DA-15-7

## Question

To identify biomarkers predictive of ALS progression rate, which individual genes show the strongest correlation with disease duration?

## Data Files

Postmortem ALS spinal-cord transcriptomics + proteomics dataset. Bulk RNA-seq is from the Nature Neuroscience "Integrative transcriptomic analysis of the amyotrophic lateral sclerosis spinal cord implicates glial activation and suggests new risk genes" paper (data via Zenodo record 6385747) spanning three spinal-cord levels (cervical, lumbar, thoracic); proteomics is the Oeckl et al. 2019 *Acta Neuropathologica* CSF-and-spinal-cord dataset for protein-level orthogonal validation.

- `Cervical_Spinal_Cord_gene_counts.tsv.gz`: Raw gene × sample read counts for the cervical spinal cord. First two columns: `ensembl_id`, `gene_name`; remaining ~100-175 columns are sample IDs (e.g., `sample_124`). Integer counts.
- `Cervical_Spinal_Cord_gene_tpm.tsv.gz`: TPM-normalized expression, same layout as the counts file.
- `Cervical_Spinal_Cord_metadata.tsv.gz`: Per-sample metadata. Key columns: `rna_id` (matches count/TPM column headers), `dna_id`, `site_id`, `tissue`, `age_rounded`, `sex`, `subject_group` (e.g., `ALS Spectrum MND`, `Non-Neurological Control`), `disease` (`ALS`, `control`), `site_of_motor_onset`, `disease_duration` (years), `mutations` (e.g., `C9orf72`, `SOD1`, `TARDBP`), `library_prep`, `seq_platform`, `rin` (RNA integrity), `c9orf72_repeat_size`, genotype PCs (`gPC1`-`gPC5`), and standard RNA-seq QC metrics.
- `Lumbar_Spinal_Cord_gene_counts.tsv.gz`: Raw gene × sample read counts for the lumbar spinal cord. First two columns: `ensembl_id`, `gene_name`; remaining ~100-175 columns are sample IDs (e.g., `sample_124`). Integer counts.
- `Lumbar_Spinal_Cord_gene_tpm.tsv.gz`: TPM-normalized expression, same layout as the counts file.
- `Lumbar_Spinal_Cord_metadata.tsv.gz`: Per-sample metadata. Key columns: `rna_id` (matches count/TPM column headers), `dna_id`, `site_id`, `tissue`, `age_rounded`, `sex`, `subject_group` (e.g., `ALS Spectrum MND`, `Non-Neurological Control`), `disease` (`ALS`, `control`), `site_of_motor_onset`, `disease_duration` (years), `mutations` (e.g., `C9orf72`, `SOD1`, `TARDBP`), `library_prep`, `seq_platform`, `rin` (RNA integrity), `c9orf72_repeat_size`, genotype PCs (`gPC1`-`gPC5`), and standard RNA-seq QC metrics.
- `Thoracic_Spinal_Cord_gene_counts.tsv.gz`: Raw gene × sample read counts for the thoracic spinal cord. First two columns: `ensembl_id`, `gene_name`; remaining ~100-175 columns are sample IDs (e.g., `sample_124`). Integer counts.
- `Thoracic_Spinal_Cord_gene_tpm.tsv.gz`: TPM-normalized expression, same layout as the counts file.
- `Thoracic_Spinal_Cord_metadata.tsv.gz`: Per-sample metadata. Key columns: `rna_id` (matches count/TPM column headers), `dna_id`, `site_id`, `tissue`, `age_rounded`, `sex`, `subject_group` (e.g., `ALS Spectrum MND`, `Non-Neurological Control`), `disease` (`ALS`, `control`), `site_of_motor_onset`, `disease_duration` (years), `mutations` (e.g., `C9orf72`, `SOD1`, `TARDBP`), `library_prep`, `seq_platform`, `rin` (RNA integrity), `c9orf72_repeat_size`, genotype PCs (`gPC1`-`gPC5`), and standard RNA-seq QC metrics.
- `gencode.v30.gene_meta.tsv.gz`: GENCODE v30 gene-symbol ↔ Ensembl-ID lookup. Columns: `genename`, `geneid`.

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
