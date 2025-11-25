This repository contains the scripts, data, and outputs that underpin the study:

“Mapping the Role of AI and Machine Learning in Urban Design Assessment: A Structural Topic Modelling Approach”
Note: Full-text PDFs are not shared due to copyright restrictions, but DOIs are provided for replication.

docs/

search_strings.txt – Full Scopus search queries and inclusion criteria.

data/

included_records.csv – Metadata of included publications (DOI, title, year, journal).

Note: Full-text PDFs are not shared due to copyright restrictions, but DOIs are provided for replication.

scripts/

preprocessing.R – R script for text cleaning and preprocessing.

stm_modelling.R – R script for STM fitting, diagnostics, and topic visualisations.

results/

topic_proportions.csv – Topic prevalence results.

diagnostics_plots.png – Coherence, exclusivity, residuals, and held-out likelihood plots.

wordclouds/ – Word clouds for each discovered topic.

GLVIA3 Topic–Attribute Rubric (R)

This single-file R script maps topic clusters (T1–T8) to GLVIA3 townscape attributes (Context, Topography, Grain, Form, Land-use, Water, Vegetation, Public-realm, Connectivity), applies a simple rubric (Primary = 2, Secondary = 1; configurable), and produces:

a heatmap showing coverage by topic, and

a CSV matrix (Attributes × Topics) for analysis and reporting. It provides a transparent way to evidence which parts of a townscape assessment framework (GLVIA3) are emphasised or under-represented across your thematic topic clusters (from STM/LSA/bibliometric analysis)

STM Sensitivity Checks for PDF Corpus (R)

One-file R workflow to read a corpus of PDFs with metadata, fit an STM model, and run two robustness checks: (1) Title+Abstract vs Full-Text; (2) Split-Half stability over time.
Outputs include a searchK() diagnostics plot and topic alignments/correlations printed to console.

What this script does?
Ingests data: Reads paper PDFs from a folder and metadata (Excel) with file names and fields (Year, Title, Abstract).
Builds corpora: Full-text corpus (from PDFs) and a reduced Title+Abstract corpus.
Fits STM: Runs searchK() (e.g., K = 2…14) and a main STM at K = 8 with prevalence = ~ Year.
Sensitivity 1 — Title+Abstract vs Full: Correlates document-level topic proportions and reports best topic matches.
Sensitivity 2 — Split-Half (Time): Splits by the median year and fits STM on early vs late halves. Compares topic–word distributions using a vocabulary-aligned correlation (common words only).
Saves diagnostics: model_diagnostics.png (from searchK()).

README.md – Project overview and instructions.

LICENSE – License for reuse (e.g., MIT/CC-BY).

