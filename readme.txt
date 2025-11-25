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

README.md – Project overview and instructions.


LICENSE – License for reuse (e.g., MIT/CC-BY).
