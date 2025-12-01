# AI-ML-Townscape-STM
This repository contains the scripts, data, and outputs that underpin the study:  “Fragmentation and Integration in AI-driven Urban Design: A Structural Topic Modelling Review of Townscape Assessment”

This repository contains the scripts, data, and outputs that underpin the study:

“Mapping the Role of AI and Machine Learning in Urban Design Assessment: A Structural Topic Modelling Approach”
Note: Full-text PDFs are not shared due to copyright restrictions, but DOIs are provided for replication.

search_strings.txt – Full Scopus search queries and inclusion criteria.
included_records.csv – Metadata of included publications (DOI, title, year, journal).
Note: Full-text PDFs are not shared due to copyright restrictions, but DOIs are provided for replication.

scripts/preprocessing.R – R script for text cleaning and preprocessing.

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

Overview GLVIA3 pipeline (TA1–TA9) using open data; Amsterdam case (Centrum vs IJburg).
Outputs: outputs/metrics_summary.csv, plots/*.png, outputs/ta_layers.gpkg.
Data sources (fetched automatically)
Buildings: 3DBAG LoD1.3 WFS → fallback to OSM.
Terrain: AHN DTM/DSM via WCS.
Land-use: CBS ELU WFS → fallback to OSM landuse/natural/leisure.
Network & water: OSM/OSMnx; Trees: Amsterdam Trees API.

Metrics implemented (TA1–TA9)
TA1 Enclosure + skyline roughness; TA2 slope vs height; TA3 tessellation-based grain;
TA4 Spacematrix (GSI/FSI/L) + optional period bins; TA5 land-use composition & κ vs ELU;
TA6 water adjacency share + enclosure correlation; TA7 trees per km;
TA8 100 m open-space index + closeness; TA9 intersection density, circuity, betweenness.

Requirements
Python 3.10+; packages: geopandas, osmnx, networkx, rasterio, shapely, momepy, scipy, matplotlib, requests, lxml, pandas, numpy.
Tip (Windows): use conda to get GDAL/GEOS/PROJ.

Townscape assessment report
The file TA_report.docx provides a narrative companion to the code and data. It describes the study areas (Amsterdam Centrum and IJburg), details all input datasets, and explains how each townscape attribute (TA1–TA9) is measured, including equations, statistics, and GLVIA3 relevance. Use this report as a guide when reproducing the metrics or adapting the pipeline to new cities.

Scopus Query Records
01_identification_original (573 records).This file contains the raw Scopus export from the initial search query.

02_screening_english only (557 records). After applying a language filter (English only), 557 records were retained. Non-English publications were excluded. This step corresponds to PRISMA: Initial screening.

03_screening (368 records).Titles and abstracts were screened for relevance based on inclusion/exclusion criteria such as:Urban design / townscape relevance, Use of ML / DL / AI, Empirical or methodological contribution, Exclusion of unrelated domains (traffic engineering, agriculture, medicine, robotics, etc.). This step corresponds to PRISMA: Title–abstract screening.

4_filtering (242 records). Further filtering was applied, removing:Duplicates, Non-peer-reviewed items, Short papers, workshop abstracts, and posters, Articles outside the domain scope.This step corresponds to PRISMA filtering.

05_Final 75 (full text 72). Final set of articles included for full-text analysis. 75 articles met inclusion criteria and 72 articles had full text accessible and were used for the final synthesis. This corresponds to PRISMA: Full-text eligibility & Inclusion.

README.md – Project overview and instructions.

LICENSE – License for reuse (e.g., MIT/CC-BY).




