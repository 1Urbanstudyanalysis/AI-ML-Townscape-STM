# =========================================================
# 0. Setup: install & load required packages
# =========================================================

required_packages <- c("pdftools", "readxl", "dplyr", "stm")

installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}

library(pdftools)
library(readxl)
library(dplyr)
library(stm)

# =========================================================
# 1. Paths: adjust these TWO lines for your machine
# =========================================================

pdf_folder <- "Paper_pdf"  # folder where PDFs are stored
excel_path <- "sensitivity_PDF_STM_scopus.xlsx"

# =========================================================
# 2. Read metadata from Excel and construct PDF paths
# =========================================================

metadata_df <- read_excel(excel_path)

cat("Columns in metadata file:\n")
print(names(metadata_df))

# ---- IMPORTANT ----
# We assume metadata_df has a column that lists the PDF file names,
# e.g. "file_name". If it's called something else (e.g. "Filename"),
# rename it here:

# Example (uncomment and adjust if needed):
# metadata_df <- metadata_df %>% rename(file_name = Filename)

if (!("file_name" %in% names(metadata_df))) {
  stop("metadata_df must contain a column named 'file_name'. Rename your column and rerun.")
}

# Build full PDF paths using folder + file_name
pdf_paths <- file.path(pdf_folder, metadata_df$file_name)

# Optional: quick check
cat("\nFirst few PDF paths:\n")
print(head(pdf_paths))

# =========================================================
# 3. Read all PDFs and attach text to metadata
# =========================================================

pdf_texts <- lapply(pdf_paths, function(f) {
  if (!file.exists(f)) {
    stop(paste("PDF not found:", f,
               "\nCheck pdf_folder or file_name in Excel."))
  }
  pages <- pdf_text(f)
  paste(pages, collapse = " ")
})

# Create Data by taking all metadata columns + a new 'text' column
Data <- metadata_df
Data$text <- unlist(pdf_texts)

# Ensure we have a 'file_name' column (already from Excel)
# Ensure Year is numeric (for prevalence)
if ("Year" %in% names(Data)) {
  Data$Year <- as.numeric(Data$Year)
} else {
  stop("metadata_df must contain a 'Year' column for prevalence formula (~ Year).")
}

# For later sensitivity check: ensure Title & Abstract exist
if (!("Title" %in% names(Data)))  Data$Title    <- ""
if (!("Abstract" %in% names(Data))) Data$Abstract <- ""

cat("\nPreview of Data (text truncated):\n")
print(head(Data[, c("file_name", "Year", "Title")]))


# =========================================================
# 4. STM main model on full text
# =========================================================

# STM's own preprocessing
out <- textProcessor(
  documents = Data$text,
  metadata  = Data
)

docs  <- out$documents
vocab <- out$vocab
meta  <- out$meta   # includes Year, file_name, Title, Abstract, etc.

# Optionally prune very rare/common terms (good practice)
prep <- prepDocuments(docs, vocab, meta)
docs  <- prep$documents
vocab <- prep$vocab
meta  <- prep$meta

# ---- 4a. Optional: search for good K (already done once in your project) ----
set.seed(123)
k_values <- seq(2, 14, by = 2)

k_search <- searchK(
  documents  = docs,
  vocab      = vocab,
  K          = k_values,
  prevalence = ~ Year,
  data       = meta
)

png("model_diagnostics.png", width = 1000, height = 1000)
plot(k_search)
dev.off()

# ---- 4b. Fit STM with chosen K = 8 (main model) ----
set.seed(123)
stm_model <- stm(
  documents  = docs,
  vocab      = vocab,
  K          = 8,
  prevalence = ~ Year,
  data       = meta,
  seed       = 123
)

cat("\nTop words per topic (full-text STM):\n")
print(labelTopics(stm_model, n = 10))


# =========================================================
# 5. Sensitivity Check 1: Title + Abstract vs Full Text
# =========================================================

cat("\n========================\n")
cat("Sensitivity 1: Title + Abstract vs Full Text\n")
cat("========================\n")

# Build reduced text from Title + Abstract
Data$Title    <- ifelse(is.na(Data$Title),    "", Data$Title)
Data$Abstract <- ifelse(is.na(Data$Abstract), "", Data$Abstract)

Data$text_ta_raw <- paste(Data$Title, Data$Abstract, sep = " ")

# Process title+abstract corpus
out_ta <- textProcessor(
  documents = Data$text_ta_raw,
  metadata  = Data
)

docs_ta  <- out_ta$documents
vocab_ta <- out_ta$vocab
meta_ta  <- out_ta$meta

# Fit STM on title+abstract only
set.seed(321)
stm_ta <- stm(
  documents  = docs_ta,
  vocab      = vocab_ta,
  K          = 8,
  prevalence = ~ Year,
  data       = meta_ta,
  seed       = 321
)

cat("\nTop words per topic (Title+Abstract STM):\n")
print(labelTopics(stm_ta, n = 10))

# Compare document-level topic proportions (theta)
theta_full <- stm_model$theta   # N_doc x 8
theta_ta   <- stm_ta$theta      # N_doc x 8

cor_mat_ta <- cor(theta_full, theta_ta)  # 8 x 8

cat("\nCorrelation matrix (Full-text vs Title+Abstract topics):\n")
print(round(cor_mat_ta, 2))

# Best-matching TA topic for each full-text topic
best_match_ta <- apply(cor_mat_ta, 1, which.max)

alignment_ta <- data.frame(
  Full_topic = paste0("Full_T", 1:8),
  TA_topic   = paste0("TA_T", best_match_ta),
  Correlation = round(cor_mat_ta[cbind(1:8, best_match_ta)], 2)
)

cat("\nAlignment: Full-text topics vs Title+Abstract topics:\n")
print(alignment_ta)

median_cor_ta <- median(alignment_ta$Correlation)
cat("\nMedian correlation (Full vs TA best matches):", median_cor_ta, "\n")


# =========================================================
# 6. Sensitivity Check 2: Split-Half Stability Across Time
# =========================================================

cat("\n========================\n")
cat("Sensitivity 2: Split-Half Stability Across Time\n")
cat("========================\n")

years <- meta$Year
cut_year <- median(years, na.rm = TRUE)

early_idx <- which(years <= cut_year)
late_idx  <- which(years >  cut_year)

cat("\nNumber of documents - Early:", length(early_idx),
    " Late:", length(late_idx), "\n")

docs_early <- docs[early_idx]
docs_late  <- docs[late_idx]

meta_early <- meta[early_idx, , drop = FALSE]
meta_late  <- meta[late_idx,  , drop = FALSE]

# STM on early half
set.seed(456)
stm_early <- stm(
  documents  = docs_early,
  vocab      = vocab,
  K          = 8,
  prevalence = ~ Year,
  data       = meta_early,
  seed       = 456
)

# STM on late half
set.seed(789)
stm_late <- stm(
  documents  = docs_late,
  vocab      = vocab,
  K          = 8,
  prevalence = ~ Year,
  data       = meta_late,
  seed       = 789
)

# Compare topic–word distributions (beta)
beta_full  <- exp(stm_model$beta$logbeta[[1]])
beta_early <- exp(stm_early$beta$logbeta[[1]])
beta_late  <- exp(stm_late$beta$logbeta[[1]])

cor_full_early <- cor(t(beta_full), t(beta_early))  # 8 x 8
cor_full_late  <- cor(t(beta_full), t(beta_late))   # 8 x 8

cat("\nCorrelation matrix (Full vs Early topics):\n")
print(round(cor_full_early, 2))

cat("\nCorrelation matrix (Full vs Late topics):\n")
print(round(cor_full_late, 2))

# Best matches
best_early <- apply(cor_full_early, 1, which.max)
best_late  <- apply(cor_full_late,  1, which.max)

alignment_early <- data.frame(
  Full_topic  = paste0("Full_T", 1:8),
  Early_topic = paste0("Early_T", best_early),
  Correlation = round(cor_full_early[cbind(1:8, best_early)], 2)
)

alignment_late <- data.frame(
  Full_topic = paste0("Full_T", 1:8),
  Late_topic = paste0("Late_T", best_late),
  Correlation = round(cor_full_late[cbind(1:8, best_late)], 2)
)

cat("\nAlignment: Full vs Early topics:\n")
print(alignment_early)

cat("\nAlignment: Full vs Late topics:\n")
print(alignment_late)

median_cor_early <- median(alignment_early$Correlation)
median_cor_late  <- median(alignment_late$Correlation)

cat("\nMedian correlation (Full vs Early best matches):", median_cor_early, "\n")
cat("Median correlation (Full vs Late best matches):",  median_cor_late,  "\n")

# =========================================================
# END OF SCRIPT
# =========================================================


# =========================================================
# Sensitivity 2 (fixed): Split-Half Stability Across Time
# =========================================================
# Assumes you already have:
#   out       <- textProcessor(documents = Data$text, metadata = Data)
#   stm_model <- stm(...)   # full-text STM with K = 8
# =========================================================

library(stm)

cat("\n========================\n")
cat("Sensitivity 2: Split-Half Stability Across Time (fixed)\n")
cat("========================\n")

# 1) Use RAW docs/vocab/meta from textProcessor (before pruning)
docs_raw  <- out$documents
vocab_raw <- out$vocab
meta_raw  <- out$meta

years_all <- meta_raw$Year
cut_year  <- median(years_all, na.rm = TRUE)

early_idx <- which(years_all <= cut_year)
late_idx  <- which(years_all >  cut_year)

cat("\nNumber of documents - Early:", length(early_idx),
    " Late:", length(late_idx), "\n")

# 2) Preprocess early subset
prep_early <- prepDocuments(
  documents = docs_raw[early_idx],
  vocab     = vocab_raw,
  meta      = meta_raw[early_idx, ]
)

docs_early  <- prep_early$documents
vocab_early <- prep_early$vocab
meta_early  <- prep_early$meta

# 3) Preprocess late subset
prep_late <- prepDocuments(
  documents = docs_raw[late_idx],
  vocab     = vocab_raw,
  meta      = meta_raw[late_idx, ]
)

docs_late  <- prep_late$documents
vocab_late <- prep_late$vocab
meta_late  <- prep_late$meta

# 4) Fit STM on early and late halves
set.seed(456)
stm_early <- stm(
  documents  = docs_early,
  vocab      = vocab_early,
  K          = 8,
  prevalence = ~ Year,
  data       = meta_early,
  seed       = 456
)

set.seed(789)
stm_late <- stm(
  documents  = docs_late,
  vocab      = vocab_late,
  K          = 8,
  prevalence = ~ Year,
  data       = meta_late,
  seed       = 789
)

# 5) Compare topic–word distributions with the FULL model
beta_full  <- exp(stm_model$beta$logbeta[[1]])
vocab_full <- stm_model$vocab  # vocabulary used in the full model

beta_early <- exp(stm_early$beta$logbeta[[1]])
beta_late  <- exp(stm_late$beta$logbeta[[1]])

# ---- Align FULL vs EARLY: use only common words ----
common_e      <- intersect(vocab_full, vocab_early)
idx_full_e    <- match(common_e, vocab_full)
idx_early     <- match(common_e, vocab_early)

beta_full_e   <- beta_full[,  idx_full_e,  drop = FALSE]
beta_early_e  <- beta_early[, idx_early,   drop = FALSE]

cor_full_early <- cor(t(beta_full_e), t(beta_early_e))  # 8 x 8

cat("\nCorrelation matrix (Full vs Early topics, common vocab):\n")
print(round(cor_full_early, 2))

best_early <- apply(cor_full_early, 1, which.max)

alignment_early <- data.frame(
  Full_topic  = paste0("Full_T", 1:8),
  Early_topic = paste0("Early_T", best_early),
  Correlation = round(cor_full_early[cbind(1:8, best_early)], 2)
)

cat("\nAlignment: Full vs Early topics:\n")
print(alignment_early)

median_cor_early <- median(alignment_early$Correlation)
cat("\nMedian correlation (Full vs Early best matches):", median_cor_early, "\n")

# ---- Align FULL vs LATE: use only common words ----
common_l      <- intersect(vocab_full, vocab_late)
idx_full_l    <- match(common_l, vocab_full)
idx_late      <- match(common_l, vocab_late)

beta_full_l   <- beta_full[,  idx_full_l,  drop = FALSE]
beta_late_l   <- beta_late[,  idx_late,    drop = FALSE]

cor_full_late <- cor(t(beta_full_l), t(beta_late_l))  # 8 x 8

cat("\nCorrelation matrix (Full vs Late topics, common vocab):\n")
print(round(cor_full_late, 2))

best_late <- apply(cor_full_late, 1, which.max)

alignment_late <- data.frame(
  Full_topic = paste0("Full_T", 1:8),
  Late_topic = paste0("Late_T", best_late),
  Correlation = round(cor_full_late[cbind(1:8, best_late)], 2)
)

cat("\nAlignment: Full vs Late topics:\n")
print(alignment_late)

median_cor_late <- median(alignment_late$Correlation)
cat("\nMedian correlation (Full vs Late best matches):", median_cor_late, "\n")

