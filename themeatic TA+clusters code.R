# install.packages(c("tidyverse", "ggplot2", "readr"))
library(tidyverse)

# ---------------------------
# 1) Topic → Label mapping
# ---------------------------
topics <- tribble(
  ~TopicID, ~TopicName, ~Primary,       ~Secondary,
  1, "Image-Based Urban Modelling",                 "Form",        "Context,Grain",
  2, "Urban Design & Affective/Generative Space",   "Public-realm","Form",
  3, "Urban Clustering & Built-Form Modelling",     "Grain",       "Form,Land-use",
  4, "Vegetation Mapping (LiDAR/Point Cloud)",      "Vegetation",  "Context",
  5, "Urban Network Prediction Modelling",          "Connectivity","Land-use,Public-realm",
  6, "Urban Landscape Indicators & Modelling",      "Public-realm","Vegetation,Form,Context",
  7, "Urban Land-Use & Public-Space Analytics",     "Land-use",    "Public-realm,Connectivity",
  8, "Street-View Imagery for Urban Environmental Analysis", "Public-realm","Form,Vegetation,Land-use"
)

# ---------------------------
# 2) Define GLVIA-style attributes (target labels)
# ---------------------------
attributes <- c("Context","Topography","Grain","Form","Land-use","Water","Vegetation","Public-realm","Connectivity")

# ---------------------------
# 3) Weights for rubric
#    primary=2, secondary=1 by default.
#    You can tune e.g. Topography/Water upweighting if desired.
# ---------------------------
w_primary   <- 2
w_secondary <- 1

# Helper to explode comma lists
explode <- function(x) {
  ifelse(is.na(x) | x=="", character(0), strsplit(x, ",")[[1]] %>% trimws())
}

# ---------------------------
# 4) Build long table of contributions
# ---------------------------
contrib <- topics %>%
  rowwise() %>%
  mutate(sec = list(explode(Secondary))) %>%
  ungroup() %>%
  select(TopicID, TopicName, Primary, sec) %>%
  # primary rows
  mutate(Primary = Primary) %>%
  pivot_longer(cols = c(Primary), names_to = "kind", values_to = "Label") %>%
  mutate(weight = ifelse(kind=="Primary", w_primary, w_secondary)) %>%
  select(TopicID, TopicName, Label, weight) %>%
  # add secondary rows
  bind_rows(
    topics %>%
      rowwise() %>%
      mutate(sec = list(explode(Secondary))) %>%
      ungroup() %>%
      select(TopicID, TopicName, sec) %>%
      unnest(sec) %>%
      transmute(TopicID, TopicName, Label = sec, weight = w_secondary)
  ) %>%
  filter(Label %in% attributes)

# ---------------------------
# 5) Create topic short codes (T1..T8)
# ---------------------------
topic_codes <- tibble(
  TopicID   = 1:8,
  TopicCode = paste0("T", 1:8)
)

# ---------------------------
# 6) Matrix: Attributes (rows) × Topics (columns)
# ---------------------------
matrix_df <- contrib %>%
  left_join(topic_codes, by="TopicID") %>%
  group_by(Label, TopicCode) %>%
  summarise(score = sum(weight), .groups="drop") %>%
  complete(Label = attributes, TopicCode = paste0("T", 1:8), fill = list(score = 0)) %>%
  arrange(factor(Label, levels = attributes), TopicCode)

# Optional: upweight *Topography* and *Water* if you want to emphasise gaps
# matrix_df <- matrix_df %>%
#   mutate(score = ifelse(Label %in% c("Topography","Water"), score*1.5, score))

# ---------------------------
# 7) Heatmap
# ---------------------------
library(ggplot2)

ggplot(matrix_df, aes(x = TopicCode, y = factor(Label, levels = rev(attributes)), fill = score)) +
  geom_tile(color = "white") +
  geom_text(aes(label = score)) +
  scale_fill_gradient(low = "white", high = "black") +
  labs(x = "Topic cluster", y = "Attributes", fill = "Score",
       title = "Coverage of Townscape Assessment Attributes by Topic Clusters") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

# ---------------------------
# 8) Export CSV/Excel-ready table if needed
# ---------------------------
wide_table <- matrix_df %>%
  pivot_wider(names_from = TopicCode, values_from = score) %>%
  arrange(factor(Label, levels = attributes))

readr::write_csv(wide_table, "GLVIA_Topic_Rubric_Matrix.csv")
# If you prefer Excel: openxlsx::write.xlsx(wide_table, "GLVIA_Topic_Rubric_Matrix.xlsx")
