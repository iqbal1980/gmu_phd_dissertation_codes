# Install and load the necessary packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GEOquery")
BiocManager::install("biomaRt")
 

 


library(GEOquery)
library(limma)
library(biomaRt)

# Define the search terms for the specific type of pericyte and ion channels
pericyte_type <- "brain pericyte"
ion_channel_keywords <- c("ion channel", "potassium channel", "calcium channel", "sodium channel", "chloride channel")

# Search for relevant datasets in GEO
geo_query <- paste0(pericyte_type, "[All Fields] AND (", paste(ion_channel_keywords, collapse = " OR "), ")")
geo_datasets <- getGEO(geo_query, getGPL = FALSE)

# Filter datasets based on specific criteria (e.g., organism, platform)
filtered_datasets <- lapply(geo_datasets, function(dataset) {
  if (dataset@organism == "Homo sapiens" && dataset@gpls[1] == "GPL570") {
    return(dataset)
  }
})
filtered_datasets <- filtered_datasets[!sapply(filtered_datasets, is.null)]

# Select the first dataset for further analysis
dataset <- filtered_datasets[[1]]

# Download the expression data and convert it to a matrix
expression_data <- exprs(dataset)

# Perform data normalization (e.g., quantile normalization)
normalized_data <- normalizeBetweenArrays(expression_data, method = "quantile")

# Annotate the gene symbols using biomaRt
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
gene_symbols <- getBM(attributes = c("affy_hg_u133_plus_2", "hgnc_symbol"), 
                      filters = "affy_hg_u133_plus_2", 
                      values = rownames(normalized_data), 
                      mart = mart)
rownames(normalized_data) <- gene_symbols$hgnc_symbol

# Filter the expression data to include only ion channel genes
ion_channel_genes <- grep(paste(ion_channel_keywords, collapse = "|"), rownames(normalized_data), value = TRUE)
ion_channel_data <- normalized_data[ion_channel_genes, ]

# Calculate the percentage of expression for each ion channel gene
sample_sums <- colSums(ion_channel_data)
percentage_matrix <- ion_channel_data / sample_sums * 100
ion_channel_percentages <- rowMeans(percentage_matrix)

# Sort the ion channel genes by percentage of expression in descending order
sorted_ion_channels <- sort(ion_channel_percentages, decreasing = TRUE)

# Print the top 20 ion channel genes with the highest percentage of expression
top_ion_channels <- head(sorted_ion_channels, 20)
print(top_ion_channels)