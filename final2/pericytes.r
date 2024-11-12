# Ensure the mlbench package is installed
if (!require(mlbench)) install.packages("mlbench")

# Load the mlbench package
library(mlbench)

# Ensure the ggplot2 package is installed for visualization
if (!require(ggplot2)) install.packages("ggplot2")

# Load the ggplot2 package
library(ggplot2)

# Load the Pima Indian diabetes dataset
data("PimaIndiansDiabetes")
pima_data <- PimaIndiansDiabetes

# View the first few rows of the dataset
print("First few rows of the Pima Indian Diabetes dataset:")
head(pima_data)

# Get a summary of the dataset
print("Summary of the Pima Indian Diabetes dataset:")
summary(pima_data)

# Check for missing values
missing_values <- sum(is.na(pima_data))
print(paste("Total missing values in the dataset:", missing_values))

# Plot the distribution of glucose levels
glucose_plot <- ggplot(pima_data, aes(x = glucose)) +
  geom_histogram(binwidth = 10, fill = "cornflowerblue", color = "black") +
  ggtitle("Distribution of Glucose Levels") +
  xlab("Glucose Level") +
  ylab("Frequency")

# Display the plot
plot(glucose_plot)

# Explore the relationship between glucose levels and diabetes outcome
outcome_plot <- ggplot(pima_data, aes(x = glucose, fill = factor(diabetes))) +
  geom_histogram(binwidth = 10, position = "dodge") +
  scale_fill_manual(values = c("neg" = "green", "pos" = "red"), name = "Diabetes Outcome") +
  ggtitle("Glucose Levels by Diabetes Outcome") +
  xlab("Glucose Level") +
  ylab("Count")

# Display the plot
plot(outcome_plot)