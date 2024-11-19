library(writexl)
library(dplyr)
library(stringr)

#file path
file_path <- ('E:/Dessertation Data/Plot-Sampling-Lacuna.csv')

# read data
data <- read.csv(file_path)

# List of strings to drop
strings_to_drop <- c("a", "B", "A", "Coff", "not", "Zae", "yy", "y", "not1", "ff", "gg", "Aa", "Henry1", "t", "Na", "T.", "Na1")

# Remove rows with unwanted strings
data_cleaned <- data[!data$plant_name.Genus %in% strings_to_drop, ]


#Convert All Names to Lowercase
data$plant_name.Genus <- str_to_lower(data$plant_name.Genus)

#Trim Extra Whitespace
data$plant_name.Genus <- str_trim(data$plant_name.Genus)


unique_names <- unique(data$plant_name.Genus)

# counting the number of unique values within the the plant genus 
name_counts <- table(data$plant_name.Genus)

# Convert to a data frame and rename columns
name_counts_df <- as.data.frame(name_counts)
colnames(name_counts_df) <- c("Name", "Count")

# Sort by the "Count" column in ascending order
sorted_name_counts_df <- name_counts_df %>% arrange(desc(Count))

# View the sorted result
print(sorted_name_counts_df)

#write to an excel
write_xlsx(sorted_name_counts_df, "sorted_plant_genus_name_counts.xlsx")



