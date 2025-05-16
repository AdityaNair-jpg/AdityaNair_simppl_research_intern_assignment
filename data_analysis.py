import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSONL data
data = []
with open('data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Error decoding line: {line}")
            continue

# Convert to DataFrame
df = pd.DataFrame(data)

# Display basic information
print(f"Total records: {len(df)}")
print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

# Show sample data
print("\nSample data:")
print(df.head(2).to_string())

# Basic statistics for numerical columns
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isna().sum())

# Display unique values for categorical columns (if applicable)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # Limit to first 5 columns to avoid too much output
    unique_count = df[col].nunique()
    print(f"\nUnique values in {col}: {unique_count}")
    if unique_count < 20:  # Only show if there aren't too many unique values
        print(df[col].value_counts().head(10))