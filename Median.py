import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Sample dataset (replace with your own DataFrame)
# -----------------------------
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'Finance', 'HR', 'IT'],
    'Score': [85, 90, 78, 88, 76, 95, 89]
}
df = pd.DataFrame(data)

# -----------------------------
# Step 1: Frequency and Median for Categorical Features
# -----------------------------
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    print(f"\n--- {col} ---")
    freq = df[col].value_counts()
    print("Frequency:\n", freq)
    
    # Median of numerical column grouped by category
    for num_col in df.select_dtypes(include=['number']).columns:
        median_val = df.groupby(col)[num_col].median()
        print(f"Median {num_col} by {col}:\n", median_val)

# -----------------------------
# Step 2: Visualization
# -----------------------------
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"Frequency of {col}")
    plt.show()

    # Boxplot to see distribution of numerical values per category
    for num_col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=col, y=num_col, data=df, palette="Set3")
        plt.title(f"{num_col} Distribution by {col}")
        plt.show()

# -----------------------------
# Step 3: Transformation based on distribution
# -----------------------------
# Example: Frequency Encoding
for col in categorical_cols:
    freq_map = df[col].value_counts(normalize=True)  # relative frequency
    df[f"{col}_freq_enc"] = df[col].map(freq_map)

# Example: Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f"{col}_label_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Example: One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Output transformed data
# -----------------------------
print("\nData with Frequency & Label Encoding:\n", df)
print("\nData with One-Hot Encoding:\n", df_one_hot)
