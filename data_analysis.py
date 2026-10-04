import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    print(f"Original dataset shape: {df.shape}")
    return df

def clean_dataset(df):
    # 1. Copy dataframe
    df = df.copy()

    # 2. Clean column names
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]

    print("\nColumns after cleaning:")
    print(df.columns.tolist())

    # 3. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\nDuplicate rows removed: {before - after}")

    # 4. Show missing values before cleaning
    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # 5. Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled NaN in {col} with median: {median_val}")

    # 6. Fill missing categorical values with mode or "Unknown"
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_value = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            print(f"Filled NaN in {col} with mode: {fill_value}")

    # 7. Standardize values for common string columns
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        df["gender"] = df["gender"].replace({
            "male": "Male",
            "m": "Male",
            "female": "Female",
            "f": "Female",
            "other": "Other",
            "unknown": "Unknown",
            "nan": "Unknown",
            "none": "Unknown"
        })
        df["gender"] = df["gender"].fillna("Unknown")

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()
        df["city"] = df["city"].replace({"nan": "Unknown", "none": "Unknown", "": "Unknown"})

    if "product_category" in df.columns:
        df["product_category"] = df["product_category"].astype(str).str.strip()
        df["product_category"] = df["product_category"].replace({"nan": "Unknown", "none": "Unknown", "": "Unknown"})

    # 8. Convert date columns
    for col in ["signup_date", "purchase_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 9. Handle impossible / invalid numeric values
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=18, upper=100)

    if "income" in df.columns:
        df["income"] = df["income"].clip(lower=0, upper=500000)

    if "purchase_amount" in df.columns:
        df["purchase_amount"] = df["purchase_amount"].clip(lower=0, upper=100000)

    # 10. Remove outliers using IQR method
    print("\nOutlier check using IQR clipping:")
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"{col}: {outliers} outliers clipped")
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print("\nMissing values after cleaning:")
    print(df.isna().sum())

    return df

def create_visualizations(df):
    ensure_output_dir()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Correlation heatmap (if enough numeric columns)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("outputs/correlation_heatmap.png")
        plt.show()

    # 2. Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        cols_to_plot = numeric_cols[:4]
        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 5 * len(cols_to_plot)))
        for ax, col in zip(axes, cols_to_plot):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig("outputs/distribution_plots.png")
        plt.show()

    # 3. Boxplot to inspect outliers
    if len(numeric_cols) > 0:
        plot_cols = numeric_cols[:min(4, len(numeric_cols))]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[plot_cols], orient="h")
        plt.title("Boxplot of Numeric Variables")
        plt.tight_layout()
        plt.savefig("outputs/boxplot_numeric.png")
        plt.show()

    # 4. Count plots for categorical features
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and df[col].nunique() <= 20
    ]

    for col in categorical_cols[:3]:
        plt.figure(figsize=(10, 5))
        order = df[col].value_counts().head(10).index
        sns.countplot(data=df, x=col, order=order)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"outputs/{col}_countplot.png")
        plt.show()

    # 5. Sales by category (if applicable)
    if "product_category" in df.columns and "purchase_amount" in df.columns:
        grouped = df.groupby("product_category")["purchase_amount"].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=grouped.index, y=grouped.values)
        plt.title("Total Purchase Amount by Product Category")
        plt.xticks(rotation=45)
        plt.ylabel("Total Purchase Amount")
        plt.tight_layout()
        plt.savefig("outputs/category_sales.png")
        plt.show()

    # 6. Scatter plot for relationship between income and purchase amount
    if "income" in df.columns and "purchase_amount" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="income", y="purchase_amount", hue="gender" if "gender" in df.columns else None)
        plt.title("Income vs Purchase Amount")
        plt.tight_layout()
        plt.savefig("outputs/income_vs_purchase.png")
        plt.show()

def generate_summary(df):
    print("\n=== Dataset Summary ===")
    print(df.describe(include="all").T)

    print("\n=== Key Insights ===")

    if "age" in df.columns:
        print(f"- Median age: {df['age'].median():.1f} years")

    if "income" in df.columns:
        print(f"- Median income: ${df['income'].median():,.2f}")

    if "purchase_amount" in df.columns:
        print(f"- Average purchase amount: ${df['purchase_amount'].mean():,.2f}")
        top_category = df.groupby("product_category")["purchase_amount"].sum().idxmax() if "product_category" in df.columns else "N/A"
        print(f"- Top revenue-generating category: {top_category}")

    if "city" in df.columns:
        city_top = df["city"].value_counts().idxmax()
        print(f"- Most common city: {city_top}")

    if "satisfaction_score" in df.columns:
        print(f"- Average satisfaction score: {df['satisfaction_score'].mean():.2f}")

    if "gender" in df.columns:
        gender_dist = df["gender"].value_counts(normalize=True) * 100
        print(f"- Gender distribution: {gender_dist.to_dict()}")

def main():
    ensure_output_dir()
    df = load_data("data/raw_data.csv")
    df = clean_dataset(df)
    create_visualizations(df)
    generate_summary(df)

if __name__ == "__main__":
    main()