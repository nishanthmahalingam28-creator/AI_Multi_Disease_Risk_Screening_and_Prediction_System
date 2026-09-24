"""
Diabetes Exploratory Data Analysis (EDA) Module.

Generates reproducible statistical summaries, missingness analysis,
IQR outlier profiles, and visualizations for the cleaned Diabetes dataset.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_diabetes_eda(data_path: Path, output_dir: Path) -> dict:
    """
    Executes comprehensive EDA on the cleaned Diabetes dataset.

    Args:
        data_path (Path): Path to cleaned.csv.
        output_dir (Path): Directory where generated plots will be saved.

    Returns:
        dict: Dictionary containing computed statistical metrics and outlier counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#1f77b4", "#e377c2"]  # Blue (Non-diabetic), Pink/Rose (Diabetic)

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    target_counts = df["Outcome"].value_counts().sort_index()
    target_labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    bars = ax.bar(target_labels, target_counts.values, color=palette, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Diabetes Target Distribution (N = 768)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, 580)
    plt.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Missingness Analysis Plot
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(null_cols.index, null_cols.values, color="#e41a1c", width=0.55, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Diabetes: Missing Values Generated from Biological Zeros", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Missing (NaN) Values", fontsize=11)
    ax.set_ylim(0, 450)
    plt.xticks(rotation=15, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "missingness_analysis.png", dpi=300)
    plt.close(fig)

    # 3. Continuous Numerical Feature Distributions
    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#1f77b4", edgecolor="black")
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
    axes[8].axis("off")
    plt.suptitle("Diabetes: Feature Histograms & Density (Valid Observations)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_distributions.png", dpi=300)
    plt.close(fig)

    # 4. Boxplots Stratified by Outcome
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.boxplot(x="Outcome", y=col, data=df, ax=axes[i], palette=palette, hue="Outcome", legend=False)
        axes[i].set_title(f"{col} by Outcome", fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
        axes[i].set_xlabel("Target Status", fontsize=10)
        axes[i].set_ylabel(col, fontsize=10)
    axes[8].axis("off")
    plt.suptitle("Diabetes: Risk Factors Stratified by Outcome", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "features_by_outcome.png", dpi=300)
    plt.close(fig)

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Diabetes: Feature Correlation Matrix (Pairwise Pearson)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # 6. IQR Outlier Analysis (on non-null data)
    outlier_summary = {}
    for col in feature_cols:
        series = df[col].dropna()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_summary[col] = {
            "valid_count": int(len(series)),
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(float(len(outliers) / len(series) * 100), 2)
        }

    results = {
        "dataset_shape": df.shape,
        "target_counts": target_counts.to_dict(),
        "missing_summary": null_cols.to_dict(),
        "outlier_summary": outlier_summary,
        "top_correlations_with_target": corr["Outcome"].sort_values(ascending=False).to_dict()
    }
    return results


if __name__ == "__main__":
    from ml.member1.config import get_disease_paths
    paths = get_disease_paths("diabetes")
    res = run_diabetes_eda(paths["data"] / "cleaned.csv", paths["reports"] / "plots")
    print("[Diabetes EDA] Generated all plots successfully.")
    print("[Diabetes EDA] Missing summary:\n", res["missing_summary"])
    print("[Diabetes EDA] Outlier Summary:\n", res["outlier_summary"])
    print("[Diabetes EDA] Top correlations with Outcome:\n", res["top_correlations_with_target"])
