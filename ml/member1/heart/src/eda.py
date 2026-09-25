"""
Heart Disease Exploratory Data Analysis (EDA) Module.

Generates reproducible statistical summaries, IQR outlier profiles,
and publication-quality visualizations for the cleaned Heart Disease dataset.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_heart_eda(data_path: Path, output_dir: Path) -> dict:
    """
    Executes comprehensive EDA on the cleaned Heart Disease dataset.

    Args:
        data_path (Path): Path to cleaned.csv.
        output_dir (Path): Directory where generated plots will be saved.

    Returns:
        dict: Dictionary containing computed statistical metrics and outlier counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#2b5c8f", "#d95f02"]  # Professional medical palette: Blue (Absence), Orange/Red (Presence)

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    target_counts = df["Heart Disease"].value_counts().sort_index()
    target_labels = ["Absence (0)", "Presence (1)"]
    bars = ax.bar(target_labels, target_counts.values, color=palette, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Heart Disease Target Distribution (N = 270)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, 180)
    plt.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Continuous Numerical Feature Distributions
    num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2b5c8f", edgecolor="black")
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
    axes[5].axis("off")
    plt.suptitle("Heart Disease: Continuous Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_distributions.png", dpi=300)
    plt.close(fig)

    # 3. Numerical Features by Target (Boxplots)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x="Heart Disease", y=col, data=df, ax=axes[i], palette=palette, hue="Heart Disease", legend=False)
        axes[i].set_title(f"{col} by Heart Disease Status", fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(["Absence (0)", "Presence (1)"])
        axes[i].set_xlabel("Target Status", fontsize=10)
        axes[i].set_ylabel(col, fontsize=10)
    axes[5].axis("off")
    plt.suptitle("Heart Disease: Numerical Risk Factors Stratified by Outcome", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_boxplots.png", dpi=300)
    plt.close(fig)

    # 4. Categorical Clinical Predictors Stratified by Target
    cat_cols = ["Sex", "Chest pain type", "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, hue="Heart Disease", data=df, ax=axes[i], palette=palette, edgecolor="black")
        axes[i].set_title(f"{col} vs Heart Disease", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
        axes[i].legend(["Absence (0)", "Presence (1)"], title="Heart Disease", loc="upper right")
    plt.suptitle("Heart Disease: Key Categorical Clinical Indicators vs Outcome", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "categorical_vs_target.png", dpi=300)
    plt.close(fig)

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Heart Disease: Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # 6. IQR Outlier Analysis
    outlier_summary = {}
    for col in num_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(float(len(outliers) / len(df) * 100), 2)
        }

    # 7. Summary metrics
    results = {
        "dataset_shape": df.shape,
        "target_counts": target_counts.to_dict(),
        "outlier_summary": outlier_summary,
        "top_correlations_with_target": corr["Heart Disease"].sort_values(ascending=False).to_dict()
    }
    return results


if __name__ == "__main__":
    from ml.member1.config import get_disease_paths
    paths = get_disease_paths("heart")
    res = run_heart_eda(paths["data"] / "cleaned.csv", paths["reports"] / "plots")
    print("[Heart EDA] Generated all plots successfully.")
    print("[Heart EDA] Outlier Summary:\n", res["outlier_summary"])
    print("[Heart EDA] Top correlations with Heart Disease:\n", res["top_correlations_with_target"])
