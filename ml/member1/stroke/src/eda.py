"""
Stroke Exploratory Data Analysis (EDA) Module.

Generates reproducible statistical summaries, severe class imbalance profiling,
BMI missingness breakdown, and visualizations for the cleaned Stroke dataset.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_stroke_eda(data_path: Path, output_dir: Path) -> dict:
    """
    Executes comprehensive EDA on the cleaned Stroke dataset.

    Args:
        data_path (Path): Path to cleaned.csv.
        output_dir (Path): Directory where generated plots will be saved.

    Returns:
        dict: Dictionary containing computed statistical metrics and outlier counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#2ca02c", "#d62728"]  # Green (No stroke), Red (Stroke)

    # 1. Target Imbalance Distribution Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    target_counts = df["stroke"].value_counts().sort_index()
    target_labels = ["No Stroke (0)", "Stroke (1)"]
    bars = ax.bar(target_labels, target_counts.values, color=palette, width=0.45, edgecolor="black", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height:,}\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Stroke Severe Target Imbalance (N = 5,110)\nMinority Prevalence: 4.87% (19.52:1 Ratio)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, 5600)
    plt.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Continuous Numerical Feature Distributions
    num_cols = ["age", "avg_glucose_level", "bmi"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2ca02c", edgecolor="black")
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
    plt.suptitle("Stroke: Continuous Feature Distributions", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_distributions.png", dpi=300)
    plt.close(fig)

    # 3. Numerical Features Stratified by Stroke Status (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        sns.boxplot(x="stroke", y=col, data=df, ax=axes[i], palette=palette, hue="stroke", legend=False)
        axes[i].set_title(f"{col} by Stroke Status", fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(["No Stroke (0)", "Stroke (1)"])
        axes[i].set_xlabel("Stroke Outcome", fontsize=10)
        axes[i].set_ylabel(col, fontsize=10)
    plt.suptitle("Stroke: Numerical Risk Factors Stratified by Outcome", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_dir / "numerical_boxplots.png", dpi=300)
    plt.close(fig)

    # 4. BMI Missingness by Target Class Plot
    bmi_missing = df.groupby("stroke")["bmi"].apply(lambda s: s.isnull().mean() * 100)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["No Stroke (0)", "Stroke (1)"], bmi_missing.values, color=palette, width=0.45, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("BMI Missingness Rate by Stroke Status\n(16.06% in Stroke Patients vs 3.31% in Non-Stroke)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Missing BMI Percentage (%)", fontsize=11)
    ax.set_ylim(0, 20)
    plt.tight_layout()
    fig.savefig(output_dir / "bmi_missingness_by_stroke.png", dpi=300)
    plt.close(fig)

    # 5. Key Categorical Risk Factors vs Stroke
    cat_cols = ["hypertension", "heart_disease", "ever_married", "smoking_status"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        # Calculate stroke prevalence percentage per category
        rate_df = df.groupby(col)["stroke"].agg(["count", "mean"]).reset_index()
        rate_df["prevalence_pct"] = rate_df["mean"] * 100
        bars = axes[i].bar(rate_df[col].astype(str), rate_df["prevalence_pct"], color="#d62728", edgecolor="black", width=0.5)
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f"{height:.1f}%",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[i].set_title(f"Stroke Prevalence by {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Stroke Prevalence (%)", fontsize=10)
    plt.suptitle("Stroke: Empirical Risk Prevalence Across Clinical/Demographic Categories", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "categorical_vs_stroke.png", dpi=300)
    plt.close(fig)

    # 6. Correlation Heatmap (Numeric & Binary Predictors)
    corr_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Stroke: Numerical & Binary Predictor Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # 7. IQR Outlier Analysis
    outlier_summary = {}
    for col in num_cols:
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
        "outlier_summary": outlier_summary,
        "bmi_missing_by_stroke": bmi_missing.to_dict(),
        "top_correlations_with_stroke": corr["stroke"].sort_values(ascending=False).to_dict()
    }
    return results


if __name__ == "__main__":
    from ml.member1.config import get_disease_paths
    paths = get_disease_paths("stroke")
    res = run_stroke_eda(paths["data"] / "cleaned.csv", paths["reports"] / "plots")
    print("[Stroke EDA] Generated all plots successfully.")
    print("[Stroke EDA] BMI missing by stroke:\n", res["bmi_missing_by_stroke"])
    print("[Stroke EDA] Outlier Summary:\n", res["outlier_summary"])
    print("[Stroke EDA] Top correlations with stroke:\n", res["top_correlations_with_stroke"])
