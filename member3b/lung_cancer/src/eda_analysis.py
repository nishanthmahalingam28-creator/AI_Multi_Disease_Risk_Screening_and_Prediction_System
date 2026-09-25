"""
Exploratory Data Analysis (EDA) Script for Lung Cancer Survey Dataset.

Generates reproducible exploratory visualizations for Member 3B:
1. target_distribution.png: Class imbalance and distribution of LUNG_CANCER.
2. feature_distribution.png: Age distribution, gender breakdown, and symptom prevalence.
3. feature_target_relationships.png: Symptom-specific lung cancer rates and age comparison.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Path definitions
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "dataset" / "survey_lung_cancer.csv"
REPORTS_DIR = REPO_ROOT / "member3b" / "lung_cancer" / "reports"


def load_dataset() -> pd.DataFrame:
    """Loads the raw lung cancer dataset without modifying source file."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates target_distribution.png illustrating class imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"YES": "#e11d48", "NO": "#0284c7"}

    # 1. Bar Chart
    target_counts = df["LUNG_CANCER"].value_counts()
    total = len(df)
    bars = ax1.bar(target_counts.index, target_counts.values, color=[palette[k] for k in target_counts.index], width=0.5, edgecolor="#0f172a", linewidth=1)
    ax1.set_title("Lung Cancer Target Class Counts", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Target (LUNG_CANCER)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Number of Patients", fontsize=11, fontweight="semibold")
    ax1.set_ylim(0, max(target_counts.values) * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Donut Chart
    wedges, texts, autotexts = ax2.pie(
        target_counts.values,
        labels=target_counts.index,
        colors=[palette[k] for k in target_counts.index],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        explode=(0.05, 0),
        wedgeprops=dict(width=0.45, edgecolor="#0f172a", linewidth=1),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Class Imbalance Ratio (6.92 : 1)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path = output_dir / "target_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated: {output_path}")


def plot_feature_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates feature_distribution.png covering demographic and symptom distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Age Distribution Histogram + KDE
    sns.histplot(df["AGE"], bins=15, kde=True, ax=axes[0, 0], color="#0d9488", edgecolor="#0f172a")
    axes[0, 0].axvline(df["AGE"].median(), color="#e11d48", linestyle="--", linewidth=2, label=f"Median ({df['AGE'].median():.0f} yrs)")
    axes[0, 0].axvline(df["AGE"].mean(), color="#d97706", linestyle=":", linewidth=2, label=f"Mean ({df['AGE'].mean():.1f} yrs)")
    axes[0, 0].set_title("Age Distribution of Survey Cohort", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Age (Years)", fontsize=10)
    axes[0, 0].set_ylabel("Count", fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # 2. Gender Breakdown
    gender_counts = df["GENDER"].value_counts()
    bars = axes[0, 1].bar(gender_counts.index, gender_counts.values, color=["#3b82f6", "#ec4899"], width=0.5, edgecolor="#0f172a")
    axes[0, 1].set_title("Gender Breakdown", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Gender", fontsize=10)
    axes[0, 1].set_ylabel("Count", fontsize=10)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Symptom Prevalence (Prevalence of value = 1)
    binary_cols = [c for c in df.columns if c not in ["GENDER", "AGE", "LUNG_CANCER"]]
    prevalence = {c.strip(): (df[c] == 1).mean() * 100 for c in binary_cols}
    prev_series = pd.Series(prevalence).sort_values(ascending=True)

    axes[1, 0].barh(prev_series.index, prev_series.values, color="#6366f1", edgecolor="#0f172a")
    axes[1, 0].set_title("Symptom / Habit Prevalence (Presence = 1)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Prevalence Percentage (%)", fontsize=10)
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(prev_series.values):
        axes[1, 0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8, fontweight="semibold")

    # 4. Age Boxplot by Gender
    sns.boxplot(x="GENDER", y="AGE", data=df, ax=axes[1, 1], palette={"MALE": "#bfdbfe", "FEMALE": "#fbcfe8"})
    axes[1, 1].set_title("Age Distribution by Gender", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Gender", fontsize=10)
    axes[1, 1].set_ylabel("Age (Years)", fontsize=10)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = output_dir / "feature_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated: {output_path}")


def plot_feature_target_relationships(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates feature_target_relationships.png comparing cancer positive rate by risk factor."""
    binary_cols = [c for c in df.columns if c not in ["GENDER", "AGE", "LUNG_CANCER"]]

    rates = []
    for col in binary_cols:
        col_clean = col.strip()
        rate_present = (df[df[col] == 1]["LUNG_CANCER"] == "YES").mean() * 100
        rate_absent = (df[df[col] == 0]["LUNG_CANCER"] == "YES").mean() * 100
        diff = rate_present - rate_absent
        rates.append({
            "feature": col_clean,
            "present_rate": rate_present,
            "absent_rate": rate_absent,
            "difference": diff,
        })

    rate_df = pd.DataFrame(rates).sort_values(by="difference", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2, 1]})

    # Grouped horizontal bar chart
    y = np.arange(len(rate_df))
    height = 0.35

    ax1.barh(y + height / 2, rate_df["present_rate"], height, label="Present (1)", color="#e11d48", edgecolor="#0f172a")
    ax1.barh(y - height / 2, rate_df["absent_rate"], height, label="Absent (0)", color="#94a3b8", edgecolor="#0f172a")

    ax1.set_yticks(y)
    ax1.set_yticklabels(rate_df["feature"], fontsize=10)
    ax1.set_xlabel("Lung Cancer Positive Rate (%) [P(LUNG_CANCER = YES)]", fontsize=11, fontweight="semibold")
    ax1.set_title("Cancer Positive Rate: Feature Present vs Absent", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(0, 115)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for i, row in enumerate(rate_df.itertuples()):
        ax1.text(row.present_rate + 1.5, i + height / 2, f"{row.present_rate:.1f}%", va="center", fontsize=8, fontweight="bold", color="#e11d48")
        ax1.text(row.absent_rate + 1.5, i - height / 2, f"{row.absent_rate:.1f}%", va="center", fontsize=8, color="#475569")

    # Age comparison by Target
    sns.boxplot(x="LUNG_CANCER", y="AGE", data=df, ax=ax2, palette={"YES": "#fecdd3", "NO": "#bae6fd"})
    ax2.set_title("Age Distribution by Target", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Lung Cancer Target", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Age (Years)", fontsize=11, fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = output_dir / "feature_target_relationships.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated: {output_path}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    plot_target_distribution(df, REPORTS_DIR)
    plot_feature_distribution(df, REPORTS_DIR)
    plot_feature_target_relationships(df, REPORTS_DIR)
    print("All EDA visualizations successfully generated.")


if __name__ == "__main__":
    main()
