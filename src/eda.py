"""
Step 2: Exploratory Data Analysis.
Saves plots to reports/figures/. The same plots are also in notebooks/eda.ipynb

Run:  python src/eda.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # no window, just save files
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_clean.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print(df.shape)
    print(df.describe().T)

    # 1. class balance
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="target", data=df, hue="target", palette="Set2",
                       legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No disease (0)", "Disease (1)"])
    plt.title("Class balance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_balance.png"))
    plt.close()

    # 2. histograms of numeric columns
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df[numeric].hist(bins=20, figsize=(12, 7), color="steelblue",
                     edgecolor="black")
    plt.suptitle("Histograms of numeric features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "histograms.png"))
    plt.close()

    # 3. correlation heatmap
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
    plt.close()

    # 4. age vs max heart rate coloured by target
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="age", y="thalach", hue="target", data=df,
                    palette="Set1")
    plt.title("Age vs max heart rate (thalach)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "age_vs_thalach.png"))
    plt.close()

    # 5. chest pain type vs target
    plt.figure(figsize=(6, 4))
    sns.countplot(x="cp", hue="target", data=df, palette="Set2")
    plt.title("Chest pain type vs target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cp_vs_target.png"))
    plt.close()

    print("Plots saved in", FIG_DIR)


if __name__ == "__main__":
    main()
