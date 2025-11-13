"""
Evaluate and visualize sentiment classification experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns # Added for better visualization of complex results

# Load results
metrics_path = "results/metrics.csv"
plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(metrics_path):
    # This will still be called by run_all.py, so keep the check
    pass 
    
# Read results
try:
    df = pd.read_csv(metrics_path)
except FileNotFoundError:
    print("metrics.csv not found. Please run experiments first.")
    exit()

print("\n=== Experiment Results ===")
print(df)

# --- 1. Text Summary ---
best_row = df.loc[df['accuracy'].idxmax()]
summary = (
    f"Best Model Configuration:\n"
    f"  Architecture: {best_row['architecture']}\n"
    f"  Optimizer: {best_row['optimizer']}\n"
    f"  Sequence Length: {best_row['length']}\n"
    f"  Stability Strategy: {best_row['clip_strategy']}\n"
    f"Highest Validation Accuracy: {best_row['accuracy']:.4f}\n\n"
    f"Full Results:\n{df.to_string(index=False)}\n"
)
with open("results/analysis.txt", "w") as f:
    f.write(summary)


# --- 2. Generate Visualizations ---

# Plot 1: Accuracy vs Sequence Length, Grouped by Architecture
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="length", y="accuracy", hue="architecture", style="optimizer", marker="o")
plt.title("Accuracy vs. Sequence Length by Architecture and Optimizer")
plt.xlabel("Sequence Length")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.legend(title='Experiment Config', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "accuracy_vs_length_full.png"))
plt.close()

# Plot 2: Accuracy by Optimizer and Clipping (Sequence Lengths Combined)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="optimizer", y="accuracy", hue="clip_strategy")
plt.title("Accuracy Distribution by Optimizer and Stability Strategy")
plt.xlabel("Optimizer")
plt.ylabel("Validation Accuracy")
plt.grid(axis='y')
plt.legend(title='Stability Strategy', loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "accuracy_by_optimizer_clipping.png"))
plt.close()


print("\n✓ Visualization generated successfully.")
print("Plots and summary saved in 'results/plots' and 'results/analysis.txt'")