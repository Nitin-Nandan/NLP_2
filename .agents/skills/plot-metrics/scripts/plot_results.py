import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/results_log.csv')
    
    # 1. Bar Chart: F1 Scores and Accuracy
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars='model_name', value_vars=['accuracy', 'f1_score'], var_name='Metric', value_name='Score')
    sns.barplot(x='model_name', y='Score', hue='Metric', data=df_melted)
    plt.title('Tokenization Models: Accuracy & F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/f1_acc_bar.png')
    plt.clf()
    
    # 2. Scatter Plot: Training Time (y) vs Parameters (x)
    plt.figure(figsize=(10, 6))
    
    # Corrected column name to match the tracking script architecture logic from Phase 2/3
    time_col = 'train_time' if 'train_time' in df.columns else 'train_time_seconds'
    
    sns.scatterplot(
        x='parameters', 
        y=time_col, 
        hue='model_name', 
        s=200, 
        data=df
    )
    plt.title('Training Time vs Total Model Parameters')
    plt.xlabel('Number of Parameters (Log Scale)')
    plt.ylabel('Training Time (Seconds)')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/time_params_scatter.png')
    plt.clf()

    print("Successfully generated and saved plots to results/ folder:")
    print(" - results/f1_acc_bar.png")
    print(" - results/time_params_scatter.png")
    inject_plots_into_report()



def inject_plots_into_report():
    """Rewrite the <!-- PLOTS_START/END --> block in docs/final_report.md with current images."""
    report_path = os.path.join('docs', 'final_report.md')
    if not os.path.exists(report_path):
        print(f"Skipping report injection: {report_path} not found.")
        return

    # Paths are relative from docs/ to results/
    block = (
        "<!-- PLOTS_START -->\n"
        "**Figure 1 — F1 Score & Accuracy by Tokenizer**\n\n"
        "![F1 and Accuracy Bar Chart](../results/f1_acc_bar.png)\n\n"
        "**Figure 2 — Training Time vs. Model Parameters**\n\n"
        "![Training Time vs Parameters Scatter Plot](../results/time_params_scatter.png)\n"
        "<!-- PLOTS_END -->"
    )

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    pattern = r'<!-- PLOTS_START -->.*?<!-- PLOTS_END -->'
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, block, content, flags=re.DOTALL)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f" - docs/final_report.md plots section updated.")
    else:
        print(f"Warning: PLOTS_START/END markers not found in {report_path}. Skipping injection.")


if __name__ == "__main__":
    main()

