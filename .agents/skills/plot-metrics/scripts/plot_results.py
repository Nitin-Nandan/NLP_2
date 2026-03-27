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

if __name__ == "__main__":
    main()
