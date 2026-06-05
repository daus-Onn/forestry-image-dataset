import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional, premium visual style for the plot
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def main():
    csv_path = "forestry_dataset_summary.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find dataset summary file at '{csv_path}'")
        return
        
    print(f"Loading dataset summary from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # Map raw labels to clean display names
    class_map = {
        'bamboo_forest': 'Bamboo',
        'coniferous_pine_forest': 'Conifer',
        'mangrove_forest': 'Mangrove',
        'tropical_rainforest': 'Tropical'
    }
    
    split_map = {
        'train': 'Train',
        'val': 'Validation',
        'test': 'Test'
    }
    
    df['Class'] = df['Class_Label'].map(class_map)
    df['Split'] = df['Dataset_Type'].map(split_map)
    
    # Print value counts to verify the distribution
    print("\n--- Dataset Distribution Table ---")
    counts = df.groupby(['Class', 'Split']).size().unstack(fill_value=0)
    # Ensure correct split column order
    counts = counts[['Train', 'Validation', 'Test']]
    counts['Total'] = counts.sum(axis=1)
    
    # Print markdown table format for the report
    print("| Class | Train | Validation | Test | Total |")
    print("|---|---|---|---|---|")
    for cls in counts.index:
        train_val = counts.loc[cls, 'Train']
        val_val = counts.loc[cls, 'Validation']
        test_val = counts.loc[cls, 'Test']
        total_val = counts.loc[cls, 'Total']
        print(f"| **{cls}** | {train_val:,} | {val_val:,} | {test_val:,} | **{total_val:,}** |")
    
    total_row = counts.sum()
    print(f"| **Total** | **{total_row['Train']:,}** | **{total_row['Validation']:,}** | **{total_row['Test']:,}** | **{total_row['Total']:,}** |")
    
    # Plotting the distribution
    plt.figure(figsize=(11, 7))
    
    # Custom premium palette matching standard branding
    # Blue-grey for Train, Coral-Orange for Validation, Sage-Green for Test
    custom_palette = {
        'Train': '#2B6CB0',       # Darker slate blue
        'Validation': '#ED8936',  # Soft warm orange
        'Test': '#48BB78'         # Soft forest green
    }
    
    # Create grouped bar chart
    # We plot the data ordered by Class index
    plot_df = df.dropna(subset=['Class', 'Split'])
    
    ax = sns.countplot(
        data=plot_df,
        x='Class',
        hue='Split',
        hue_order=['Train', 'Validation', 'Test'],
        order=['Bamboo', 'Conifer', 'Mangrove', 'Tropical'],
        palette=custom_palette,
        edgecolor='black',
        linewidth=0.8
    )
    
    # Customize the appearance
    plt.title("Forestry Dataset Distribution by Class and Data Split", pad=20, fontweight='bold')
    plt.xlabel("Forest Class Type", labelpad=12, fontweight='semibold')
    plt.ylabel("Number of Images", labelpad=12, fontweight='semibold')
    
    # Add values on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=9,
                color='#2D3748',
                fontweight='semibold'
            )
            
    # Clean legend and grid
    plt.legend(title="Data Split", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    # Save the plot
    output_png = "dataset_distribution.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Distribution chart saved to '{output_png}'.")

if __name__ == '__main__':
    main()
