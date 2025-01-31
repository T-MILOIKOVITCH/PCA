# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:11:36 2025

@author: tmilo / Titouan Miloïkovitch 
titouan.miloikovitch@cyu.fr ; titouan.miloikovitch@vub.be
PhD Student in Geosciences et CY Cergy Paris Université (ISTEP GEC Lab)
and Vrije Universiteit Brussel (AMGC Lab)
"""

"""This script performs a Principal Component Analysis (PCA) on a geochemical
 dataset to reduce dimensionality and explore relationships between samples.
 The PCA is conducted in two stages: a first with all dimensions to examine
 the variance explained by each component (using the "elbow" and "Kaiser rule"
 methods), and a second after reducing the number of components based on these
 criteria. The results are visualized through various plots, including
 explained variance plots, heatmaps of variable contributions, and biplots
 to interpret the relationships between samples and variables in the reduced
 space. Sample groups are also visualized with distinct colors for comparative
 analysis."""
 
"""
# === IMPORTS === contains the libraries used for this code
# === FUNCTIONS === contains the logical and graphical functions
# === MAIN === runs the code
"""

# === IMPORTS ===
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# === FUNCTIONS ===
def load_data(filepath):
    """Load quantitative and qualitative data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The file {filepath} was not found. Please check the path.")
    
    data_total = pd.read_csv(filepath, sep=';')
    data_quant = (
        # Extract numerical data except the last column (qualitative)
        data_total.iloc[:, :-1]
        # Convert decimal separators if necessary
        .replace(",", ".", regex=True)
        # Ensure numerical format
        .apply(pd.to_numeric, errors="coerce")
        # Replace missing values with zero
        .fillna(0)
    )
    data_qual = data_total['Cluster']  # Extract the categorical column
    return data_quant, data_qual

def standardize_data(data):
    """Standardize data using StandardScaler (zero mean, unit variance)."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data=data_scaled, columns=data.columns)

def perform_pca(data, n_components):
    """Perform Principal Component Analysis (PCA)."""
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data

def plot_scree_plot(pca):
    """Generate a scree plot to visualize the explained variance ratio."""
    PC_number = np.arange(pca.n_components_) + 1
    prop_var = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.plot(PC_number, prop_var, 'ro-')
    plt.title('Scree Plot (Elbow Method) PCA_Loc123', fontsize=15)
    plt.xlabel('Component Number', fontsize=15)
    plt.ylabel('Proportion of Variance', fontsize=15)
    plt.grid()
    plt.show()
    plt.clf()

def plot_kaiser_rule(pca):
    """Generate a scree plot with the Kaiser rule (eigenvalues > 1)."""
    PC_number = np.arange(pca.n_components_) + 1
    eigenvalues = pca.explained_variance_
    
    plt.figure(figsize=(10, 6))
    plt.plot(PC_number, eigenvalues, 'ro-')
    plt.axhline(y=1, color='r', linestyle='--')  # Kaiser threshold
    plt.title('Scree Plot (Kaiser Rule) PCA_Loc123', fontsize=15)
    plt.xlabel('Component Number', fontsize=15)
    plt.ylabel('Eigenvalues', fontsize=15)
    plt.grid()
    plt.show()
    plt.clf()

def compute_variance(pca, n_components):
    """Calculate explained variance for a given number of components."""
    prop_var = pca.explained_variance_ratio_[:n_components]
    variance_cumulative = np.cumsum(prop_var)
    return prop_var, variance_cumulative
    
def plot_variance_bar(prop_var, variance_cumulative, n_components, title):
    """Create a bar plot for explained variance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_components + 1), prop_var*100, alpha=0.5)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Percentage of Explained Variance')
    ax.set_title(title)
    ax.axvline(x=3, color='red', linestyle='--')
    ax.text(3.2, variance_cumulative[2] * 0.002, 
            f'{variance_cumulative[2] * 100:.2f}% of explained variance', 
            color='red', ha='left', va='center', fontsize=12)
    plt.show()
    
def plot_heatmap(loadings, features):
    """Plot a heatmap showing the contributions of
    variables to principal components."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        loadings, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.yticks(
        range(len(loadings)), [f"PC{i+1}" for i in range(len(loadings))])
    plt.xlabel("Variables")
    plt.ylabel('Principal Components')
    plt.title("Variables' contributions to principal components PCA_Loc123")
    plt.show()
    plt.clf()

def plot_biplot(PC, PC_x, PC_y, loadings, features, data_qual, title, pca):
    """Generate a biplot to visualize PCA results 
    with vectors representing features."""
    var_x = pca.explained_variance_ratio_[PC_x] * 100
    var_y = pca.explained_variance_ratio_[PC_y] * 100
    
    plt.figure(figsize=(14, 9))

    # Clean the labels in data_qual to remove spaces and standardize the case
    data_qual = data_qual.str.strip().str.capitalize()

    # Check the unique labels in data_qual
    unique_labels = data_qual.unique()
    print("Unique labels:", unique_labels)  

    # Adjust colors_map to match the labels
    colors_map = {'Green': 'green',
                  'Black': 'black',
                  'Yellow': 'yellow'}
    
    # Prepare legend mapping
    legend_map = {'Green': 'Green: LOC2',
                  'Black': 'Black: LOC3',
                  'Yellow': 'Yellow: LOC1'}

    # Check if all the labels are correctly mapped
    if not all(label in colors_map for label in unique_labels):
        print("Warning: some labels are not in colors_map.")
        print("Unmapped labels:",
              [label for label in unique_labels if label not in colors_map])
    
    # Assign colors to labels
    colors = data_qual.map(colors_map).fillna('gray')  

    # Check the color mapping
    print("Mapped colors:", colors.unique())

    # Plot the points with the legend, without black borders
    for label in unique_labels:
        mask = data_qual == label
        plt.scatter(PC[mask, PC_x], PC[mask, PC_y],
                    label=legend_map.get(label, label),
                    c=colors[mask])  # Remove edgecolors='k'

    # Add the feature vectors
    for i, feature in enumerate(features):
        plt.arrow(0, 0, loadings[PC_x, i]*8, loadings[PC_y, i]*8,
                  head_width=0.02*8, head_length=0.02*8, color='red')
        plt.text(loadings[PC_x, i] * 1.1*8, loadings[PC_y, i] * 1.1*8,
                 feature, color='red', fontsize=12)
    
    # Labels and title
    plt.xlabel(f'PC{PC_x+1} ({var_x:.2f}% of total variance)')
    plt.ylabel(f'PC{PC_y+1} ({var_y:.2f}% of total variance)')
    plt.title(title)
    plt.grid()

    # Add the legend with formatted labels
    plt.legend(title="Groups", loc="best")

    plt.show()
    plt.clf()

def main():
    """Main function to execute the PCA analysis
    and generate visualizations."""
    filepath = r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_Loc123\PCA_Loc123.csv"
    data_quant, data_qual = load_data(filepath)
    DF_scaled = standardize_data(data_quant)
    
    pca, PC = perform_pca(DF_scaled, n_components=6)
    plot_scree_plot(pca)
    plot_kaiser_rule(pca)
    plot_variance_bar(*compute_variance(pca, 6), 6,
                      "PCA on 6 components / Percentage of Variance Explained")
    plot_heatmap(pca.components_, data_quant.columns)
    
    pca, PC = perform_pca(DF_scaled, n_components=3)
    plot_variance_bar(*compute_variance(pca, 3), 3,
                      "PCA on 3 components / Percentage of Variance Explained")
    plot_heatmap(pca.components_, data_quant.columns)
    plot_biplot(PC, 0, 1, pca.components_,
                data_quant.columns, data_qual, "Biplot on PC1/PC2", pca)
    plot_biplot(PC, 0, 2, pca.components_,
                data_quant.columns, data_qual, "Biplot on PC1/PC3", pca)
    plot_biplot(PC, 1, 2, pca.components_,
                data_quant.columns, data_qual, "Biplot on PC2/PC3", pca)

# === MAIN ===
if __name__ == "__main__":
    main()


