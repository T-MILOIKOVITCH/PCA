# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:39:53 2024

@author: tmilo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:39:01 2024

@author: tmilo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:40:36 2024

@author: tmilo
"""
### Import librairy
import numpy as np
import pandas as pd
#from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Charger les données
#data = pd.read_csv(r"C:\Users\tmilo\Desktop\Aniket PCA\FL38_T&A.csv", sep=';', index_col=0)
data_total = pd.read_csv(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\PCA_15_38_57.csv", sep=';')
# Suppression de la première colonne
#data_total = data_total.drop(data_total.columns[0], axis=1)


# Séparer les données quantitatives et qualitatives
data_quant = data_total.iloc[:, :-1].replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0)
groups = data_total["Cluster"]  # Colonne qualitative

# Sélection de toutes les colonnes sauf la dernière
#df_sans_derniere = data_total.iloc[:, :-1]

data_qual = data_total['Cluster']
data = data_quant

# Remplacer les virgules par des points et convertir en types numériques
data_clean = data.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')

# Remplacer les valeurs manquantes par des zéros
data_clean = data_clean.fillna(0)

# Vérifiez les données nettoyées
print("Données nettoyées :")
print(data_clean.head())

DF = data_clean
###

### Standardize Data
scaler = StandardScaler() #Create scaler
data_scaled = scaler.fit_transform(DF) #Fit scaler
print(data_scaled)

### Print Standardized Data in Dataframe Format
DF_scaled = pd.DataFrame(data = data_scaled,columns = DF.columns)
DF_scaled.head(6) #Print first 6 rows of DF_scaled




### Ideal Number of Components
pca = PCA(n_components= 6)                  # Create PCA object forming 6 PCs
pca_trans = pca.fit_transform(DF_scaled)    # Transorm data
print(pca_trans)                            # Print transformed data
print(pca_trans.shape)                      # Print dimensions of transormed data

prop_var = pca.explained_variance_ratio_    # Extract proportion of explained variance
print(prop_var)                             # Print proportion of explained variance
sum_explained_variance = prop_var.sum()
print("Somme des pourcentages de variance expliquée:", sum_explained_variance)



PC_number = np.arange(pca.n_components_) + 1 # Enumarate component numbers
print(PC_number)                            # Print component numbers       

### Scree Plot
plt.figure(figsize=(10,6))                  # Set figure and size
plt.plot(PC_number,prop_var,'ro-')  
plt.title('Scree Plot (Elbow Method) PCA_15_38_57',fontsize = 15)
plt.xlabel('Component Number',fontsize = 15)
plt.ylabel('Proportion of Variance',fontsize = 15)   
plt.grid()
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Scree Plot (Elbow Method) PCA_15_38_57.png", dpi=500, bbox_inches='tight')                                  # Add grid lines                                         
plt.show()                                  # Print graph 
plt.clf()

### Extract explained variance
var = pca.explained_variance_               # Extract explained variance
print(var)                                  #Print explained variance

### Alternative Scree Plot
plt.figure(figsize=(10,6))                  # Set figure and size
plt.plot(PC_number,var,'ro-')  
plt.title('Scree Plot (Kaiser Rule) PCA_15_38_57',fontsize = 15)
plt.xlabel('Component Number',fontsize = 15)
plt.ylabel('Variance',fontsize = 15) 
plt.axhline(y=1,color = 'r',linestyle ='--')  
plt.grid()                                  # Add grid lines
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Scree Plot (Kaiser Rule) PCA_15_38_57.png", dpi=500, bbox_inches='tight')                                   
plt.show()          
plt.clf()

###### Variance exprimée par les 3 premières dimensions de l'ACP
# Calculer la variance expliquée cumulée pour les 3 premières composantes
variance_cumulee_3 = np.sum(prop_var[:3])
variance_cumulee_3 = variance_cumulee_3*100

variance_cumulee_12 = prop_var[0]+prop_var[1]
variance_cumulee_13 = prop_var[0]+prop_var[2]
variance_cumulee_23 = prop_var[1]+prop_var[2]


### Créer un graphique en barres pour toutes les composantes principales
plt.figure(figsize=(10, 6))  # Ajuste la taille en pouces
plt.bar(range(1, len(prop_var)+1), prop_var, alpha=0.5)
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Explained Variance')
plt.title('Percentage of variance Explained by component PCA_15_38_57')
# Ajouter une ligne verticale pour mettre en évidence les 3 premières composantes
plt.axvline(x=3, color='red', linestyle='--')
plt.text(3.2, variance_cumulee_3 * 0.002, f'{variance_cumulee_3:.2f}% of explained variance', color='red', ha='left', va='center', fontsize=12)
# Ajuster les marges
plt.subplots_adjust(right=0.8)
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Percentage of Variance explained by component PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()
plt.clf()

# Extraction des valeurs propres de l'acp sur 6
eigenvalues = pca.explained_variance_

# Pas besoin d'élever au carré les composantes (on utilise directement les corrélations)
contributions = pca.components_

# Création du dataframe pour la heatmap utilisant les corrélations
df_contributions = pd.DataFrame(contributions, columns=[f'CP{i+1}' for i in range(6)], index=data.columns)
features = list(DF_scaled.columns) # Feature/Varaible names

# Création de la heatmap
plt.figure(figsize=(12*1.5, 6*1.5))
sns.heatmap(df_contributions, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.xticks(range(len(features)), features, rotation=45, ha='right')
plt.yticks(range(len(pca.components_)), [f"PC{i+1}" for i in range(len(pca.components_))])
plt.xlabel("Variables",fontsize=20)
plt.xticks(rotation=0, ha='left', rotation_mode='anchor',fontsize=15)
plt.yticks(rotation=90, ha='right', rotation_mode='anchor',fontsize=15)
plt.ylabel('Principal Components',fontsize=20)
plt.title("Correlations between variables and principal components PCA_15_38_57",fontsize=20)
plt.margins(x=0.1, y=0.1)  # Augmente les marges de 10% dans chaque direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\heatmap_correlations_7_components PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()
plt.clf()

### Perform PCA forming 3 PCs
pca = PCA(n_components = 3)                 # Create PCA object forming 2 components
PC = pca.fit_transform(DF_scaled)           #Transform data
print(PC)                                   #Print transformed data
print(PC.shape)                             #Print dimensions of transformed data

### Biplot Data
PC1 = PC[:, 0]                              #Extract PC1
PC2 = PC[:, 1]                              #Extract PC2
PC3 = PC[:, 2]                              #Extract PC3

loadings = pca.components_                  # Extract Loadings
print(loadings)                             # Print loadings
print(loadings.shape)                       #Loadings size

scalePC1 = 1.0/(PC1.max() - PC1.min())      #Create min-max scale for PC1
print(scalePC1)                             # Print scalePC1
scalePC2 = 1.0/(PC2.max() - PC2.min())      # Create min-max scale for PC2
print(scalePC2)
scalePC3 = 1.0/(PC3.max() - PC3.min())      # Create min-max scale for PC2
print(scalePC3)                             # Print scalePC3
features = list(DF_scaled.columns) # Feature/Varaible names
print(features)

# Extraction des valeurs propres
eigenvalues = pca.explained_variance_

# Création du barplot
plt.figure(figsize=(8*1.5, 6*1.5))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.8, align='center')
plt.ylabel('Eigen values')
plt.xlabel('Principal Components')
plt.title('Bar plot of eigenvalues PCA_15_38_57')
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Bar plot of eigenvalues PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()
plt.clf()


# Créer la heatmap Variables / Composantes principales
plt.figure(figsize=(12, 8))
sns.heatmap(loadings, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
# Personnaliser les axes, le titre et la légende
plt.xticks(range(len(features)), features, rotation=45, ha='right')
plt.yticks(range(len(pca.components_)), [f"PC{i+1}" for i in range(len(pca.components_))])
plt.xlabel("Variables")
plt.ylabel('Principal Components')
plt.title("Variables' contributions to principal components PCA_15_38_57")
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\heatmap_upgraded PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()
plt.clf()

# Définir un facteur de réduction pour les flèches
scaling_factor = 0.8  # Ajustez ce facteur selon vos besoin

#### Simple Biplot Test with Customized Arrows
fig, ax = plt.subplots(figsize=(14,9))  #set figure and size
for i, feature in enumerate(features):
    ax.arrow(0,
             0,
             loadings[0,i],
             loadings[1,i],
             head_width = 0.01,
             head_length = 0.01,
             color="red",
             linewidth  = 1.5)
    ax.text(loadings[0,i]*1.1,              # Plot arrow texts
            loadings[1,i]*1.1,
            feature,
            color ="red",
            fontsize=15)   

    ax.scatter(PC1 * scalePC1,              # Plot data points
               PC2 * scalePC2)

    ax.set_xlabel('PC1',                    #Add annotations
                  fontsize = 20)
    ax.set_ylabel('PC2',
                  fontsize = 20)
    ax.set_title('Simple Test Biplot with customize arrows PCA_15_38_57',
                 fontsize = 20)
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.show()    
plt.clf()

#### Biplot with Customized Arrows
fig, ax = plt.subplots(figsize=(14*1.5,9*1.5))  #set figure and size
for i, feature in enumerate(features):
    ax.arrow(0,
             0,
             loadings[0,i],
             loadings[1,i],
             head_width = 0.01,
             head_length = 0.01,
             color="red",
             linewidth  = 1.5)
    ax.text(loadings[0,i]*1.1,              # Plot arrow texts
            loadings[1,i]*1.1,
            feature,
            color ="red",
            fontsize=15)   
    ax.scatter(PC1 * scalePC1,              # Plot data points
               PC2 * scalePC2,
               s=8)
for i, label in enumerate(DF.index):# Label Data Points
    ax.text(PC1[i] *scalePC1,
            PC2[i] *scalePC2,
            str(label),
            fontsize = 2)
    ax.set_xlabel('PC1',                    #Add annotations
                  fontsize = 20)
    ax.set_ylabel('PC2',
                  fontsize = 20)
    ax.set_title('Biplot customized Data Labels PCA_15_38_57',
                 fontsize = 20)
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Biplot customized Data Labels PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()   
plt.clf()


############################## Final Biplots for the 3 PC
##############################


### création des groupes
groups = data_total["Cluster"] #Extract data groups
print(groups) # Print groups
# Associer chaque groupe à une couleur spécifique
group_to_color = {
    'Location 2': 'green',
    'Location 3': 'black',
    'Location 1': 'yellow'}
# Convertir les groupes en couleurs en utilisant le dictionnaire de mapping
color_values = data_qual.replace(group_to_color)

###################### Biplot PC 12
fig, ax = plt.subplots(figsize=(14*2,9*2))  #set figure and size

for i, feature in enumerate(features):
    ax.arrow(0,
             0,
             loadings[0,i]*scaling_factor,
             loadings[1,i]*scaling_factor,
             head_width = 0.01,
             head_length = 0.01,
             color="red",
             linewidth  = 1.5)
    ax.text(loadings[0,i]*1.1*scaling_factor,              # Plot arrow texts
            loadings[1,i]*1.1*scaling_factor,
            feature,
            color ="black",
            bbox=dict(facecolor='lightsteelblue', edgecolor='black',
                   boxstyle='round4,pad=0.6',alpha=0.7),
            fontsize=15)    
scatter = ax.scatter(PC1 * scalePC1,              # Plot data points
               PC2 * scalePC2,
               c = color_values)
for i, label in enumerate(DF.index):# Label Data Points
    ax.text(PC1[i] *scalePC1,
            PC2[i] *scalePC2,
            str(label),
            fontsize = 2,
            color='blue')
    ax.set_xlabel('PC1',                    #Add annotations
                  fontsize = 20)
    ax.set_ylabel('PC2',
                  fontsize = 20)
    ax.set_title('Biplot customized Data Labels',
                 fontsize = 20)
ax.set_xlabel('PC1',fontsize = 20)
ax.set_ylabel('PC2',fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Définir la légende manuellement
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Green: FL15'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Black: FL38'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Yellow: FL57')]
ax.legend(handles=legend_elements, title="Groupes", loc='upper left',fontsize=20, title_fontsize=20)
ax.set_title("Biplot on PC1/PC2: individuals' point cloud colored by sample and variable vectors PCA_15_38_57",fontsize=30)
ax.set_xlabel(f'PC1 ({prop_var[0]*100:.1f}% of total variance)',fontsize=20)
ax.set_ylabel(f'PC2 ({prop_var[1]*100:.1f}% of total variance)',fontsize=20)
# Add margins to increase axis size
plt.text(0.95, 0.95, f'Total variance explained: {variance_cumulee_12*100:.1f}% of total variance', ha='right', va='top', transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.5),fontsize=20)
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Biplot12 PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()    
plt.clf()

###################### Biplot PC 13

fig, ax = plt.subplots(figsize=(14*2,9*2))  #set figure and size
for i, feature in enumerate(features):
    ax.arrow(0,
             0,
             loadings[0,i]*scaling_factor,
             loadings[2,i]*scaling_factor,
             head_width = 0.01,
             head_length = 0.01,
             color="red",
             linewidth  = 1.5)
    ax.text(loadings[0,i]*1.1*scaling_factor,              # Plot arrow texts
            loadings[2,i]*1.1*scaling_factor,
            feature,
            color ="black",
            bbox=dict(facecolor='lightsteelblue', edgecolor='black',
                   boxstyle='round4,pad=0.6',alpha=0.7),
            fontsize=15)    
scatter = ax.scatter(PC1 * scalePC1,              # Plot data points
               PC3 * scalePC3,
               c = color_values)
for i, label in enumerate(DF.index):# Label Data Points
    ax.text(PC1[i] *scalePC1,
            PC3[i] *scalePC3,
            str(label),
            fontsize = 2,
            color='blue')
    ax.set_xlabel('PC1',                    #Add annotations
                  fontsize = 20)
    ax.set_ylabel('PC3',
                  fontsize = 20)
    ax.set_title('Biplot customized Data Labels',
                 fontsize = 20)
ax.set_xlabel('PC1',fontsize = 20)
ax.set_ylabel('PC3',fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Définir la légende manuellement
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Green: FL15'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Black: FL38'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Yellow: FL57')]
ax.legend(handles=legend_elements, title="Groupes", loc='lower right',fontsize=20, title_fontsize=20)
ax.set_title("Biplot on PC1/PC3: individuals' point cloud colored by sample and variable vectors PCA_15_38_57",fontsize=30)
ax.set_xlabel(f'PC1 ({prop_var[0]*100:.1f}% of total variance)',fontsize=20)
ax.set_ylabel(f'PC3 ({prop_var[2]*100:.1f}% of total variance)',fontsize=20)
plt.text(0.95, 0.95, f'Total variance explained: {variance_cumulee_13*100:.1f}% of total variance', ha='right', va='top', transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.5),fontsize=20)
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Biplot13 PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()    
plt.clf()


###################### Biplot PC 23
fig, ax = plt.subplots(figsize=(14*2,9*2))  #set figure and size
for i, feature in enumerate(features):
    ax.arrow(0,
             0,
             loadings[1,i]*scaling_factor,
             loadings[2,i]*scaling_factor,
             head_width = 0.01,
             head_length = 0.01,
             color="red",
             linewidth  = 1.5)
    ax.text(loadings[1,i]*1.1*scaling_factor,              # Plot arrow texts
            loadings[2,i]*1.1*scaling_factor,
            feature,
            color ="black",
            bbox=dict(facecolor='lightsteelblue', edgecolor='black',
                   boxstyle='round4,pad=0.6',alpha=0.7),
            fontsize=15)   
scatter = ax.scatter(PC2 * scalePC2,              # Plot data points
               PC3 * scalePC3,
               c = color_values)
for i, label in enumerate(DF.index):# Label Data Points
    ax.text(PC2[i] *scalePC2,
            PC3[i] *scalePC3,
            str(label),
            fontsize = 2,
            color='blue')
    ax.set_xlabel('PC2',                    #Add annotations
                  fontsize = 20)
    ax.set_ylabel('PC3',
                  fontsize = 20)
    ax.set_title('Biplot customized Data Labels',
                 fontsize = 20)
ax.set_xlabel('PC2', fontsize = 40)
ax.set_ylabel('PC3', fontsize = 40)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Définir la légende manuellement
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Green: FL15'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Black: FL38'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Yellow: FL57')]
ax.legend(handles=legend_elements, title="Groupes", loc='upper left',fontsize=20, title_fontsize=20)
ax.set_title("Biplot on PC2/PC3: individuals' point cloud colored by sample and variable vectors PCA_15_38_57",fontsize=30)
ax.set_xlabel(f'PC2 ({prop_var[1]*100:.1f}% of total variance)',fontsize=20)
ax.set_ylabel(f'PC3 ({prop_var[2]*100:.1f}% of total variance)',fontsize=20)
plt.text(0.95, 0.95, f'Total variance explained: {variance_cumulee_23*100:.1f}% of total variance', ha='right', va='top', transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.5),fontsize=20)
plt.margins(x=0.1, y=0.1)  # Increase margins by 10% in each direction
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Biplot23 PCA_15_38_57.png", dpi=500, bbox_inches='tight')
plt.show()    
plt.clf()

################ Graph 3D
# Facteur d’échelle pour allonger les vecteurs
scale_factor = 8  # Ajustez ce facteur selon la longueur souhaitée des vecteurs

# Création d'un graphique 3D avec les données projetées dans l'espace des trois premières composantes
fig = go.Figure(data=[go.Scatter3d(
    x=PC[:, 0],  # PC1
    y=PC[:, 1],  # PC2
    z=PC[:, 2],  # PC3
    mode='markers',
    marker=dict(
        size=5,
        color=color_values,  # Couleurs basées sur les groupes
        opacity=0.8
    ),
#    text=groups.index  
    text=groups # Texte des groupes pour chaque point
)])

# Ajouter les vecteurs des variables
for i in range(loadings.shape[1]):  # Itérer sur les colonnes (variables)
    fig.add_trace(go.Scatter3d(
        x=[0, scale_factor *loadings[0, i]],
        y=[0, scale_factor *loadings[1, i]],
        z=[0, scale_factor *loadings[2, i]],
        mode='lines+text',
        line=dict(color='red',width=2),
        text=features[i]
        
    ))
    
    # Ajout de la tête de la flèche
    fig.add_trace(go.Cone(
        x=[scale_factor * loadings[0, i]],  # Position de la tête de la flèche
        y=[scale_factor * loadings[1, i]],
        z=[scale_factor * loadings[2, i]],
        u=[loadings[0, i]],  # Direction de la flèche
        v=[loadings[1, i]],
        w=[loadings[2, i]],
        sizemode="absolute",
        sizeref=0.3,  # Taille de la tête de flèche, ajustez si nécessaire
        anchor="tip",
        colorscale=[[0, 'red'], [1, 'red']]
    ))

# Configuration des axes et du titre
fig.update_layout(
    scene=dict(
        xaxis_title=f'PC1 ({prop_var[0]*100:.1f}% variance)',
        yaxis_title=f'PC2 ({prop_var[1]*100:.1f}% variance)',
        zaxis_title=f'PC3 ({prop_var[2]*100:.1f}% variance)'
    ),
    title="3D biplot of the first three principal components of PCA with variable vectors PCA_15_38_57",
    width=800,
    height=600
)

# Afficher le graphique
fig.show(renderer="browser")

############
############
############ HCPC

# Convertir les composantes principales en DataFrame pour utilisation dans la HCPC
df_pca = pd.DataFrame({'PC1': PC1, 'PC2': PC2, 'PC3': PC3})
df_pca_color = pd.merge(data, data_qual, left_index=True, right_index=True)
### Étape 2 : Réaliser la classification hiérarchique sur les composantes principales
# Utilisation de la méthode linkage pour la matrice de liens :
#très précisement on crée une matrice de liaison (ou matrice de distance),
#Cette matrice contient les distances entre chaque paire d’échantillons.
#Les distances sont calculées généralement à l’aide de la distance euclidienne,
#bien que d’autres types de distances puissent être utilisés, selon le contexte.
#Une fois cette matrice crée on applique une méthode algorithmique pour créer
#les cluster, ici on choisi la ward's method
Z = linkage(df_pca, method='ward')  # Utilisation de 'ward' pour la classification hiérarchique

# Appliquer le thème Seaborn pour un fond blanc
sns.set_theme(style="white")  # Choix du fond blanc


n_clusters = 5  # Par exemple, choisissons 5 clusters
# Calculer le seuil de coupe correspondant à ce nombre de clusters
# fcluster permet de récupérer la distance de coupe pour le nombre de clusters voulu
max_d = max(Z[:, 2])  # La distance maximale dans la matrice de liaison
threshold = max_d * (n_clusters - 1) / n_clusters  # Calcul d'un seuil approximatif


# Affichage du dendrogramme pour visualiser les clusters
plt.figure(figsize=(15*3, 7*3))
dendrogram(Z, color_threshold=threshold, leaf_font_size=0.2)
plt.title("Hierarchical Classification dendogram")
# Afficher seulement certaines étiquettes des échantillons (par exemple, chaque 50ème échantillon)
#plt.xticks(range(0, len(PC), 50), rotation=90)
plt.xlabel("Samples")
plt.ylabel("Distance")
# Ajuster la position des étiquettes
plt.xticks(rotation=90)  # Rotation des étiquettes pour éviter la superposition
plt.savefig(r"C:\Users\tmilo\Desktop\Aniket PCA\PCA_15_38_57\Dendogramme de la classification hierrarchique.png", dpi=800, bbox_inches='tight')
plt.show()
plt.clf()