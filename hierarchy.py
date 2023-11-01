import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Assume that 'fromage' is your DataFrame
fromage=pd.read_table("fromage1.txt",sep="\t",header=0,index_col=0)

# Standardiser les données si nécessaire (utile pour KMeans)
from sklearn.preprocessing import StandardScaler
fromage_standardized = StandardScaler().fit_transform(fromage)

# Appliquer l'ACP pour réduire la dimension des données à 2 composantes principales
acp = PCA(n_components=2).fit_transform(fromage_standardized)

# Initialiser le nombre de clusters à 1 (un seul cluster pour commencer)
num_clusters = 1

# Stocker les indices des observations dans chaque cluster
clusters = {0: np.arange(len(fromage))}

# Continuer jusqu'à ce que le nombre de clusters atteigne le nombre d'observations
while num_clusters < len(fromage):
    # Sélectionner le cluster à diviser (le dernier ajouté)
    current_cluster = max(clusters.keys())
    observations_in_cluster = fromage_standardized[clusters[current_cluster]]

    # Vérifier que le cluster a plus d'une observation avant de diviser
    if len(observations_in_cluster) > 1:
        # Appliquer KMeans sur le cluster sélectionné
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(observations_in_cluster)

        # Diviser les indices en deux clusters
        indices_cluster_1 = clusters[current_cluster][kmeans.labels_ == 0]
        indices_cluster_2 = clusters[current_cluster][kmeans.labels_ == 1]

        # Mettre à jour le dictionnaire des clusters
        del clusters[current_cluster]
        clusters[num_clusters] = indices_cluster_1
        clusters[num_clusters + 1] = indices_cluster_2

        # Mettre à jour le nombre de clusters
        num_clusters += 1
    else:
        # Si le cluster a une seule observation, ignorez-le et passez au suivant
        del clusters[current_cluster]

# Tracer le nuage de points pour l'algorithme hiérarchique descendant
for k, indices in clusters.items():
    plt.scatter(acp[indices, 0], acp[indices, 1], label=f'Cluster {k}')

plt.title("Algorithme Hiérarchique Descendant avec KMeans")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.legend()
plt.show()
