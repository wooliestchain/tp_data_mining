import pandas as pd


fromage=pd.read_table("fromage1.txt",sep="\t",header=0,index_col=0)
import numpy as np
from sklearn import cluster
from sklearn import metrics
#utilisation de la métrique "silhouette"
# Crée un tableau NumPy de taille 9 pour stocker les scores de silhouette pour différentes valeurs de k
res = np.arange(9, dtype ="double")
for k in np.arange(9):
 #Crée un objet KMeans avec le nombre de clusters spécifié par k+2
 km = cluster.KMeans(n_clusters = k+2, n_init=10)
 #Applique l'algorithme KMeans aux données contenues dans le DataFrame fromage.
 km.fit(fromage)
 #Calcule le score de silhouette pour les clusters obtenus avec l'algorithme KMeans et stocke le résultat dans le tableau res.
 res[k] = metrics.silhouette_score(fromage,km.labels_)
 #Affiche les scores de silhouette pour chaque nombre de clusters.
print (res)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Assume that 'fromage' is your DataFrame

# Standardiser les données si nécessaire (utile pour la CAH)
from sklearn.preprocessing import StandardScaler
fromage_standardized = StandardScaler().fit_transform(fromage)

# Appliquer l'ACP pour réduire la dimension des données à 2 composantes principales
acp = PCA(n_components=2).fit_transform(fromage_standardized)

# Appliquer la CAH avec AgglomerativeClustering
cah = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
labels_cah = cah.fit_predict(fromage_standardized)

# Tracer le nuage de points pour la CAH
for couleur, k in zip(['red', 'blue', 'lawngreen', 'aqua'], range(4)):
    plt.scatter(acp[labels_cah == k, 0], acp[labels_cah == k, 1], c=couleur, label=f'Cluster {k}')

plt.title("CAH avec AgglomerativeClustering")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.legend()
plt.show()
