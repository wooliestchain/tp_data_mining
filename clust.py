import pandas as pd

#Chargmenr du ficher fromage.txt
#L'attribut 'sep' précise que les caractères dans le fichier sont séparés
#Le header précise que la première ligne contient le nom des colonnes
#La première colone du dataframe est utilisé comme index
fromage=pd.read_table("fromage1.txt",sep="\t",header=0,index_col=0)

#print(fromage.describe())


#pd.scatter_matrix(fromage, figsize=(9,9))

import numpy as np
from sklearn import cluster
#Fixe la graine du générateur de nombres aléatoires de NumPy à 0
np.random.seed(0)
#Crée un objet KMeans avec 4 clusters.
kmeans = cluster.KMeans(n_clusters = 4, n_init=10)
#Applique l'algorithme k-means aux données dans la variable "fromage"
kmeans.fit(fromage)
#Trie les indices des clusters dans l'ordre croissant
idk = np.argsort(kmeans.labels_)
# Crée un DataFrame pandas avec les indices triés selon les labels de k-means
print(pd.DataFrame(fromage.index[idk],kmeans.labels_[idk]))
#Affiche la distance de chaque point de données par rapport à chaque centroïde de cluster.
print(kmeans.transform(fromage))

# Obtenir les centres de chaque cluster
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=fromage.columns)

# Afficher les centres de chaque cluster
print(cluster_centers)

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
#graphique
import matplotlib.pyplot as plt
#Définit le titre du graphique comme "silhouette".
plt.title("silhouette")
#Définit le libellé de l'axe des x comme le nombre de clusters.
plt.xlabel("Number of clusters")
#Trace la courbe des scores de silhouette en fonction du nombre de clusters, en utilisant les valeurs de k
plt.plot(np.arange(2,11,1),(res))
#affiche
plt.show()


import pandas
import numpy as np
#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# Effectue la CAH en utilisant la méthode de liaison de Ward et la distance euclidienne comme métrique
Z = linkage(fromage,method='ward', metric='euclidean')
#affichage du dendrogramme
#Définit le titre du dendrogramme comme "CAH"
plt.title("CAH")
#Ajoute un titre supplémentaire au dendrogramme.
plt.title('CAH avec matérialisation des 4 classes')
#Affiche le dendrogramme avec des étiquettes sur l'axe des y et un seuil de couleur à 255.
dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=255)
plt.show()
#Attribue des groupes aux observations en fonction d'un seuil de distance dans le dendrogramme (255 dans ce cas) à l'aide de la fonction fcluster.
groupes_cah = fcluster(Z, t = 255,criterion='distance')
#Affiche les groupes attribués à chaque observation.
print(groupes_cah)
#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observations et leurs groupes
print(pandas.DataFrame(fromage.index[idg],groupes_cah[idg]))

#La Classification Ascendante Hiérarchique (CAH) est une technique d'analyse de regroupement qui organise les observations en groupes
# hiérarchiques basés sur leurs similarités. Dans notre cas
# , le code utilise la méthode de liaison de Ward et la distance euclidienne comme métrique pour regrouper les fromages.

import pandas as pd

# groupes_cah : Résultats de la CAH
groupes_cah = fcluster(Z, t = 255,criterion='distance')
# kmeans.labels_ : Résultats de KMeans
res[k] = metrics.silhouette_score(fromage, km.labels_)
# Créer une DataFrame avec les deux résultats
df = pd.DataFrame({'CAH': groupes_cah, 'KMeans': kmeans.labels_})

# Créer la table croisée
table_crosstab = pd.crosstab(df['CAH'], df['KMeans'])

# Afficher la table croisée
print(table_crosstab)


from sklearn.decomposition import PCA
# Applique l'ACP avec deux composantes principales sur les données du DataFrame fromage
acp = PCA(n_components =2).fit_transform(fromage)
#Parcourt une boucle avec des couleurs prédéfinies (rouge, bleu, vert, cyan)
# et les indices des clusters (0, 1, 2, 3) attribués par l'algorithme KMeans.
for couleur,k in zip(['red','blue','lawngreen', 'aqua'],[0,1,2,3]):
 #Trace un nuage de points où les points appartiennent au cluster k
 plt.scatter(acp[km.labels_==k,0],acp[km.labels_==k,1],c=couleur)
plt.show()

#Le code utilise l'Analyse en Composantes Principales (ACP) pour réduire la dimension des données à deux composantes principales,
# puis trace un nuage de points en fonction de ces deux composantes,
# avec des couleurs différentes pour chaque cluster attribué par l'algorithme KMeans.