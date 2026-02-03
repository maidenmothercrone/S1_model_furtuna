import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

#prelucrare fisier de intrare
t=pd.read_csv("data_in/MiscareaNaturala.csv", index_col=0) #de pus None la indice pt cerintele 1 si 2 de la a
t_clean = t.dropna()
t_clean = t_clean[~(t_clean == np.inf).any(axis=1)]
variabile_observate = list(t_clean)[1:]
x=t_clean[variabile_observate].values

#functii cerinte a1 si a2

def cerinta_1(t):
    rs_global_mean = t['RS'].mean()
    t_filtered = t[t['RS'] < rs_global_mean].copy()
    t_filtered = t_filtered.sort_values('RS', ascending=False)
    t_result = t_filtered[['Country_Number', 'Country_Name', 'RS']].copy()
    t_result.to_csv("data_out/Cerinta1.csv")

def cerinta_2(t): #maximul pt fiecara tara ca nu am continente
    indicators = ['RS','FR','LM','MMR','LE','LEM','LEF']
    row_data = {}
    for indicator in indicators:
        max_value =t[indicator].max()
        countries_with_max = t[t[indicator]==max_value]['Country_Name'].values
        country_codes = ','.join(countries_with_max)
        row_data[indicator] = country_codes
    t_result = pd.DataFrame([row_data])
    t_result=t_result[indicators]
    t_result.to_csv("data_out/Cerinta2.csv")


#cerinta b1
x = t_clean[variabile_observate].values
metoda_clusterizare="ward"
h = linkage(x, metoda_clusterizare)
print("Matricea ierarhiei")
print(h)

t_clean_h = pd.DataFrame(h, columns=['Cluster_1', 'Cluster_2', 'Distanta', 'Numar_Instante'])

t_clean_h.index = [f'Fuziune {i+1}' for i in range (len(h))]
t_clean_h.index.name = 'Pas_Junctiune'
t_clean_h.to_csv("data_out/h.csv")

def plot_ierarhie(h:np.ndarray, etichete=None, color_threshold=0, titlu="Plot ierarhie"):
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    dendrogram(h, labels=etichete, color_threshold=color_threshold, ax=ax)

plot_ierarhie(h, t_clean.index)
#np.savetxt("data_out/h.csv", h, delimiter=",", fmt="%.6f") #mai usor si nu am nevoie de dataframe ca sa fac csv ul corect dar incomplet, salveaza doar matricea
plt.show()

cerinta_1(t_clean)
cerinta_2(t_clean)
