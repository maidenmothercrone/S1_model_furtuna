import pandas as pd
from scipy.cluster.hierarchy import linkage


# A.1. Țările cu RS sub media globală

def cerinta1():
    miscare_df = pd.read_csv("data_in/MiscareaNaturala.csv", index_col=0)
    coduri_df = pd.read_csv("data_in/CoduriTariExtins.csv", index_col=0)

    #calcul medie globala a rs
    media_rs_global = miscare_df['RS'].mean()

    #filtrare tari cu rs sub medie
    tari_sub_medie = miscare_df[miscare_df['RS'] < media_rs_global].copy()

    #sortare descrescatoare dupa rs
    tari_sub_medie = tari_sub_medie.sort_values('RS', ascending=False)

    #selectare coloane cerute si redenumire output
    output = tari_sub_medie[['Three_Letter_Country_Code', 'Country_Name', 'RS']].copy()
    output.columns = ['Three_Letter_Country_Code', 'Country_Name', 'RS']

    #salvare fisier
    output.to_csv("data_out/Cerinta1.csv")
    print("Cerinta1.csv salvat cu succes!")
    return miscare_df, coduri_df

# A.2. Țările cu valori maxime pe continent
def cerinta2(miscare_df, coduri_df):
    #Unirea datelor pentru a avea continentul pentru fiecare tara
    df_complet = pd.merge(miscare_df, coduri_df, on='Three_Letter_Country_Code')

    #lista de indicatori pentru care se cauta maxim
    indicatori = ['FR','IM','LE','LEF','LEM','MMR','RS']

    #initializare data frame gol pt rezultate
    continente = df_complet['Continent_Name'].unique()
    rezultate = pd.DataFrame()
    rezultate['Continent_Name'] = continente

    for indicator in indicatori:
        coduri_continent = []

        for continent in continente:
            df_continent = df_complet[df_complet['Continent_Name']== continent]

            if indicator in ['IM', 'MMR']:
                # val minima
                min_val = df_continent[indicator].min()
                tara = df_continent[df_continent[indicator]==min_val].iloc[0]
            else:
                #val maxima
                max_val = df_continent[indicator].max()
                tara = df_continent[df_continent[indicator]==max_val].iloc[0]
            coduri_continent.append(tara['Three_Letter_Country_Code'])

        #adaugare la coloana de rezultate
        rezultate[indicator] = coduri_continent

    #salvare
    rezultate.to_csv("data_out/Cerinta2.csv", index=False)
    print("Cerinta2.csv salvat cu succes!")
    return df_complet
# B. Analiza de clusteri

def analiza_cluster(df_complet):
    #pregatire date clusterizare prin selectare tari cu toti indicatorii completati
    indicatori = ['FR','IM','LE','LEF','LEM','MMR','RS']
    df_clean = df_complet.dropna(subset=indicatori)

    #standardizare date (importanta pt ward)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean[indicatori])

    # B.1. Matricea ierarhie
    # Calcularea distanței euclidiene și linkage Ward
    Z = linkage(data_scaled, method='ward', metric='euclidean')

    #creare matrice ierarhie
    matrice_ierarhie=[]
    for i, (clust1, clust2, dist, size) in enumerate(Z,1):
        #conversie la indici orifinal pt clusterele cu < n_observatii
        n_obs = len(data_scaled)
        clust1_str = str(int(clust1)) if clust1 < n_obs else f"C{int(clust1 - n_obs + 1)}"
        clust2_str = str(int(clust2)) if clust2 < n_obs else f"C{int(clust2 - n_obs + 1)}"

        matrice_ierarhie.append({
            'Jonctiune': i,
            'Cluster 1': clust1_str,
            'Cluster 2': clust2_str,
            'Distanta': f"{dist:.4f}",
            'Dimensiune_Cluster': int(size)
        })

    #salvare matrice ierarhie
    pd.DataFrame(matrice_ierarhie).to_csv("data_out/h.csv", index=False)
    print("h.csv salvat cu succes!")
    return Z

def main():
    print("\nCerinta A.1.")
    miscare_df, coduri_df = cerinta1()

    print("\nCerinta A.2.")
    df_complet = cerinta2(miscare_df, coduri_df)


    print("\nCerinta B.1.")
    Z = analiza_cluster(df_complet)

if __name__ == "__main__":
    main()