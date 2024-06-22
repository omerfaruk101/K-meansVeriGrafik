import pandas as pd     #veri manipulasyon, analiz
from sklearn.cluster import KMeans      #kmeans aldigim kutuphane
from sklearn.preprocessing import StandardScaler    #veriyi standartlastirma
import matplotlib.pyplot as plt #grafik olarak cikti

# Verileri içeren Excel dosyasının yolu
excel_path = 'C:\\Users\\gs__s\\PycharmProjects\\yapayzeka\\yapayzekaveriler.xlsx'

laptop_df = pd.read_excel(excel_path) #excel dosyasını okuttugum yer

X = laptop_df[['Fiyat']] #kumeleme yapilacak ozellik

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

laptop_df['Kume'] = kmeans.labels_

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

print(laptop_df)

print("\nKüme Merkezleri:")
for i, center in enumerate(cluster_centers):
    print(f"Küme {i + 1} Merkezi: {center[0]:.2f} TL")

plt.scatter(laptop_df['Fiyat'], [0] * len(laptop_df), c=laptop_df['Kume'], cmap='viridis')
plt.scatter(cluster_centers, [0] * len(cluster_centers), marker='X', c='red', s=200)                #matplotlib
plt.xlabel('Fiyat')
plt.title('Laptop Fiyatlarına Göre K-means Kümeleme')
plt.show()