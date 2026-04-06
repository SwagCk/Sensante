import pandas as pd 

#Charger les données

df = pd.read_csv("dat/patients_dakar.csv")

#Premier apercu du dataset
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

#Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")


#Apercu des 5 premières lignes
print("\n-----Aperçu des 5 premières lignes-----")
print(df.head())

#Statistiques de base
print(f"\n---Statistiques de descriptives---")
print(df.describe().round(2))

#Repartition des diagnostiques
print(f"\n---Répartition des diagnostics---")
diag_counts = df['diagnostic'].value_counts()
for diag , count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")

#Repartition par region
print(f"\n---Répartition par région(top 5)---")
region_counts = df['region'].value_counts().head(5)
for region, count in region_counts.items():
    pct = count / len(df) * 100
    print(f"{region:15s} : {count:3d} patients ({pct:.1f}%)")

#Temperature moyenne par diagnostic
print(f"\n---Température moyenne par diagnostic---")
temp_by_diag = df.groupby('diagnostic')['temperature'].mean().round(2)
for diag, temp in temp_by_diag.items():
    print(f"{diag:12s} : {temp:.1f} °C")

#Repartition par sexe et diagnostic
print(f"\n---Patients par sexe et diagnostic---")
print(df.groupby(['diagnostic', 'sexe']).size())

print(f"\n{'=' * 50}")
print("Exploration terminée.")
print(f"Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")