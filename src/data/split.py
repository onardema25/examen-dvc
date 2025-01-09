import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Chemins des fichiers
INPUT_FILE = "data/raw_data/raw.csv"  
OUTPUT_DIR = "data/processed"        

# Création du dossier de sortie si nécessaire
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        split data ready to be normalized (saved in../split).
    """    
    # Charger le dataset
    print("Chargement du dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Séparer les features (X) et la cible (y)
    print("Séparation des features et de la cible...")
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    
    # Split des données en ensemble d'entraînement et de test
    print("Division en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    
    # Sauvegarder les datasets dans le dossier de sortie
    print(f"Sauvegarde des fichiers dans {OUTPUT_DIR}...")
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    print("Split des données terminé avec succès !")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur rencontrée : {e}")
