import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Chemins des fichiers
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
X_TRAIN_FILE = os.path.join(INPUT_DIR, "X_train.csv")
X_TEST_FILE = os.path.join(INPUT_DIR, "X_test.csv")

# Création du dossier de sortie si nécessaire
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Normalisation des données d'entraînement et de test,
       avec sauvegarde dans ../processed.
    """
    # Vérifier si les fichiers d'entrée existent
    if not os.path.exists(X_TRAIN_FILE) or not os.path.exists(X_TEST_FILE):
        raise FileNotFoundError("Les fichiers X_train.csv ou X_test.csv sont introuvables.")

    # Charger les fichiers d'entrée
    print("Chargement des données...")
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)

    # Identifier et supprimer les colonnes non numériques
    print("Suppression des colonnes non numériques...")
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    # Initialiser le scaler
    scaler = StandardScaler()

    # Ajuster le scaler sur les données d'entraînement et transformer
    print("Normalisation des données d'entraînement...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir les données normalisées en DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Sauvegarder les résultats dans des fichiers CSV
    print(f"Sauvegarde des fichiers normalisés dans {OUTPUT_DIR}...")
    X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_test_scaled.csv"), index=False)

    print("Normalisation des données terminée avec succès !")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur rencontrée : {e}")
