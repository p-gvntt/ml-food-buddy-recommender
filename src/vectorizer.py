import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import PREPROCESSED_PATH, VECTORIZER_FILE, VECTORS_FILE
import os

def train_and_save_vectorizer():
    df = pd.read_csv(PREPROCESSED_PATH, compression="gzip")

    vectorizer = TfidfVectorizer(max_features=2000)
    recipe_vectors = vectorizer.fit_transform(df['combined_text'])
    recipe_vectors = recipe_vectors.toarray()
    recipe_vectors = recipe_vectors / (np.linalg.norm(recipe_vectors, axis=1, keepdims=True) + 1e-10)

    os.makedirs(os.path.dirname(VECTORIZER_FILE), exist_ok=True)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    np.save(VECTORS_FILE, recipe_vectors)

    print(f"✅ Vectorizer saved to {VECTORIZER_FILE}")
    print(f"✅ Recipe vectors saved to {VECTORS_FILE}")

if __name__ == "__main__":
    train_and_save_vectorizer()
