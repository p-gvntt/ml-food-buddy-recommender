import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PREPROCESSED = os.path.join(BASE_DIR, "data", "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Files
RAW_DATA_FILE = os.path.join(DATA_RAW, "recipes.csv")
PREPROCESSED_PATH = os.path.join(DATA_PREPROCESSED, "final_recipes.csv.gzip")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
VECTORS_FILE = os.path.join(MODELS_DIR, "recipe_vectors.npy")