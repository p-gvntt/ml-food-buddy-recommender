from fastapi import FastAPI, Query
from pathlib import Path
import pickle, numpy as np, pandas as pd
from src.recommender import recommend

DATA_PATH = Path("data/preprocessed/final_recipes.csv.gzip")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
VECTORS_PATH = Path("models/recipe_vectors.npy")

# Load once
df = pd.read_csv(DATA_PATH, compression='gzip')
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
vectors = np.load(VECTORS_PATH)

app = FastAPI()

@app.get("/recommend")
def get_recommendations(
    query: str, 
    time_pref: str = None, 
    calorie_pref: str = None,
    top_n: int = 3
):
    results = recommend(
        user_prefs=query,
        dataset=df,
        recipe_vectors_matrix=vectors,
        vectorizer=vectorizer,
        time_pref=time_pref,
        calorie_pref=calorie_pref,
        top_n=top_n,
        return_columns=[
            "Name", "Image_first", "TotalTime_str",
            "recipe_instructions_clean", "ingredients_clean", "Calories", "similarity"
        ]
    )
    
    # Drop duplicates, limit to top_n, and replace NaN/inf with None
    results_clean = (
        results.drop_duplicates(subset=["Name"])
               .head(top_n)
               .replace({np.nan: None, np.inf: None, -np.inf: None})
    )

    # Ensure numeric columns are JSON-safe
    for col in ["Calories", "similarity"]:
        if col in results_clean.columns:
            results_clean[col] = results_clean[col].apply(lambda x: float(x) if x is not None else None)

    return results_clean.to_dict(orient="records")
