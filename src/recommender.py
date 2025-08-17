import os
import pickle
import numpy as np
import pandas as pd
from src.utils import clean_text, correct_query, apply_categorical_filters, format_time
from src.config import PREPROCESSED_PATH, VECTORIZER_FILE, VECTORS_FILE

def load_artifacts():
    df = pd.read_csv(PREPROCESSED_PATH, compression="gzip")
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    recipe_vectors = np.load(VECTORS_FILE)
    return df, vectorizer, recipe_vectors

def recommend(
    user_prefs,
    dataset,
    recipe_vectors_matrix,
    vectorizer,
    top_n=3,
    time_pref=None,
    calorie_pref=None,
    vocab_column='combined_text',
    time_column='time_bin',
    calorie_column='calorie_bin',
    return_columns=None
):
    """
    Recommend recipes based on user preferences using TF-IDF cosine similarity.

    This function cleans and autocorrects the user query, computes similarity
    with recipe vectors, applies optional filters, and returns the top N matching recipes.
    (Unchanged logic from your dev code.)
    """

    # Clean and correct query
    cleaned_query = clean_text(user_prefs)
    corrected_query = correct_query(query=cleaned_query, recipes=dataset, vocab_columns=[vocab_column])

    # Vectorize query
    user_vec = vectorizer.transform([corrected_query]).toarray()[0]
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-10)

    # Cosine similarity
    dataset = dataset.copy()
    dataset['similarity'] = recipe_vectors_matrix @ user_vec

    # If max similarity is 0, return a message
    if dataset['similarity'].max() == 0:
        return pd.DataFrame([{
            'message': 'No matching recipes found. Please add correct ingredients.'
        }])

    # Apply filters
    filters = {
        time_column: time_pref,
        calorie_column: calorie_pref
    }
    dataset = apply_categorical_filters(dataset, filters)

    # Sort by similarity and pick top_n
    results = dataset.sort_values(by='similarity', ascending=False).head(top_n).copy()

    # Format TotalTime
    if 'TotalTime_min' in results.columns:
        results['TotalTime_str'] = results['TotalTime_min'].apply(format_time)
    else:
        results['TotalTime_str'] = ""

    # Return requested columns
    if return_columns:
        valid_columns = [c for c in return_columns if c in results.columns]
        if not valid_columns:
            valid_columns = results.columns.tolist()
    else:
        valid_columns = results.columns.tolist()

    return results[valid_columns]

if __name__ == "__main__":
    # simple smoke test
    df, vectorizer, vectors = load_artifacts()
    top = recommend(
        user_prefs="Italiaan pastaa vegegtarian",
        dataset=df,
        recipe_vectors_matrix=vectors,
        vectorizer=vectorizer,
        top_n=5,
        time_pref="fast",
        calorie_pref="low",
        vocab_column="combined_text",
        time_column="time_bin",
        calorie_column="calorie_bin",
        return_columns=["Name", "Image_first", "recipe_instructions_clean", "TotalTime_str", "Calories"]
    )
    print(top)