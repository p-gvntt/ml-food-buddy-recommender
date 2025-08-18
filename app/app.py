import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

# --- Project root ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Relative paths ---
DATA_PATH = project_root / "data/preprocessed/final_recipes.csv.gzip"
VECTORIZER_PATH = project_root / "models/tfidf_vectorizer.pkl"
VECTORS_PATH = project_root / "models/recipe_vectors.npy"

# --- Check required files ---
@st.cache_data
def check_files():
    missing_files = []
    for path, desc in [
        (DATA_PATH, "preprocessed recipes CSV"),
        (VECTORIZER_PATH, "TF-IDF vectorizer"),
        (VECTORS_PATH, "recipe vectors")
    ]:
        if not path.exists():
            missing_files.append(desc)
    return missing_files

missing_files = check_files()
if missing_files:
    st.error(
        "❌ The following files are missing:\n" +
        "\n".join(f"- {f}" for f in missing_files) +
        "\n\nPlease run the preprocessing and vectorization scripts in this order:\n"
        "1️⃣ `src.preprocessing.preprocess_and_save()` to generate preprocessed recipes\n"
        "2️⃣ `src.vectorizer.train_and_save_vectorizer()` to generate vectorizer and vectors\n"
        "After that, restart this app."
    )
    st.stop()

# --- Load artifacts ---
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, compression='gzip')

@st.cache_resource
def load_models():
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    vectors = np.load(VECTORS_PATH)
    return vectorizer, vectors

df = load_data()
vectorizer, vectors = load_models()

@st.cache_resource
def get_recommender():
    from src.recommender import recommend
    return recommend

recommend_func = get_recommender()

# --- Session State Initialization ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_query_params' not in st.session_state:
    st.session_state.last_query_params = None

# --- Main App ---
st.title("🍳 Food Buddy Recommender")
st.write(f"✅ Loaded {len(df)} recipes!")

# Use form to prevent unnecessary reruns
with st.form("recommendation_form"):
    query = st.text_input("Type ingredients you like (e.g. 'italian pasta')")
    
    col1, col2 = st.columns(2)
    with col1:
        time_filter = st.selectbox(
            "Time preference", 
            ["any", "fast", "medium", "long"]
        )
    with col2:
        calorie_filter = st.selectbox(
            "Calorie preference", 
            ["any", "low", "medium", "high"]
        )
    
    num_recs = st.slider(
        "Number of recommendations", 
        min_value=1, max_value=5, 
        value=3, step=1
    )
    
    submitted = st.form_submit_button("Get Recommendations")

# --- Process form submission ---
if submitted and query:
    time_filter_val = None if time_filter == "any" else time_filter
    calorie_filter_val = None if calorie_filter == "any" else calorie_filter

    with st.spinner(f"Finding recipes for '{query}'..."):
        recs = recommend_func(
            user_prefs=query,
            dataset=df,
            recipe_vectors_matrix=vectors,
            vectorizer=vectorizer,
            time_pref=time_filter_val,
            calorie_pref=calorie_filter_val,
            top_n=num_recs * 2,
            return_columns=[
                "Name", "Image_first", "TotalTime_str",
                "recipe_instructions_clean", "ingredients_clean", "Calories", "similarity"
            ]
        )

    # Deduplicate and limit to requested number
    st.session_state.results = recs.drop_duplicates(subset=["Name"]).head(num_recs)

elif submitted and not query:
    st.warning("Please enter some ingredients to get recommendations!")
    st.session_state.results = None

# --- Display Results ---
if st.session_state.results is not None and not st.session_state.results.empty:
    st.write("---")
    results = st.session_state.results
    
    for i, row in results.iterrows():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if pd.notna(row['Image_first']) and isinstance(row['Image_first'], str):
                try:
                    st.image(row['Image_first'], width=200)
                except:
                    st.write("🖼 Image unavailable")
            else:
                st.write("🖼 No image available")
        
        with col2:
            st.subheader(f"{i+1}. {row['Name']}")
            total_time = row['TotalTime_str'] if pd.notna(row['TotalTime_str']) else "N/A"
            calories = int(row['Calories']) if pd.notna(row['Calories']) else "N/A"
            similarity = f"{row['similarity']:.2f}" if pd.notna(row['similarity']) else "N/A"
            st.write(f"⏱ **{total_time}** | 🔥 **{calories} calories** | 🔍 **Match: {similarity}**")
            
            # Display ingredients
            ingredients = row['ingredients_clean'] if pd.notna(row['ingredients_clean']) else "No ingredients available."
            with st.expander("🥘 View Ingredients"):
                if isinstance(ingredients, str):
                    # If ingredients are stored as a string (possibly comma-separated or list-like)
                    if ingredients.startswith('[') and ingredients.endswith(']'):
                        # Handle string representation of list
                        try:
                            import ast
                            ingredient_list = ast.literal_eval(ingredients)
                            for ingredient in ingredient_list:
                                st.write(f"• {ingredient}")
                        except:
                            # Fallback to raw string
                            st.write(ingredients)
                    else:
                        # Handle comma-separated string
                        ingredient_list = [ing.strip() for ing in ingredients.split(',')]
                        for ingredient in ingredient_list:
                            st.write(f"• {ingredient}")
                else:
                    st.write(ingredients)
            
            instructions = row['recipe_instructions_clean'] if pd.notna(row['recipe_instructions_clean']) else "No instructions available."
            with st.expander("📜 View Instructions"):
                st.write(instructions)
        
        st.divider()