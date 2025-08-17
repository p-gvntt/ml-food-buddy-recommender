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
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.results = None
    st.session_state.query = ""
    st.session_state.num_recs = 3
    st.session_state.time_filter = "any"
    st.session_state.calorie_filter = "any"
    st.session_state.show_results = False

# --- Main App Layout ---
def main_app():
    st.title("🍳 ML Food Buddy Recommender")
    st.write(f"✅ Loaded {len(df)} recipes!")

    # Use form to prevent unnecessary reruns
    with st.form("recommendation_form"):
        query = st.text_input(
            "Type ingredients you like (e.g. 'italian pasta')", 
            value=st.session_state.query
        )
        
        col1, col2 = st.columns(2)
        with col1:
            time_filter = st.selectbox(
                "Time preference", 
                ["any", "fast", "medium", "long"],
                index=["any", "fast", "medium", "long"].index(st.session_state.time_filter)
            )
        with col2:
            calorie_filter = st.selectbox(
                "Calorie preference", 
                ["any", "low", "medium", "high"],
                index=["any", "low", "medium", "high"].index(st.session_state.calorie_filter)
            )
        
        num_recs = st.slider(
            "Number of recommendations", 
            min_value=1, max_value=5, 
            value=st.session_state.num_recs, step=1
        )
        
        submitted = st.form_submit_button("Get Recommendations")

    # --- Compute only if query or filters changed ---
    should_compute = (
        submitted and (
            st.session_state.query != query
            or st.session_state.num_recs != num_recs
            or st.session_state.time_filter != time_filter
            or st.session_state.calorie_filter != calorie_filter
        )
    )

    if should_compute:
        if query:
            st.session_state.query = query
            st.session_state.num_recs = num_recs
            st.session_state.time_filter = time_filter
            st.session_state.calorie_filter = calorie_filter

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
                        "recipe_instructions_clean", "Calories", "similarity"
                    ]
                )

            # Deduplicate and limit to requested number
            st.session_state.results = recs.drop_duplicates(subset=["Name"]).head(num_recs)
            st.session_state.show_results = True
        else:
            st.warning("Please enter some ingredients to get recommendations!")
            st.session_state.show_results = False

    # --- Display Results from session state ---
    if st.session_state.show_results and st.session_state.results is not None:
        results = st.session_state.results
        for i, row in results.iterrows():
            with st.container():
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
                    
                    instructions = row['recipe_instructions_clean'] if pd.notna(row['recipe_instructions_clean']) else "No instructions available."
                    with st.expander("📜 View Instructions"):
                        st.write(instructions)
                
                st.divider()

if __name__ == "__main__":
    main_app()
