import os
import pandas as pd
from src.utils import (
    data_loader, parse_time, parse_r_list_column, clean_text,
    clip_top_outliers, time_bin, calorie_bin, get_first_url
)
from src.config import PREPROCESSED_PATH

def preprocess_and_save():
    # Load raw data (expects recipes.csv in data/raw/)
    recipes = data_loader("recipes.csv", source="raw")

    # Convert times to minutes
    recipes['PrepTime_min'] = recipes['PrepTime'].apply(parse_time)
    recipes['CookTime_min'] = recipes['CookTime'].apply(parse_time)
    recipes['TotalTime_min'] = recipes['TotalTime'].apply(parse_time)

    # Fill unparseable times with medians
    recipes['PrepTime_min'] = recipes['PrepTime_min'].fillna(recipes['PrepTime_min'].median())
    recipes['CookTime_min'] = recipes['CookTime_min'].fillna(recipes['CookTime_min'].median())
    recipes['TotalTime_min'] = recipes['PrepTime_min'] + recipes['CookTime_min']

    # Date parsing
    recipes['DatePublished'] = pd.to_datetime(recipes['DatePublished'], errors='coerce')
    if hasattr(recipes['DatePublished'].dt, "tz"):
        try:
            recipes['DatePublished'] = recipes['DatePublished'].dt.tz_convert(None)
        except Exception:
            pass

    # Drop rows where Description is NaN
    if 'Description' in recipes.columns:
        recipes = recipes.dropna(subset=['Description'])

    # Parse and clean string columns
    recipes['ingredients_clean'] = parse_r_list_column(recipes['RecipeIngredientParts'])
    recipes['category_clean'] = parse_r_list_column(recipes['RecipeCategory']) if 'RecipeCategory' in recipes.columns else ""
    recipes['keywords_clean'] = parse_r_list_column(recipes['Keywords']) if 'Keywords' in recipes.columns else ""
    recipes['recipe_instructions_clean'] = parse_r_list_column(
        recipes['RecipeInstructions'], extended_clean=True
    ) if 'RecipeInstructions' in recipes.columns else ""
    recipes['Name'] = parse_r_list_column(recipes['Name'], extended_clean=True) if 'Name' in recipes.columns else ""

    # Combined text
    recipes['combined_text'] = (
        recipes['ingredients_clean'] + ", " +
        recipes['category_clean'] + ", " +
        recipes['keywords_clean'] + ", " +
        recipes['Description'].fillna("")
    ).apply(clean_text)

    # Clip outliers
    numeric_cols_to_clip = [
        "TotalTime_min", "Calories", "FatContent", "SaturatedFatContent",
        "CholesterolContent", "SodiumContent", "CarbohydrateContent",
        "FiberContent", "SugarContent", "ProteinContent"
    ]
    recipes_clipped, _ = clip_top_outliers(recipes, numeric_cols_to_clip)

    # Derived columns
    recipes_clipped['time_bin'] = recipes_clipped['TotalTime_min'].apply(time_bin)
    recipes_clipped['calorie_bin'] = recipes_clipped['Calories'].apply(calorie_bin)
    recipes_clipped['Image_first'] = recipes_clipped['Images'].apply(get_first_url)

    # Replace 0 calories with median of non-zero values
    if 'Calories' in recipes_clipped.columns:
        nonzero_med = recipes_clipped.loc[recipes_clipped['Calories'] > 0, 'Calories'].median()
        recipes_clipped['Calories'] = recipes_clipped['Calories'].replace(0, nonzero_med)

    # Final subset of important columns
    important_columns = [
        'Name',
        'Image_first',
        'recipe_instructions_clean',
        'ingredients_clean',
        'Calories',
        'combined_text',
        'TotalTime_min',
        'time_bin',
        'calorie_bin'
    ]
    final_df = recipes_clipped[important_columns].copy().reset_index(drop=True)

    # Save compressed
    os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
    final_df.to_csv(PREPROCESSED_PATH, index=False, compression='gzip')
    print(f"✅ Preprocessed data saved to {PREPROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_and_save()