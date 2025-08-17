# Configuration file for the ML Food Buddy Recommender project
import os
import re
import html
import pandas as pd
import numpy as np
import gzip
from io import StringIO
from rapidfuzz import process, fuzz

def data_loader(filename: str, source: str = "raw"):
    """
    Data loader that handles:
    - Corrupted/mislabeled GZIP files
    - Encoding issues
    - Multiple compression formats
    """
    # Validate source and find file
    if source not in ["raw", "preprocessed"]:
        raise ValueError("source must be 'raw' or 'preprocessed'")

    cwd = os.getcwd()
    repo_root = cwd
    while True:
        data_path = os.path.join(repo_root, "data", source, filename)
        if os.path.exists(data_path):
            break
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            raise FileNotFoundError(f"Could not find {filename} in data/{source} from {cwd}")
        repo_root = parent

    # Special handling for problematic GZIP files
    if filename.endswith(('.gz', '.gzip')):
        try:
            # Method 1: Try standard pandas reading
            return pd.read_csv(data_path, compression='gzip', encoding='utf-8')
        except:
            try:
                # Method 2: Manual gzip decompression with encoding fallback
                with gzip.open(data_path, 'rb') as f:
                    content = f.read().decode('latin1')
                return pd.read_csv(StringIO(content))
            except:
                # Method 3: Binary read and manual cleanup
                with open(data_path, 'rb') as f:
                    content = f.read()
                    # Skip GZIP header if present
                    if content.startswith(b'\x1f\x8b'):
                        content = gzip.decompress(content)
                    return pd.read_csv(StringIO(content.decode('latin1')))
    
    # Handle other file types
    elif filename.endswith('.zip'):
        return pd.read_csv(data_path, compression='zip')
    else:
        # Try multiple encodings for regular CSV
        for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(data_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to decode file with any supported encoding")

def parse_time(t):
    """
    Parse time in minutes. Supports:
    - Raw numeric strings (e.g., "45")
    - Already numeric values
    
    Returns:
        float (minutes) or np.nan if parsing fails
    """
    if pd.isna(t):
        return np.nan
    if isinstance(t, str):
        t = t.strip()
        if t.startswith("PT"):  # ISO 8601 duration
            hours = re.search(r'(\d+)H', t)
            minutes = re.search(r'(\d+)M', t)
            secs = re.search(r'(\d+)S', t)
            total_minutes = 0
            if hours:
                total_minutes += int(hours.group(1)) * 60
            if minutes:
                total_minutes += int(minutes.group(1))
            if secs:
                total_minutes += int(secs.group(1)) / 60
            return total_minutes if total_minutes > 0 else np.nan
        # fallback: try to parse as float
        try:
            return float(t)
        except:
            return np.nan
    # If already numeric
    try:
        return float(t)
    except:
        return np.nan

def format_time(t):
    """
    Convert a time in minutes to a human-readable string format.

    Args:
        t (int or float or None): Total time in minutes. Can be NaN.

    Returns:
        str or None: A string representing the time in hours and minutes, e.g.:
                     - 135 → "2 hours 15 minutes"
                     - 60  → "1 hour"
                     - 45  → "45 minutes"
                     - 0 or NaN → "0 minutes" or None if input is NaN
    """
    if pd.isna(t):
        return None
    t = int(round(t))
    hours, minutes = divmod(t, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    return " ".join(parts) if parts else "0 minutes"

def clip_top_outliers(df, cols, z_thresh=3.5):
    """
    Clip only extreme outliers of selected columns using modified Z-score.
    
    Parameters:
    - df: pd.DataFrame
    - cols: list of str, columns to clip
    - z_thresh: float, threshold for modified Z-score (default=3.5)
    
    Returns:
    - df_clipped: pd.DataFrame with clipped values
    - thresholds: dict of column:clip_value for reference
    """
    df_clipped = df.copy()
    thresholds = {}
    
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            continue  # can't detect outliers if MAD is zero
        mod_z = 0.6745 * (series - median) / mad
        upper_limit = series[mod_z <= z_thresh].max()  # largest non-outlier
        df_clipped[col] = np.minimum(series, upper_limit)
        thresholds[col] = upper_limit
    
    return df_clipped, thresholds

def correct_query(query, recipes, vocab_columns=None, threshold=80):
    """
    Corrects a user query using RapidFuzz, matching words to recipe vocabulary.
    Only replaces words if a close match is found above threshold.
    
    Args:
        query: Input query string (e.g., "Italain paste vegeterian")
        recipes: DataFrame with vocabulary columns
        vocab_columns: List of column names to extract vocabulary from. 
        threshold: Minimum similarity score to accept correction (0-100)
    
    Returns:
        Corrected query string (e.g., "Italian pasta vegetarian")
    """
    # Return original query if empty
    if vocab_columns is None:
        return query
    
    # Extract vocabulary from specified columns
    vocab = set()
    
    for col_name in vocab_columns:
        if col_name in recipes.columns:
            for cell_value in recipes[col_name].dropna():
                if isinstance(cell_value, str):
                    words = cell_value.replace(',', ' ').split()
                    vocab.update(word.strip().lower() for word in words if word.strip())
                elif isinstance(cell_value, list):
                    vocab.update(word.strip().lower() for word in cell_value if isinstance(word, str) and word.strip())
    
    vocab = [word for word in vocab if len(word) > 2]  # Filter out very short words
    
    if not vocab:
        return query
    
    words = query.lower().split()
    corrected = []
    
    for word in words:
        # Skip very short words
        if len(word) <= 2:
            corrected.append(word)
            continue
            
        # Find the best match from vocab
        result = process.extractOne(word, vocab, scorer=fuzz.WRatio)
        
        if result and result[1] >= threshold:
            corrected.append(result[0])
        else:
            corrected.append(word)
    
    return ' '.join(corrected)

def parse_r_list_column(col, extended_clean=False):
    """
    Parse and clean R-style list strings in a pandas Series.
    Args:
        col: pandas Series containing R-style list strings
        extended_clean: bool, if False applies basic cleaning only,
                       if True applies extended cleaning (whitespace collapse + trailing comma removal)
    Returns:
        pandas Series with cleaned text
    """
    # Handle None input
    if col is None:
        return pd.Series([], dtype=str)
    
    # Basic cleaning (always applied)
    cleaned = col.fillna("").astype(str).str.strip()
    
    # Remove c( and ) wrapper (always applied)
    cleaned = cleaned.str.replace(r'^c\(|\)$', '', regex=True)  # Fixed: added $ at the end
    
    # Remove quotes (always applied)
    cleaned = cleaned.str.replace(r'"', '', regex=False)
    
    # Extended cleaning (only if requested)
    if extended_clean:
        # Collapse multiple spaces/newlines
        cleaned = cleaned.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        # Remove comma immediately following a period (anywhere in text, not just end)
        cleaned = cleaned.str.replace(r'\.(?=\s*,)', '.', regex=True)
        cleaned = cleaned.str.replace(r'\.,\s*', '. ', regex=True)
        
        # Decode HTML entities
        cleaned = cleaned.apply(lambda x: html.unescape(x))
    
    return cleaned

def clean_text(text):
    """
    Clean a text string by normalizing case, whitespace, and punctuation.

    - Converts to lowercase
    - Replaces multiple whitespace characters with a single space
    - Removes punctuation and special characters
    - Strips leading and trailing whitespace
    """
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def time_bin(minutes):
    """
    Categorize a recipe's total time into a time bin.

    Parameters:
    minutes (float or int): Total time of the recipe in minutes.

    Returns:
    str or None: 
        - "fast" if time is 30 minutes or less
        - "medium" if time is between 31 and 90 minutes
        - "long" if time is more than 90 minutes
        - None if input is NaN
    """
    if pd.isna(minutes):
        return None
    if minutes <= 30:
        return "fast"
    elif minutes <= 90:
        return "medium"
    else:
        return "long"

def calorie_bin(cals):
    """
    Categorize a recipe's calories into a calorie bin.

    Parameters:
    cals (float or int): Total calories of the recipe.

    Returns:
    str or None:
        - "low" if calories are less than 300
        - "medium" if calories are between 300 and 600
        - "high" if calories are more than 600
        - None if input is NaN
    """
    if pd.isna(cals):
        return None
    if cals < 300:
        return "low"
    elif cals <= 600:
        return "medium"
    else:
        return "high"

def get_first_url(cell):
    """
    Extract the first URL from a string, trying multiple methods.
    
    Parameters:
        cell (str or any): Input that might contain URLs
        
    Returns:
        str or None: First found URL or None
    """
    if pd.isna(cell):
        return 'Image not available'
        
    text = str(cell)
    
    # Try 1: Look for properly quoted URLs first
    quoted_url = re.search(r'"(https?://[^"]*)"', text)
    if quoted_url:
        return quoted_url.group(1)
    
    # Try 2: Look for any URL pattern without quotes
    any_url = re.search(r'(https?://\S+)', text)
    if any_url:
        return any_url.group(1)
    
    # Try 3: Look for common image extensions without full URL
    common_ext = re.search(r'(\S+\.(?:jpg|jpeg|png|gif|webp)\b)', text, re.IGNORECASE)
    if common_ext:
        return common_ext.group(1)
        
    return None

def apply_categorical_filters(dataset, filters: dict):
    """
    Apply generic categorical filters to a DataFrame.

    Args:
        dataset (pd.DataFrame): DataFrame to filter.
        filters (dict): Dictionary where keys are column names and values are filter values
                        (str or list of str) to keep in that column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered = dataset.copy()
    
    for column, values in filters.items():
        if column not in filtered.columns or not values:
            continue
        if isinstance(values, str):
            values = [values]
        clean_values = [str(v).strip().lower() for v in values]
        filtered = filtered[filtered[column].fillna('').astype(str).str.strip().str.lower().isin(clean_values)]
    
    return filtered