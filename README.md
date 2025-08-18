# 🍳 ML Food Buddy Recommender

An intelligent recipe recommendation system that helps users discover personalized recipes based on ingredients, cooking time, and calorie preferences using TF-IDF vectorization and cosine similarity.

## ✨ Features

- **Smart Ingredient Matching**: Uses TF-IDF vectorization for semantic recipe matching
- **Auto-correction**: Fixes common typos in ingredient queries using fuzzy matching
- **Flexible Filtering**: Filter by cooking time (fast/medium/long) and calories (low/medium/high)
- **Interactive Web App**: Clean Streamlit interface with recipe images and detailed information
- **Recipe Details**: View ingredients, instructions, cooking time, and nutritional information

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- ~2GB free disk space (for dataset)

### Installation

1. **Check your environment**
   
   Check Python version:
   ```bash
   python3 --version
   ```
   
   Check pip version:
   ```bash
   pip3 --version
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-food-buddy-recommender.git
   cd ml-food-buddy-recommender
   ```

3. **Create a virtual environment**
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   ```
   
   This creates a folder called `venv` in your project directory.

4. **Activate the virtual environment**
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   After activation, your terminal prompt should show `(venv)`.

5. **Upgrade pip (recommended)**
   ```bash
   pip install --upgrade pip
   ```

6. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs all required Python packages locally inside the virtual environment, avoiding conflicts with system Python.

7. **Download the dataset**
   - Go to [Food.com Recipes Dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
   - Download `recipes.csv` (704MB)
   - Place it in `data/raw/recipes.csv`

8. **Run preprocessing and vectorizer scripts**
   
   1. **Run preprocessing:**
      ```bash
      python -c "from src.preprocessing import preprocess_and_save; preprocess_and_save()"
      ```
   
   2. **Train and save vectorizer:**
      ```bash
      python -c "from src.vectorizer import train_and_save_vectorizer; train_and_save_vectorizer()"
      ```
   
   Both commands must be run **inside the virtual environment**.

9. **Launch the web app**
   ```bash
   streamlit run app/app.py
   ```

## 📁 Project Structure

```
ml-food-buddy-recommender/
├── app/
│   └── app.py                   # Main Streamlit web application
├── data/
│   ├── raw/                     # Place recipes.csv here
│   └── preprocessed/            # Generated processed data
├── docs/
│   ├── index.html               # Demo page
│   └── media/                   # Demo videos
├── models/                      # Generated ML models
│   ├── tfidf_vectorizer.pkl
│   └── recipe_vectors.npy
├── notebooks/
│   └── recommender_demo.ipynb   # Interactive demo notebook
├── src/
│   ├── config.py                # Configuration settings
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── vectorizer.py           # TF-IDF model training
│   ├── recommender.py          # Core recommendation logic
│   └── utils.py                # Utility functions
├── tests/                       # Test files
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## 🔧 How It Works

1. **Data Preprocessing**: Raw recipe data is cleaned, normalized, and feature-engineered
2. **Vectorization**: Recipe descriptions are converted to TF-IDF vectors for similarity matching
3. **Query Processing**: User inputs are cleaned and auto-corrected for better matching
4. **Similarity Computation**: Cosine similarity between user query and recipe vectors
5. **Filtering & Ranking**: Results are filtered by preferences and ranked by relevance

## 💻 Usage Examples

### Web Application
```python
# Launch the Streamlit app
streamlit run app/app.py

# Then enter queries like:
# - "italian pasta vegetarian"
# - "quick chicken dinner"
# - "healthy breakfast low calorie"
```

### Jupyter Notebook
Open `notebooks/recommender_demo.ipynb` for an interactive exploration of the recommendation system.

## 🎯 Key Components

### Smart Query Correction
Automatically fixes common typos in ingredient names:
- "italain pastaa" → "italian pasta"
- "chiken" → "chicken"
- "vegeterian" → "vegetarian"

### Flexible Filtering
- **Time**: fast (≤30 min), medium (31-90 min), long (>90 min)
- **Calories**: low (<300), medium (300-600), high (>600)

### Robust Data Processing
- Handles malformed CSV files and encoding issues
- Cleans and normalizes recipe instructions and ingredients
- Removes outliers and fills missing values intelligently

## 📊 Dataset

The system uses the [Food.com Recipes and Reviews dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) containing:
- 500,000+ recipes
- Ingredients, instructions, nutritional info
- Cooking times, categories, and user ratings
- Recipe images and metadata

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
- Built with Python, scikit-learn, pandas, and Streamlit
- Uses RapidFuzz for intelligent query correction

Buon appetito! 👨‍🍳👩‍🍳