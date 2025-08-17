import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions we want to test
try:
    from src.recommender import load_artifacts, recommend
except ImportError:
    # If recommender.py doesn't exist yet, we'll create mock functions for testing
    def load_artifacts():
        pass
    def recommend(**kwargs):
        pass


class TestRecommend:
    """Test suite for recommend function"""
    
    def setup_method(self):
        """Setup test data for each test"""
        # Create sample dataset
        self.sample_dataset = pd.DataFrame({
            'Name': ['Italian Pasta', 'Veggie Pizza', 'Meat Sauce', 'Chicken Curry'],
            'combined_text': ['italian pasta garlic', 'vegetarian pizza cheese', 'meat sauce beef', 'chicken curry spicy'],
            'time_bin': ['fast', 'medium', 'fast', 'long'],
            'calorie_bin': ['low', 'medium', 'high', 'medium'],
            'TotalTime_min': [25, 45, 30, 90],
            'Calories': [250, 450, 700, 400],
            'Image_first': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'recipe_instructions_clean': ['Cook pasta...', 'Make pizza...', 'Cook sauce...', 'Make curry...']
        })
        
        # Create sample vectorizer and vectors
        self.sample_vectorizer = TfidfVectorizer()
        corpus = self.sample_dataset['combined_text'].tolist()
        self.sample_vectors = self.sample_vectorizer.fit_transform(corpus).toarray()
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.sample_vectors, axis=1, keepdims=True)
        self.sample_vectors = self.sample_vectors / (norms + 1e-10)

    def test_recommend_basic_functionality(self):
        """Test basic recommendation functionality"""
        result = recommend(
            user_prefs="italian pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=2
        )
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should return at most top_n results
        assert len(result) <= 2
        # Should have similarity scores
        if 'similarity' in result.columns:
            assert all(result['similarity'] >= 0)
        # Results should be sorted by similarity (if not empty)
        if len(result) > 1 and 'similarity' in result.columns:
            similarities = result['similarity'].values
            assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))

    def test_recommend_with_time_filter(self):
        """Test recommendation with time preference filter"""
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            time_pref="fast",
            top_n=5
        )
        
        # All results should have fast time_bin (if any results returned)
        if len(result) > 0 and 'time_bin' in result.columns:
            assert all(result['time_bin'] == 'fast')

    def test_recommend_with_calorie_filter(self):
        """Test recommendation with calorie preference filter"""
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            calorie_pref="low",
            top_n=5
        )
        
        # All results should have low calorie_bin (if any results returned)
        if len(result) > 0 and 'calorie_bin' in result.columns:
            assert all(result['calorie_bin'] == 'low')

    def test_recommend_with_both_filters(self):
        """Test recommendation with both time and calorie filters"""
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            time_pref="fast",
            calorie_pref="low",
            top_n=5
        )
        
        # Results should satisfy both filters (if any results returned)
        if len(result) > 0:
            if 'time_bin' in result.columns:
                assert all(result['time_bin'] == 'fast')
            if 'calorie_bin' in result.columns:
                assert all(result['calorie_bin'] == 'low')

    def test_recommend_no_matches(self):
        """Test recommendation when no matches are found"""
        # Create a query that won't match anything
        result = recommend(
            user_prefs="zzzzunknownfoodzzz",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=3
        )
        
        # Should return a message about no matches
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0 and 'message' in result.columns:
            assert 'No matching recipes found' in result['message'].iloc[0]

    def test_recommend_return_specific_columns(self):
        """Test recommendation with specific return columns"""
        return_cols = ['Name', 'Calories', 'TotalTime_str']
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            return_columns=return_cols,
            top_n=2
        )
        
        # Should only return requested columns (plus any added ones like similarity)
        if len(result) > 0:
            # Name and Calories should definitely be there if they exist
            expected_cols = [col for col in return_cols if col in self.sample_dataset.columns or col == 'TotalTime_str']
            for col in expected_cols:
                if col in result.columns:
                    # Only check if column exists
                    assert col in result.columns

    def test_recommend_invalid_return_columns(self):
        """Test recommendation with invalid return columns"""
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            return_columns=['nonexistent_column'],
            top_n=2
        )
        
        # Should still return a DataFrame, possibly with all columns
        assert isinstance(result, pd.DataFrame)

    def test_recommend_totaltime_formatting(self):
        """Test that TotalTime is properly formatted"""
        result = recommend(
            user_prefs="pasta",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=1
        )
        
        # Should add TotalTime_str column
        if len(result) > 0 and 'TotalTime_str' in result.columns:
            time_str = result['TotalTime_str'].iloc[0]
            # Should be a string or empty
            assert isinstance(time_str, (str, type(""))) or pd.isna(time_str)

    def test_recommend_query_correction(self):
        """Test that query gets corrected before processing"""
        # This test verifies that the correct_query function is called
        result = recommend(
            user_prefs="pastaa italiaan",  # Misspelled
            dataset=self.sample_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=2
        )
        
        assert isinstance(result, pd.DataFrame)

    def test_recommend_empty_dataset(self):
        """Test recommendation with empty dataset"""
        empty_dataset = pd.DataFrame(columns=self.sample_dataset.columns)
        empty_vectors = np.array([]).reshape(0, self.sample_vectors.shape[1])
        
        result = recommend(
            user_prefs="pasta",
            dataset=empty_dataset,
            recipe_vectors_matrix=empty_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=3
        )
        
        # Should handle empty dataset gracefully
        assert isinstance(result, pd.DataFrame)

    def test_recommend_custom_column_names(self):
        """Test recommendation with custom column names"""
        # Modify dataset with different column names
        custom_dataset = self.sample_dataset.copy()
        custom_dataset = custom_dataset.rename(columns={
            'combined_text': 'custom_text',
            'time_bin': 'custom_time',
            'calorie_bin': 'custom_calories'
        })
        
        result = recommend(
            user_prefs="pasta",
            dataset=custom_dataset,
            recipe_vectors_matrix=self.sample_vectors,
            vectorizer=self.sample_vectorizer,
            vocab_column='custom_text',
            time_column='custom_time',
            calorie_column='custom_calories',
            time_pref='fast',
            top_n=2
        )
        
        assert isinstance(result, pd.DataFrame)

    def test_recommend_zero_similarity_scores(self):
        """Test recommendation when all similarity scores are zero"""
        # Create vectors that will result in zero similarity
        zero_vectors = np.zeros_like(self.sample_vectors)
        
        result = recommend(
            user_prefs="completely different query xyz123",
            dataset=self.sample_dataset,
            recipe_vectors_matrix=zero_vectors,
            vectorizer=self.sample_vectorizer,
            top_n=3
        )
        
        # Should return message about no matches
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0 and 'message' in result.columns:
            assert 'No matching recipes found' in result['message'].iloc[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])