import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import gzip
from io import StringIO
from unittest.mock import patch, mock_open
import sys
import tempfile
from src.utils import (
    data_loader, parse_time, format_time, clip_top_outliers,
    correct_query, parse_r_list_column, clean_text, time_bin,
    calorie_bin, get_first_url, apply_categorical_filters
)


class TestDataLoader:
    """Test suite for data_loader function"""
    
    def test_invalid_source_raises_error(self):
        """Test that invalid source parameter raises ValueError"""
        with pytest.raises(ValueError, match="source must be 'raw' or 'preprocessed'"):
            data_loader("test.csv", source="invalid")
    
    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            data_loader("nonexistent_file.csv")
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_regular_csv_loading(self, mock_read_csv, mock_exists):
        """Test loading regular CSV files"""
        # Setup mocks
        mock_exists.return_value = True
        expected_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = expected_df
        
        result = data_loader("test.csv")
        
        assert result.equals(expected_df)
        mock_read_csv.assert_called_once()
    
    def test_gzip_file_loading_with_real_data(self):
        """Test loading gzipped files with actual data"""
        # Create a temporary gzipped CSV
        test_data = "col1,col2\n1,a\n2,b\n3,c"
        
        with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as f:
            with gzip.open(f.name, 'wt', encoding='utf-8') as gz_file:
                gz_file.write(test_data)
            
            # Create the expected directory structure
            temp_dir = tempfile.mkdtemp()
            data_dir = os.path.join(temp_dir, 'data', 'raw')
            os.makedirs(data_dir)
            
            # Move the file to the expected location
            target_file = os.path.join(data_dir, 'test.csv.gz')
            os.rename(f.name, target_file)
            
            try:
                with patch('os.getcwd', return_value=temp_dir):
                    result = data_loader("test.csv.gz", source="raw")
                    
                    expected_df = pd.DataFrame({
                        'col1': [1, 2, 3],
                        'col2': ['a', 'b', 'c']
                    })
                    
                    pd.testing.assert_frame_equal(result, expected_df)
            finally:
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir)


class TestParseTime:
    """Test suite for parse_time function"""
    
    def test_parse_time_nan_input(self):
        """Test parsing NaN input returns NaN"""
        assert pd.isna(parse_time(np.nan))
        assert pd.isna(parse_time(None))
    
    def test_parse_time_numeric_string(self):
        """Test parsing numeric string"""
        assert parse_time("45") == 45.0
        assert parse_time("45.5") == 45.5
        assert parse_time(" 30 ") == 30.0
    
    def test_parse_time_iso_duration(self):
        """Test parsing ISO 8601 duration format"""
        assert parse_time("PT1H30M") == 90.0  # 1 hour 30 minutes
        assert parse_time("PT45M") == 45.0    # 45 minutes
        assert parse_time("PT2H") == 120.0    # 2 hours
        assert parse_time("PT30S") == 0.5     # 30 seconds
        assert parse_time("PT1H15M30S") == 75.5  # 1h 15m 30s
    
    def test_parse_time_numeric_input(self):
        """Test parsing already numeric input"""
        assert parse_time(45) == 45.0
        assert parse_time(45.5) == 45.5
    
    def test_parse_time_invalid_string(self):
        """Test parsing invalid string returns NaN"""
        assert pd.isna(parse_time("invalid"))
        assert pd.isna(parse_time(""))


class TestFormatTime:
    """Test suite for format_time function"""
    
    def test_format_time_nan(self):
        """Test formatting NaN returns None"""
        assert format_time(np.nan) is None
        assert format_time(None) is None
    
    def test_format_time_minutes_only(self):
        """Test formatting minutes only"""
        assert format_time(45) == "45 minutes"
        assert format_time(1) == "1 minute"
        assert format_time(0) == "0 minutes"
    
    def test_format_time_hours_only(self):
        """Test formatting exact hours"""
        assert format_time(60) == "1 hour"
        assert format_time(120) == "2 hours"
    
    def test_format_time_hours_and_minutes(self):
        """Test formatting hours and minutes"""
        assert format_time(90) == "1 hour 30 minutes"
        assert format_time(135) == "2 hours 15 minutes"
        assert format_time(61) == "1 hour 1 minute"


class TestClipTopOutliers:
    """Test suite for clip_top_outliers function"""
    
    def test_clip_outliers_basic(self):
        """Test basic outlier clipping functionality"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'col2': [10, 20, 30, 40, 50]
        })
        
        clipped_df, thresholds = clip_top_outliers(df, ['col1'], z_thresh=2.0)
        
        # The outlier should be clipped
        assert clipped_df['col1'].max() < 100
        assert 'col1' in thresholds
        # col2 should be unchanged
        pd.testing.assert_series_equal(clipped_df['col2'], df['col2'])
    
    def test_clip_outliers_no_outliers(self):
        """Test when no outliers exist"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })
        
        clipped_df, thresholds = clip_top_outliers(df, ['col1'])
        
        pd.testing.assert_frame_equal(clipped_df, df)
    
    def test_clip_outliers_nonexistent_column(self):
        """Test with non-existent column"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        clipped_df, thresholds = clip_top_outliers(df, ['nonexistent'])
        
        pd.testing.assert_frame_equal(clipped_df, df)
        assert 'nonexistent' not in thresholds


class TestCorrectQuery:
    """Test suite for correct_query function"""
    
    def test_correct_query_basic(self):
        """Test basic query correction"""
        recipes = pd.DataFrame({
            'ingredients': ['italian pasta', 'vegetarian pizza', 'meat sauce'],
            'cuisine': ['italian', 'italian', 'american']
        })
        
        result = correct_query(
            "italain pastaa vegeterian", 
            recipes, 
            vocab_columns=['ingredients', 'cuisine'],
            threshold=70
        )
        
        assert 'italian' in result.lower()
        assert 'pasta' in result.lower()
        assert 'vegetarian' in result.lower()
    
    def test_correct_query_no_vocab_columns(self):
        """Test with no vocab columns specified"""
        recipes = pd.DataFrame({'col': ['test']})
        query = "test query"
        
        result = correct_query(query, recipes, vocab_columns=None)
        
        assert result == query
    
    def test_correct_query_empty_vocab(self):
        """Test with empty vocabulary"""
        recipes = pd.DataFrame({'col': [np.nan, '', None]})
        query = "test query"
        
        result = correct_query(query, recipes, vocab_columns=['col'])
        
        assert result == query
    
    def test_correct_query_high_threshold(self):
        """Test with high threshold (no corrections)"""
        recipes = pd.DataFrame({'ingredients': ['pasta']})
        
        result = correct_query(
            "pastaa", 
            recipes, 
            vocab_columns=['ingredients'],
            threshold=99
        )
        
        # Should not correct due to high threshold
        assert result == "pastaa"


class TestParseRListColumn:
    """Test suite for parse_r_list_column function"""
    
    def test_parse_r_list_basic(self):
        """Test basic R-style list parsing"""
        col = pd.Series(['c("apple", "banana")', 'c("orange")'])
        
        result = parse_r_list_column(col)
        
        assert result.iloc[0] == 'apple, banana'
        assert result.iloc[1] == 'orange'
    
    def test_parse_r_list_extended_clean(self):
        """Test extended cleaning functionality"""
        col = pd.Series(['c("test.,  multiple   spaces")', 'c("normal text")'])
        
        result = parse_r_list_column(col, extended_clean=True)
        
        assert 'test. multiple spaces' in result.iloc[0]
    
    def test_parse_r_list_none_input(self):
        """Test with None input"""
        result = parse_r_list_column(None)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0
    
    def test_parse_r_list_with_nan(self):
        """Test with NaN values"""
        col = pd.Series([np.nan, 'c("test")'])
        
        result = parse_r_list_column(col)
        
        assert result.iloc[0] == ''
        assert result.iloc[1] == 'test'


class TestCleanText:
    """Test suite for clean_text function"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "Hello, World! This is a TEST."
        
        result = clean_text(text)
        
        assert result == "hello world this is a test"
    
    def test_clean_text_multiple_spaces(self):
        """Test cleaning multiple spaces"""
        text = "hello    world\n\ttab"
        
        result = clean_text(text)
        
        assert result == "hello world tab"
    
    def test_clean_text_punctuation(self):
        """Test punctuation removal"""
        text = "hello, world! @#$%"
        
        result = clean_text(text)
        
        assert result == "hello world"
    
    def test_clean_text_numbers_preserved(self):
        """Test that numbers are preserved"""
        text = "Recipe123 with 2 cups"
        
        result = clean_text(text)
        
        assert "123" in result
        assert "2" in result


class TestTimeBin:
    """Test suite for time_bin function"""
    
    def test_time_bin_fast(self):
        """Test fast time bin"""
        assert time_bin(15) == "fast"
        assert time_bin(30) == "fast"
    
    def test_time_bin_medium(self):
        """Test medium time bin"""
        assert time_bin(31) == "medium"
        assert time_bin(60) == "medium"
        assert time_bin(90) == "medium"
    
    def test_time_bin_long(self):
        """Test long time bin"""
        assert time_bin(91) == "long"
        assert time_bin(120) == "long"
    
    def test_time_bin_nan(self):
        """Test NaN input"""
        assert time_bin(np.nan) is None
        assert time_bin(None) is None


class TestCalorieBin:
    """Test suite for calorie_bin function"""
    
    def test_calorie_bin_low(self):
        """Test low calorie bin"""
        assert calorie_bin(250) == "low"
        assert calorie_bin(299) == "low"
    
    def test_calorie_bin_medium(self):
        """Test medium calorie bin"""
        assert calorie_bin(300) == "medium"
        assert calorie_bin(450) == "medium"
        assert calorie_bin(600) == "medium"
    
    def test_calorie_bin_high(self):
        """Test high calorie bin"""
        assert calorie_bin(601) == "high"
        assert calorie_bin(1000) == "high"
    
    def test_calorie_bin_nan(self):
        """Test NaN input"""
        assert calorie_bin(np.nan) is None
        assert calorie_bin(None) is None


class TestGetFirstUrl:
    """Test suite for get_first_url function"""
    
    def test_get_first_url_quoted(self):
        """Test extracting quoted URL"""
        text = 'Some text "https://example.com/image.jpg" more text'
        
        result = get_first_url(text)
        
        assert result == "https://example.com/image.jpg"
    
    def test_get_first_url_unquoted(self):
        """Test extracting unquoted URL"""
        text = "Check out https://example.com/page for more info"
        
        result = get_first_url(text)
        
        assert result == "https://example.com/page"
    
    def test_get_first_url_image_extension(self):
        """Test extracting image with common extension"""
        text = "image123.jpg in this text"
        
        result = get_first_url(text)
        
        assert result == "image123.jpg"
    
    def test_get_first_url_nan_input(self):
        """Test with NaN input"""
        result = get_first_url(np.nan)
        
        assert result == "Image not available"
    
    def test_get_first_url_no_url(self):
        """Test with text containing no URL"""
        result = get_first_url("just plain text")
        
        assert result is None


class TestApplyCategoricalFilters:
    """Test suite for apply_categorical_filters function"""
    
    def test_apply_filters_single_value(self):
        """Test applying filter with single value"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4]
        })
        
        filters = {'category': 'A'}
        result = apply_categorical_filters(df, filters)
        
        assert len(result) == 2
        assert all(result['category'] == 'A')
    
    def test_apply_filters_multiple_values(self):
        """Test applying filter with multiple values"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4]
        })
        
        filters = {'category': ['A', 'B']}
        result = apply_categorical_filters(df, filters)
        
        assert len(result) == 3
        assert all(result['category'].isin(['A', 'B']))
    
    def test_apply_filters_case_insensitive(self):
        """Test case-insensitive filtering"""
        df = pd.DataFrame({
            'category': ['Apple', 'BANANA', 'apple'],
            'value': [1, 2, 3]
        })
        
        filters = {'category': 'apple'}
        result = apply_categorical_filters(df, filters)
        
        assert len(result) == 2
    
    def test_apply_filters_nonexistent_column(self):
        """Test filtering with non-existent column"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        filters = {'nonexistent': 'value'}
        result = apply_categorical_filters(df, filters)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_apply_filters_empty_filters(self):
        """Test with empty filters"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        result = apply_categorical_filters(df, {})
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_apply_filters_with_nan_values(self):
        """Test filtering with NaN values in data"""
        df = pd.DataFrame({
            'category': ['A', np.nan, 'A', 'B'],
            'value': [1, 2, 3, 4]
        })
        
        filters = {'category': 'A'}
        result = apply_categorical_filters(df, filters)
        
        assert len(result) == 2
        assert all(result['category'] == 'A')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])