"""
Data preprocessing script for agricultural Q&A data.
Loads raw data, cleans text, removes duplicates, and exports clean data.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaning import clean_qa_pair


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Q&A data from CSV file.
    
    Args:
        filepath: Path to raw CSV file
        
    Returns:
        DataFrame with raw data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def extract_qa_columns(df: pd.DataFrame, 
                       question_col: str = 'question',
                       answer_col: str = 'answer') -> Tuple[pd.DataFrame, int]:
    """
    Extract and identify question and answer columns.
    
    Args:
        df: Input DataFrame
        question_col: Name of question column
        answer_col: Name of answer column
        
    Returns:
        Tuple of (DataFrame with extracted columns, rows removed count)
        
    Raises:
        ValueError: If required columns don't exist
    """
    # Check if columns exist (case-insensitive)
    cols_lower = [col.lower() for col in df.columns]
    
    if question_col.lower() not in cols_lower or answer_col.lower() not in cols_lower:
        available_cols = list(df.columns)
        raise ValueError(
            f"Required columns '{question_col}' and '{answer_col}' not found. "
            f"Available columns: {available_cols}"
        )
    
    # Select only Q&A columns, handling case-insensitivity
    q_col = next(col for col in df.columns if col.lower() == question_col.lower())
    a_col = next(col for col in df.columns if col.lower() == answer_col.lower())
    
    df_qa = df[[q_col, a_col]].copy()
    df_qa.columns = ['question', 'answer']
    
    initial_count = len(df_qa)
    return df_qa, initial_count


def remove_empty_entries(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with null or empty question/answer values.
    
    Args:
        df: DataFrame with question and answer columns
        
    Returns:
        Tuple of (cleaned DataFrame, rows removed count)
    """
    initial_count = len(df)
    
    # Remove rows where question or answer is null
    df_clean = df.dropna(subset=['question', 'answer'])
    
    # Remove rows where question or answer is empty string
    df_clean = df_clean[
        (df_clean['question'].astype(str).str.strip() != '') &
        (df_clean['answer'].astype(str).str.strip() != '')
    ]
    
    removed = initial_count - len(df_clean)
    print(f"Removed {removed} empty entries")
    return df_clean, removed


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate Q&A pairs.
    
    Args:
        df: DataFrame with question and answer columns
        
    Returns:
        Tuple of (DataFrame with duplicates removed, duplicates count)
    """
    initial_count = len(df)
    df_unique = df.drop_duplicates(subset=['question', 'answer'], keep='first')
    removed = initial_count - len(df_unique)
    print(f"Removed {removed} duplicate entries")
    return df_unique, removed


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to question and answer columns.
    
    Args:
        df: DataFrame with question and answer columns
        
    Returns:
        DataFrame with cleaned text
    """
    df_cleaned = df.copy()
    
    # Apply cleaning to each Q&A pair
    cleaned_pairs = df_cleaned.apply(
        lambda row: pd.Series(clean_qa_pair(row['question'], row['answer'])),
        axis=1
    )
    
    df_cleaned['question'] = cleaned_pairs[0]
    df_cleaned['answer'] = cleaned_pairs[1]
    
    print(f"Applied text cleaning to {len(df_cleaned)} rows")
    return df_cleaned


def add_metadata(df: pd.DataFrame) -> List[Dict]:
    """
    Create Q&A pairs with metadata.
    
    Args:
        df: DataFrame with cleaned question and answer columns
        
    Returns:
        List of dictionaries with question, answer, and metadata
    """
    qa_pairs = []
    timestamp = datetime.now().isoformat()
    
    for idx, row in df.iterrows():
        pair = {
            'id': idx,
            'question': row['question'],
            'answer': row['answer'],
            'metadata': {
                'source': 'kisan-call-centre',
                'processed_at': timestamp,
                'index': int(idx)
            }
        }
        qa_pairs.append(pair)
    
    return qa_pairs


def save_clean_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned data to CSV file.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to output CSV file
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved cleaned CSV to {output_path}")


def save_qa_pairs_json(qa_pairs: List[Dict], output_path: str) -> None:
    """
    Save Q&A pairs with metadata to JSON file.
    
    Args:
        qa_pairs: List of Q&A pair dictionaries
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved Q&A pairs JSON to {output_path}")


def preprocess_data(raw_data_path: str,
                   clean_csv_path: str,
                   qa_pairs_json_path: str,
                   question_col: str = 'question',
                   answer_col: str = 'answer') -> Dict:
    """
    Complete preprocessing pipeline.
    
    Args:
        raw_data_path: Path to raw CSV file
        clean_csv_path: Path to save cleaned CSV
        qa_pairs_json_path: Path to save Q&A JSON
        question_col: Name of question column
        answer_col: Name of answer column
        
    Returns:
        Dictionary with preprocessing statistics
    """
    print("=" * 60)
    print("Starting data preprocessing pipeline")
    print("=" * 60)
    
    # Load raw data
    df = load_raw_data(raw_data_path)
    initial_rows = len(df)
    
    # Extract Q&A columns
    df, _ = extract_qa_columns(df, question_col, answer_col)
    
    # Remove empty entries
    df, empty_removed = remove_empty_entries(df)
    
    # Remove duplicates
    df, duplicates_removed = remove_duplicates(df)
    
    # Apply text cleaning
    df = apply_text_cleaning(df)
    
    final_rows = len(df)
    
    # Save cleaned CSV
    save_clean_csv(df, clean_csv_path)
    
    # Create and save Q&A pairs with metadata
    qa_pairs = add_metadata(df)
    save_qa_pairs_json(qa_pairs, qa_pairs_json_path)
    
    # Print summary statistics
    stats = {
        'initial_rows': initial_rows,
        'empty_removed': empty_removed,
        'duplicates_removed': duplicates_removed,
        'final_rows': final_rows,
        'reduction_percentage': round((1 - final_rows / initial_rows) * 100, 2)
    }
    
    print("=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Initial rows: {stats['initial_rows']}")
    print(f"Empty entries removed: {stats['empty_removed']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Final rows: {stats['final_rows']}")
    print(f"Data reduction: {stats['reduction_percentage']}%")
    print("=" * 60)
    
    return stats


def main():
    """Main entry point for the preprocessing script."""
    # Set file paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw_kcc.csv')
    clean_csv_path = os.path.join(project_root, 'data', 'clean_kcc.csv')
    qa_pairs_json_path = os.path.join(project_root, 'data', 'kcc_qa_pairs.json')
    
    try:
        preprocess_data(
            raw_data_path=raw_data_path,
            clean_csv_path=clean_csv_path,
            qa_pairs_json_path=qa_pairs_json_path
        )
        print("\n✓ Preprocessing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
