"""
Text cleaning utilities for agricultural Q&A data.
"""

import re
from typing import Optional, Tuple


def remove_special_characters(text: str) -> str:
    """
    Remove special characters while preserving spaces and basic punctuation.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text with removed special characters
    """
    # Keep alphanumeric, spaces, and common punctuation (. , ? ! - ')
    text = re.sub(r'[^a-zA-Z0-9\s.?,!\'&\-]', '', text)
    return text


def remove_extra_spaces(text: str) -> str:
    """
    Remove leading, trailing, and extra spaces between words.
    
    Args:
        text: Input text string
        
    Returns:
        Text with normalized spacing
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return text.strip()


def normalize_lowercase(text: str) -> str:
    """
    Normalize text to lowercase.
    
    Args:
        text: Input text string
        
    Returns:
        Lowercase text
    """
    return text.lower()


def handle_null_values(text: Optional[str]) -> str:
    """
    Handle null, None, or empty values safely.
    
    Args:
        text: Input text which may be None or empty
        
    Returns:
        Empty string if input is None/empty, otherwise returns text
    """
    if text is None or (isinstance(text, str) and len(text.strip()) == 0):
        return ""
    return str(text)


def clean_text(text: Optional[str]) -> str:
    """
    Apply full cleaning pipeline to text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    # Handle null/empty values first
    text = handle_null_values(text)
    
    if not text:
        return ""
    
    # Remove special characters
    text = remove_special_characters(text)
    
    # Remove extra spaces
    text = remove_extra_spaces(text)
    
    # Normalize to lowercase
    text = normalize_lowercase(text)
    
    return text


def clean_question(question: Optional[str]) -> str:
    """
    Clean agricultural question text.
    
    Args:
        question: Raw question string
        
    Returns:
        Cleaned question string
    """
    return clean_text(question)


def clean_answer(answer: Optional[str]) -> str:
    """
    Clean agricultural answer text.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Cleaned answer string
    """
    return clean_text(answer)


def clean_qa_pair(question: Optional[str], answer: Optional[str]) -> Tuple[str, str]:
    """
    Clean both question and answer in a Q&A pair.
    
    Args:
        question: Raw question string
        answer: Raw answer string
        
    Returns:
        Tuple of (cleaned_question, cleaned_answer)
    """
    cleaned_question = clean_question(question)
    cleaned_answer = clean_answer(answer)
    
    return cleaned_question, cleaned_answer
