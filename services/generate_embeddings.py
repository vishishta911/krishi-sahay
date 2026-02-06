"""
Embedding generation script for agricultural Q&A data.
Uses SentenceTransformer to create embeddings for question-answer pairs.
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers")
    sys.exit(1)


# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_qa_pairs(json_path: str) -> List[Dict]:
    """
    Load Q&A pairs from JSON file.
    
    Args:
        json_path: Path to Q&A pairs JSON file
        
    Returns:
        List of Q&A pair dictionaries
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Q&A pairs file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"✓ Loaded {len(qa_pairs)} Q&A pairs from {json_path}")
    return qa_pairs


def initialize_model(model_name: str) -> SentenceTransformer:
    """
    Initialize SentenceTransformer model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"\nInitializing SentenceTransformer model: {model_name}")
    print("(First run may take a few minutes to download the model)")
    
    try:
        model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)


def create_text_for_embedding(question: str, answer: str) -> str:
    """
    Combine question and answer for embedding generation.
    
    Args:
        question: Question text
        answer: Answer text
        
    Returns:
        Combined text for embedding
    """
    return f"{question} {answer}"


def generate_embeddings(model: SentenceTransformer,
                       qa_pairs: List[Dict],
                       batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings for question-answer pairs.
    
    Args:
        model: SentenceTransformer model
        qa_pairs: List of Q&A pair dictionaries
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (embeddings array, list of combined texts)
    """
    print(f"\nGenerating embeddings for {len(qa_pairs)} Q&A pairs...")
    
    # Prepare texts for embedding
    texts = []
    for pair in qa_pairs:
        combined_text = create_text_for_embedding(
            pair['question'],
            pair['answer']
        )
        texts.append(combined_text)
    
    # Generate embeddings with progress display
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        
        current_batch = (batch_idx // batch_size) + 1
        progress_pct = (batch_idx + len(batch_texts)) / len(texts) * 100
        print(f"  Progress: Batch {current_batch}/{total_batches} ({progress_pct:.1f}%)")
    
    embeddings_array = np.array(embeddings)
    print(f"✓ Generated embeddings with shape: {embeddings_array.shape}")
    
    return embeddings_array, texts


def create_embedding_store(qa_pairs: List[Dict],
                          embeddings: np.ndarray,
                          texts: List[str]) -> Dict:
    """
    Create a dictionary storing embeddings with metadata.
    
    Args:
        qa_pairs: Original Q&A pairs with metadata
        embeddings: Generated embeddings array
        texts: Combined texts used for embedding
        
    Returns:
        Dictionary with embeddings and metadata
    """
    embedding_store = {
        'embeddings': embeddings,
        'metadata': [],
        'model_name': MODEL_NAME,
        'total_pairs': len(qa_pairs)
    }
    
    for idx, (pair, embedding, text) in enumerate(zip(qa_pairs, embeddings, texts)):
        metadata_entry = {
            'id': pair.get('id', idx),
            'question': pair['question'],
            'answer': pair['answer'],
            'combined_text': text,
            'embedding_index': idx,
            'source': pair.get('metadata', {}).get('source', 'unknown'),
            'processed_at': pair.get('metadata', {}).get('processed_at', '')
        }
        embedding_store['metadata'].append(metadata_entry)
    
    print(f"✓ Created embedding store with {len(embedding_store['metadata'])} entries")
    return embedding_store


def save_embeddings(embedding_store: Dict, output_path: str) -> None:
    """
    Save embeddings and metadata to pickle file.
    
    Args:
        embedding_store: Dictionary with embeddings and metadata
        output_path: Path to save pickle file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving embeddings to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_store, f)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Embeddings saved successfully ({file_size_mb:.2f} MB)")


def load_embeddings(pickle_path: str) -> Dict:
    """
    Load embeddings from pickle file.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        Dictionary with embeddings and metadata
        
    Raises:
        FileNotFoundError: If pickle file doesn't exist
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Embeddings file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        embedding_store = pickle.load(f)
    
    return embedding_store


def generate_embeddings_pipeline(qa_json_path: str,
                                output_pkl_path: str,
                                model_name: str = MODEL_NAME) -> Dict:
    """
    Complete embedding generation pipeline.
    
    Args:
        qa_json_path: Path to Q&A pairs JSON file
        output_pkl_path: Path to save embeddings pickle file
        model_name: Name of SentenceTransformer model to use
        
    Returns:
        Dictionary with embedding store and statistics
    """
    print("=" * 70)
    print("EMBEDDING GENERATION PIPELINE")
    print("=" * 70)
    
    # Load Q&A pairs
    qa_pairs = load_qa_pairs(qa_json_path)
    
    # Initialize model
    model = initialize_model(model_name)
    
    # Generate embeddings
    embeddings, texts = generate_embeddings(model, qa_pairs)
    
    # Create embedding store
    embedding_store = create_embedding_store(qa_pairs, embeddings, texts)
    
    # Save embeddings
    save_embeddings(embedding_store, output_pkl_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Total Q&A pairs processed: {len(qa_pairs)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output file: {output_pkl_path}")
    print("=" * 70)
    
    return embedding_store


def main():
    """Main entry point for the embedding generation script."""
    # Set file paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa_json_path = os.path.join(project_root, 'data', 'kcc_qa_pairs.json')
    output_pkl_path = os.path.join(project_root, 'embeddings', 'kcc_embeddings.pkl')
    
    try:
        generate_embeddings_pipeline(
            qa_json_path=qa_json_path,
            output_pkl_path=output_pkl_path,
            model_name=MODEL_NAME
        )
        print("\n✓ Embedding generation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to run preprocessing first:")
        print("  python services/preprocess_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
