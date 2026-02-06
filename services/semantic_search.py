"""
FAISS-based semantic search module for agricultural Q&A.
Builds vector index and provides similarity search functionality.
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Optional

try:
    import faiss
except ImportError:
    print("Error: faiss-cpu not installed. Please run: pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers")
    sys.exit(1)


# Model configuration - must match embedding generation
MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum query length to avoid noise
MIN_QUERY_LENGTH = 3


class SemanticSearch:
    """
    FAISS-based semantic search engine for Q&A pairs.
    Uses lazy loading to initialize model and index only once.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Name of SentenceTransformer model to use
        """
        # Configuration
        self.model_name = model_name
        
        # Lazy-loaded resources (initialized only once)
        self._model = None  # Will be loaded on first use
        self._index = None  # Will be loaded on first use
        self._metadata = None  # Will be loaded on first use
        self._index_path = None  # Cache the path for lazy loading
        self._meta_path = None  # Cache the path for lazy loading
        
        # Embedding dimension info
        self.embedding_dim = None
        
    # ============ Lazy Loading Properties ============
    
    @property
    def model(self):
        """Lazily load embedding model on first access."""
        if self._model is None:
            self._load_model_lazy()
        return self._model
    
    @property
    def index(self):
        """Lazily load FAISS index on first access."""
        if self._index is None and self._index_path is not None:
            self._load_index_lazy()
        return self._index
    
    @property
    def metadata(self):
        """Lazily load metadata on first access."""
        if self._metadata is None and self._meta_path is not None:
            self._load_metadata_lazy()
        return self._metadata
    
    # ============ Lazy Loading Methods ============
    
    def _load_model_lazy(self) -> None:
        """Lazy-load the embedding model on first use."""
        print(f"Initializing model: {self.model_name}")
        try:
            self._model = SentenceTransformer(self.model_name)
            print(f"✓ Model initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize model: {e}")
            sys.exit(1)
    
    def _load_index_lazy(self) -> None:
        """Lazy-load the FAISS index from disk."""
        if self._index_path is None:
            raise RuntimeError("Index path not set. Call load_index() first.")
        
        if not os.path.exists(self._index_path):
            raise FileNotFoundError(f"FAISS index file not found: {self._index_path}")
        
        print(f"Loading FAISS index from {self._index_path}...")
        self._index = faiss.read_index(self._index_path)
        self.embedding_dim = self._index.d
        print(f"✓ Index loaded ({self._index.ntotal} vectors)")
    
    def _load_metadata_lazy(self) -> None:
        """Lazy-load the metadata from disk."""
        if self._meta_path is None:
            raise RuntimeError("Metadata path not set. Call load_index() first.")
        
        if not os.path.exists(self._meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self._meta_path}")
        
        print(f"Loading metadata from {self._meta_path}...")
        with open(self._meta_path, 'rb') as f:
            self._metadata = pickle.load(f)
        print(f"✓ Metadata loaded ({len(self._metadata)} entries)")
        
    # ============ Index Management ============
    
    def load_index(self, index_path: str, meta_path: str) -> None:
        """
        Register FAISS index and metadata file paths for lazy loading.
        Files are loaded on first search, not during initialization.
        
        Args:
            index_path: Path to FAISS index file
            meta_path: Path to metadata pickle file
            
        Raises:
            FileNotFoundError: If files don't exist
        """
        # Validate files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        # Store paths for lazy loading later
        self._index_path = index_path
        self._meta_path = meta_path
        print(f"✓ Index paths registered (will load on first search)")
    
    def load_embeddings(self, pkl_path: str) -> None:
        """
        Load embeddings from pickle file (for building index).
        
        Args:
            pkl_path: Path to embeddings pickle file
            
        Raises:
            FileNotFoundError: If pickle file doesn't exist
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Embeddings file not found: {pkl_path}")
        
        print(f"Loading embeddings from {pkl_path}...")
        
        with open(pkl_path, 'rb') as f:
            embedding_store = pickle.load(f)
        
        embeddings = embedding_store['embeddings']
        self._metadata = embedding_store['metadata']
        self.embedding_dim = embeddings.shape[1]
        
        print(f"✓ Loaded {len(embeddings)} embeddings with dimension {self.embedding_dim}")
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS IndexFlatL2 from embeddings and store internally.
        
        Args:
            embeddings: Numpy array of embeddings
        """
        print(f"\nBuilding FAISS IndexFlatL2...")
        
        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Create IndexFlatL2 (L2 distance metric for similarity search)
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)
        self.embedding_dim = embeddings.shape[1]
        
        print(f"✓ Built index with {self._index.ntotal} vectors")
    
    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save index
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        print(f"Saving FAISS index to {index_path}...")
        faiss.write_index(self._index, index_path)
        
        file_size_mb = os.path.getsize(index_path) / (1024 * 1024)
        print(f"✓ Index saved successfully ({file_size_mb:.2f} MB)")
    
    def save_metadata(self, meta_path: str) -> None:
        """
        Save metadata to pickle file.
        
        Args:
            meta_path: Path to save metadata
        """
        if self._metadata is None:
            raise RuntimeError("Metadata not available. Load embeddings first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        
        print(f"Saving metadata to {meta_path}...")
        
        with open(meta_path, 'wb') as f:
            pickle.dump(self._metadata, f)
        
        file_size_kb = os.path.getsize(meta_path) / 1024
        print(f"✓ Metadata saved successfully ({file_size_kb:.2f} KB)")
    
    # ============ Query Processing ============
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text to embedding using the cached model.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as float32 numpy array
        """
        # Model is lazily loaded on first access via property
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.astype('float32')
    
    def search_top_k(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for top-k most similar Q&A pairs and remove duplicates.
        
        Args:
            query: Query text (user's agricultural question)
            k: Number of top results to return
            
        Returns:
            List of result dictionaries with keys:
            - question: Original question text
            - answer: Answer text
            - score: Similarity score (0-1)
            
        Raises:
            RuntimeError: If index or model not loaded
            ValueError: If query is too short or invalid
        """
        # ========== Input Validation ==========
        # Check that index and metadata are available
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        # Clean query: strip whitespace and convert to string
        query = str(query).strip()
        
        # Validate query length to avoid noise
        if len(query) < MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query '{query}' is too short (minimum {MIN_QUERY_LENGTH} characters)"
            )
        
        # ========== Query Encoding ==========
        # Encode query using the embedding model
        query_embedding = self.encode_query(query)
        
        # ========== FAISS Search ==========
        # Reshape to (1, embedding_dim) as FAISS expects batch format
        # Search for k*2 results to account for duplicates we'll remove
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(k * 2, self.index.ntotal)  # Don't request more than available
        )
        
        # ========== Build Results with Deduplication ==========
        # Track seen answers to avoid duplicates
        seen_answers = set()
        results = []
        
        # Process each search result from FAISS
        for distance, idx in zip(distances[0], indices):
            # -1 index means invalid result (no match found)
            if idx < 0:
                continue
            
            # Get metadata for this result
            idx = int(idx)
            metadata = self.metadata[idx]
            
            # Normalize answer for duplicate detection (lowercase, stripped)
            answer_normalized = metadata['answer'].lower().strip()
            
            # Skip if we've already seen this answer
            if answer_normalized in seen_answers:
                continue
            
            # Convert L2 distance to similarity score (0-1 scale)
            # Lower distance = higher similarity
            similarity_score = 1 / (1 + distance)
            
            # Create result dictionary with requested keys
            result = {
                'question': metadata['question'],
                'answer': metadata['answer'],
                'score': float(similarity_score)
            }
            
            results.append(result)
            seen_answers.add(answer_normalized)
            
            # Stop when we have k unique results
            if len(results) >= k:
                break
        
        return results
    
    def get_answers(self, query: str, top_k: int = 5) -> Dict:
        """
        High-level function to get formatted answers for a query.
        Performs FAISS search and prepares output for both display and LLM.
        
        Args:
            query: User's agricultural question
            top_k: Number of top results to return
            
        Returns:
            Dictionary with:
            - offline_answers: List of formatted answer strings (numbered bullets)
            - llm_context: Context string for use with LLM
            
        Example:
            result = searcher.get_answers("How to prevent crop disease?", top_k=5)
            
            # Display offline answers
            for answer in result['offline_answers']:
                print(answer)
            
            # Use context with LLM
            llm_response = llm_client.generate_answer(
                user_query="How to prevent crop disease?",
                context_answers=result['llm_context']
            )
        """
        # Search for top results
        search_results = self.search_top_k(query, k=top_k)
        
        # Format offline answers as numbered bullets
        offline_answers = []
        for i, result in enumerate(search_results, 1):
            # Create numbered bullet format
            formatted_answer = (
                f"{i}. {result['answer']}\n"
                f"   (From: \"{result['question']}\" | Relevance: {result['score']:.0%})"
            )
            offline_answers.append(formatted_answer)
        
        # Build context string for LLM (shorter, cleaner format)
        llm_context_parts = []
        for i, result in enumerate(search_results, 1):
            context_item = (
                f"Reference {i}: {result['question']}\n"
                f"Answer: {result['answer']}"
            )
            llm_context_parts.append(context_item)
        
        llm_context = "\n\n".join(llm_context_parts)
        
        return {
            'offline_answers': offline_answers,
            'llm_context': llm_context,
            'search_results': search_results,
            'total_results': len(search_results)
        }
    
    # ============ Pipeline Methods ============
    
    def build_index_from_embeddings(self, embeddings_pkl_path: str,
                                   index_output_path: str,
                                   meta_output_path: str) -> None:
        """
        Complete pipeline: load embeddings, build index, save index and metadata.
        
        Args:
            embeddings_pkl_path: Path to embeddings pickle file
            index_output_path: Path to save FAISS index
            meta_output_path: Path to save metadata
        """
        # Load embeddings (populates _metadata)
        embeddings = self.load_embeddings(embeddings_pkl_path)
        
        # Build FAISS index (stores in _index)
        self.build_index(embeddings)
        
        # Save index and metadata to disk
        self.save_index(index_output_path)
        self.save_metadata(meta_output_path)
        
        print(f"\n✓ Vector store built successfully")
    
    def get_statistics(self) -> Dict:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'status': 'ready' if self._index is not None else 'not_initialized',
            'total_vectors': self._index.ntotal if self._index is not None else 0,
            'embedding_dimension': self.embedding_dim,
            'model': self.model_name,
            'metadata_entries': len(self._metadata) if self._metadata is not None else 0,
            'model_loaded': self._model is not None,
            'index_loaded': self._index is not None
        }


def build_vector_store(embeddings_pkl_path: str,
                      index_output_path: str,
                      meta_output_path: str) -> SemanticSearch:
    """
    Build vector store from embeddings (complete setup pipeline).
    
    Args:
        embeddings_pkl_path: Path to embeddings pickle file
        index_output_path: Path to save FAISS index
        meta_output_path: Path to save metadata
        
    Returns:
        Initialized SemanticSearch object with built index
    """
    print("=" * 70)
    print("BUILDING VECTOR STORE")
    print("=" * 70)
    
    # Create search engine and build index
    search_engine = SemanticSearch(model_name=MODEL_NAME)
    search_engine.build_index_from_embeddings(
        embeddings_pkl_path,
        index_output_path,
        meta_output_path
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("VECTOR STORE SUMMARY")
    print("=" * 70)
    stats = search_engine.get_statistics()
    print(f"Status: {stats['status']}")
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Model: {stats['model']}")
    print("=" * 70)
    
    return search_engine


def main():
    """Main entry point for building and testing vector store."""
    # Set file paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_pkl_path = os.path.join(project_root, 'embeddings', 'kcc_embeddings.pkl')
    index_output_path = os.path.join(project_root, 'vector_store', 'faiss.index')
    meta_output_path = os.path.join(project_root, 'vector_store', 'meta.pkl')
    
    try:
        # Build vector store
        search_engine = build_vector_store(
            embeddings_pkl_path,
            index_output_path,
            meta_output_path
        )
        print("\n✓ Vector store built successfully!")
        
        # ========== Demo Search ==========
        print("\n" + "=" * 70)
        print("DEMO SEARCH")
        print("=" * 70)
        
        # Sample query
        sample_query = "How to prevent crop disease?"
        print(f"\nQuery: '{sample_query}'")
        print(f"Searching for top 3 results with duplicate removal...\n")
        
        # Reload search engine to test lazy loading and search
        search_engine = SemanticSearch()
        search_engine.load_index(index_output_path, meta_output_path)
        results = search_engine.search_top_k(sample_query, k=3)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Question: {result['question']}")
            print(f"  Answer: {result['answer'][:100]}...")
            print(f"  Similarity Score: {result['score']:.1%}")
            print()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to run embedding generation first:")
        print("  python services/generate_embeddings.py")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
