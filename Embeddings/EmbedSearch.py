import json
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple
import pickle
import os
from pathlib import Path

class LegalEmbeddingSearch:
    def __init__(self):
        self.word2vec_model = None
        self.document_embeddings = {}
        self.section_data = {}
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess Thai text for embedding."""
        # Normalize text
        text = normalize(text)
        
        # Tokenize using Thai word tokenization
        tokens = word_tokenize(text, engine='newmm')
        
        # Remove special characters and numbers
        tokens = [token for token in tokens if token.strip() and not token.isnumeric()]
        
        return tokens

    def train_word2vec(self, json_file: str, vector_size: int = 100):
        """Train Word2Vec model on legal documents."""
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            legal_data = json.load(f)
        
        # Prepare sentences for training
        sentences = []
        for section in legal_data['sections']:
            # Process main content
            tokens = self.preprocess_text(section['content'])
            sentences.append(tokens)
            
            # Process subsections
            for subsection in section['subsections']:
                tokens = self.preprocess_text(subsection['content'])
                sentences.append(tokens)
                
            # Store section data for later use
            self.section_data[section['section_number']] = {
                'content': section['content'],
                'reference': section.get('reference', {})
            }
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Generate document embeddings
        self.generate_document_embeddings()

    def generate_document_embeddings(self):
        """Generate embeddings for each section and subsection."""
        for section_number, data in self.section_data.items():
            # Generate embedding for main content
            tokens = self.preprocess_text(data['content'])
            if tokens:
                embedding = self.get_text_embedding(tokens)
                self.document_embeddings[section_number] = {
                    'embedding': embedding,
                    'content': data['content'],
                    'reference': data['reference']
                }

    def get_text_embedding(self, tokens: List[str]) -> np.ndarray:
        """Get embedding for a list of tokens."""
        word_vectors = []
        for token in tokens:
            if token in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[token])
        
        if word_vectors:
            # Average word vectors to get document vector
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.word2vec_model.vector_size)

    def find_related_sections(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find sections related to the query using cosine similarity."""
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        query_embedding = self.get_text_embedding(query_tokens)
        
        # Calculate similarities
        similarities = []
        for section_number, data in self.document_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                data['embedding'].reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'section_number': section_number,
                'content': data['content'],
                'similarity': similarity,
                'reference': data['reference']
            })
        
        # Sort by similarity and get top_k results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def save_model(self, model_path: str, embeddings_path: str):
        """Save the model and embeddings."""
        self.word2vec_model.save(model_path)
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.document_embeddings,
                'section_data': self.section_data
            }, f)

    def load_model(self, model_path: str, embeddings_path: str):
        """Load the model and embeddings."""
        self.word2vec_model = Word2Vec.load(model_path)
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.document_embeddings = data['embeddings']
            self.section_data = data['section_data']

def setup_paths():
    """Setup directory paths for data storage"""
    # Define base directory
    base_dir = Path("../DataConverter")
    
    # Create directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'json_path': base_dir / 'legal_document.json',
        'model_path': base_dir / 'legal_word2vec.model',
        'embeddings_path': base_dir / 'legal_embeddings.pkl'
    }

def main():
    # Initialize the search system
    search_system = LegalEmbeddingSearch()
    
    # Setup paths
    paths = setup_paths()
    
    print("Starting the process...")
    
    try:
        # Train on your legal document JSON
        print("Training Word2Vec model...")
        search_system.train_word2vec(str(paths['json_path']))
        
        # Save the model and embeddings
        print("Saving model and embeddings...")
        search_system.save_model(
            str(paths['model_path']),
            str(paths['embeddings_path'])
        )
        
        # Now we can load the model (optional in this case since we just trained it)
        print("Loading saved model...")
        search_system.load_model(
            str(paths['model_path']),
            str(paths['embeddings_path'])
        )
        
        # Example search
        print("\nTesting search functionality...")
        query = "การเช่าซื้อที่ดิน"
        results = search_system.find_related_sections(query)
        
        print(f"\nQuery: {query}")
        print("\nRelated Sections:")
        for result in results:
            print(f"\nSection {result['section_number']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            if result['reference']:
                print(f"Reference: {result['reference']}")
                
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()