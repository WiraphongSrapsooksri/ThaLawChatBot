# Search.py
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from typing import List, Dict

class LegalSearchEngine:
    def __init__(self):
        self.word2vec_model = None
        self.document_embeddings = {}
        self.section_data = {}
        
    def load_model(self, model_path: str, embeddings_path: str):
        """Load the saved model and embeddings."""
        print(f"Loading Word2Vec model from {model_path}")
        self.word2vec_model = Word2Vec.load(model_path)
        
        print(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.document_embeddings = data['embeddings']
            self.section_data = data['section_data']

    def preprocess_query(self, query: str) -> list:
        """Preprocess search query."""
        text = normalize(query)
        tokens = word_tokenize(text, engine='newmm')
        return [token for token in tokens if token.strip() and not token.isnumeric()]

    def get_query_embedding(self, tokens: list) -> np.ndarray:
        """Convert query tokens to embedding."""
        word_vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.word2vec_model.vector_size)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for related legal sections."""
        # Preprocess query
        query_tokens = self.preprocess_query(query)
        query_embedding = self.get_query_embedding(query_tokens)
        
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
        
        # Sort by similarity and return top_k results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

def get_model_paths():
    """Get paths to model files."""
    base_dir = Path("../DataConverter")
    return {
        'model_path': base_dir / 'legal_word2vec.model',
        'embeddings_path': base_dir / 'legal_embeddings.pkl'
    }

def format_search_result(result: Dict) -> str:
    """Format a single search result for display."""
    output = []
    output.append(f"\nSection {result['section_number']}")
    output.append(f"Similarity: {result['similarity']:.4f}")
    output.append(f"Content: {result['content'][:200]}...")
    
    if result['reference']:
        ref = result['reference']
        ref_text = f"Reference: {ref.get('law_type', '')} {ref.get('law_number', '')}"
        if 'index' in ref:
            ref_text += f" (รายการที่ {ref['index']})"
        output.append(ref_text)
    
    return '\n'.join(output)

def main():
    # Example usage
    try:
        # Initialize search engine
        search_engine = LegalSearchEngine()
        paths = get_model_paths()
        
        # Load saved model
        search_engine.load_model(
            str(paths['model_path']),
            str(paths['embeddings_path'])
        )
        
        # Interactive search
        while True:
            query = input("\nEnter search query (or 'q' to quit): ")
            if query.lower() == 'q':
                break
                
            results = search_engine.search(query)
            print(f"\nFound {len(results)} related sections for: {query}")
            
            for result in results:
                print(format_search_result(result))
                
    except FileNotFoundError:
        print("Error: Model files not found. Please run Embedding.py first to generate the model.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()