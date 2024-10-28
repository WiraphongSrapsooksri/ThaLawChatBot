# Embedding.py
import json
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from gensim.models import Word2Vec
import pickle
from pathlib import Path

class LegalEmbeddingGenerator:
    def __init__(self):
        self.word2vec_model = None
        self.document_embeddings = {}
        self.section_data = {}
        
    def preprocess_text(self, text: str) -> list:
        """Preprocess Thai text for embedding."""
        text = normalize(text)
        tokens = word_tokenize(text, engine='newmm')
        tokens = [token for token in tokens if token.strip() and not token.isnumeric()]
        return tokens

    def train_word2vec(self, json_file: str, vector_size: int = 100):
        """Train Word2Vec model on legal documents."""
        print(f"Loading JSON data from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            legal_data = json.load(f)
        
        print("Preparing sentences for training...")
        sentences = []
        for section in legal_data['sections']:
            # Process main content
            tokens = self.preprocess_text(section['content'])
            sentences.append(tokens)
            
            # Process subsections
            for subsection in section['subsections']:
                tokens = self.preprocess_text(subsection['content'])
                sentences.append(tokens)
                
            # Store section data
            self.section_data[section['section_number']] = {
                'content': section['content'],
                'reference': section.get('reference', {})
            }
        
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        
        print("Generating document embeddings...")
        self._generate_document_embeddings()

    def _generate_document_embeddings(self):
        """Generate embeddings for each section."""
        for section_number, data in self.section_data.items():
            tokens = self.preprocess_text(data['content'])
            if tokens:
                embedding = self._get_text_embedding(tokens)
                self.document_embeddings[section_number] = {
                    'embedding': embedding,
                    'content': data['content'],
                    'reference': data['reference']
                }

    def _get_text_embedding(self, tokens: list) -> np.ndarray:
        """Get embedding for a list of tokens."""
        word_vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.word2vec_model.vector_size)

    def save_model(self, model_path: str, embeddings_path: str):
        """Save the model and embeddings."""
        print(f"Saving Word2Vec model to {model_path}")
        self.word2vec_model.save(model_path)
        
        print(f"Saving embeddings to {embeddings_path}")
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.document_embeddings,
                'section_data': self.section_data
            }, f)

def setup_paths():
    """Setup directory paths for data storage"""
    base_dir = Path("../DataConverter")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'json_path': base_dir / 'legal_document.json',
        'model_path': base_dir / 'legal_word2vec.model',
        'embeddings_path': base_dir / 'legal_embeddings.pkl'
    }

def main():
    print("Initializing embedding generator...")
    generator = LegalEmbeddingGenerator()
    paths = setup_paths()
    
    try:
        # Generate and save embeddings
        generator.train_word2vec(str(paths['json_path']))
        generator.save_model(
            str(paths['model_path']),
            str(paths['embeddings_path'])
        )
        print("Embedding generation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()