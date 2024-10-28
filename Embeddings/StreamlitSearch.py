# app.py
import streamlit as st
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from typing import List, Dict
import pandas as pd

class LegalSearchEngine:
    def __init__(self):
        self.word2vec_model = None
        self.document_embeddings = {}
        self.section_data = {}
        
    def load_model(self, model_path: str, embeddings_path: str):
        """Load the saved model and embeddings."""
        self.word2vec_model = Word2Vec.load(model_path)
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.document_embeddings = data['embeddings']
            self.section_data = data['section_data']

    def preprocess_query(self, query: str) -> list:
        text = normalize(query)
        tokens = word_tokenize(text, engine='newmm')
        return [token for token in tokens if token.strip() and not token.isnumeric()]

    def get_query_embedding(self, tokens: list) -> np.ndarray:
        word_vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.word2vec_model.vector_size)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = self.preprocess_query(query)
        query_embedding = self.get_query_embedding(query_tokens)
        
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
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

def get_model_paths():
    base_dir = Path("../DataConverter")
    return {
        'model_path': base_dir / 'legal_word2vec.model',
        'embeddings_path': base_dir / 'legal_embeddings.pkl'
    }

def format_content(content: str, max_length: int = 300) -> str:
    """Format content with expandable text if too long."""
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

def initialize_search_engine():
    """Initialize and load the search engine."""
    try:
        paths = get_model_paths()
        engine = LegalSearchEngine()
        engine.load_model(
            str(paths['model_path']),
            str(paths['embeddings_path'])
        )
        return engine
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Thai Legal Search Engine",
        page_icon="⚖️",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:24px !important;
            font-weight: bold;
        }
        .result-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        .similarity-score {
            color: #0066cc;
            font-weight: bold;
        }
        .reference-text {
            color: #666;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="big-font">🔍 ระบบค้นหากฎหมายอัจฉริยะ</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize search engine
    engine = initialize_search_engine()
    if not engine:
        st.error("ไม่สามารถโหลดระบบค้นหาได้ กรุณาตรวจสอบว่ามีไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง")
        return

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔎 ค้นหากฎหมาย", placeholder="พิมพ์คำค้นหาที่นี่...")
    with col2:
        top_k = st.number_input("จำนวนผลลัพธ์", min_value=1, max_value=20, value=5)

    # Search button
    if st.button("ค้นหา", type="primary"):
        if query:
            with st.spinner("กำลังค้นหา..."):
                results = engine.search(query, top_k=top_k)
            
            st.markdown(f"### พบ {len(results)} ผลการค้นหาสำหรับ: '{query}'")
            
            for i, result in enumerate(results, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>ผลการค้นหาที่ {i}: มาตรา {result['section_number']}</h4>
                        <p class="similarity-score">ค่าความเกี่ยวข้อง: {result['similarity']:.4f}</p>
                        <p>{format_content(result['content'])}</p>
                    """, unsafe_allow_html=True)
                    
                    if result['reference']:
                        ref = result['reference']
                        st.markdown(f"""
                        <p class="reference-text">อ้างอิง: {ref.get('law_type', '')} {ref.get('law_number', '')}
                        {f" (รายการที่ {ref['index']})" if 'index' in ref else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        else:
            st.warning("กรุณากรอกคำค้นหา")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
        <p>พัฒนาโดยใช้ Word Embeddings และ Semantic Search</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()