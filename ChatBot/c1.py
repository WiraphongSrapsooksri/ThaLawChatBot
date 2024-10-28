# ChatBot.py
import streamlit as st
import openai
from pathlib import Path
import pickle
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

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

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for related legal sections."""
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

class LegalChatBot:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.search_engine = self.initialize_search_engine()
        self.conversation_history = []
        self.total_tokens_used = {'input': 0, 'output': 0}

    def initialize_search_engine(self):
        base_dir = Path("../DataConverter")
        paths = {
            'model_path': base_dir / 'legal_word2vec.model',
            'embeddings_path': base_dir / 'legal_embeddings.pkl'
        }
        
        engine = LegalSearchEngine()
        engine.load_model(str(paths['model_path']), str(paths['embeddings_path']))
        return engine

    def format_legal_references(self, search_results: List[Dict]) -> str:
        """Format legal references for the prompt."""
        references = []
        for result in search_results:
            ref = f"มาตรา {result['section_number']}: {result['content']}"
            if result['reference']:
                ref += f"\nอ้างอิง: {result['reference'].get('law_type', '')} {result['reference'].get('law_number', '')}"
            references.append(ref)
        return "\n\n".join(references)

    def get_chatgpt_response(self, query: str, legal_refs: str) -> str:
        """Get response from ChatGPT with legal context."""
        system_prompt = """คุณเป็นผู้เชี่ยวชาญด้านกฎหมายไทย ให้คำปรึกษาโดยอ้างอิงจากบทบัญญัติกฎหมายที่เกี่ยวข้อง 
        ตอบคำถามโดยใช้ภาษาที่เข้าใจง่าย และอ้างอิงมาตราที่เกี่ยวข้องเสมอ"""
        
        # Prepare conversation history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"บทบัญญัติกฎหมายที่เกี่ยวข้อง:\n{legal_refs}"}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message['content']
            input_tokens = response['usage']['prompt_tokens']
            output_tokens = response['usage']['completion_tokens']
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Update token usage
            self.total_tokens_used['input'] += input_tokens
            self.total_tokens_used['output'] += output_tokens
            
            # Keep only last 5 exchanges to maintain context without overloading
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
            return answer, input_tokens, output_tokens
            
        except Exception as e:
            return f"ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {str(e)}", 0, 0

    def process_query(self, query: str) -> str:
        """Process user query and return response."""
        # Search for relevant legal sections
        search_results = self.search_engine.search(query)
        
        # Format legal references
        legal_refs = self.format_legal_references(search_results)
        
        # Get ChatGPT response
        response, input_tokens, output_tokens = self.get_chatgpt_response(query, legal_refs)
        
        return response, search_results, input_tokens, output_tokens

# def get_openai_api_key():
#     """Get OpenAI API key from various sources."""
#     current_dir = Path(__file__).parent
    
#     # 1. Try to get from local secrets.toml in the same directory as ChatBot.py
#     try:
#         secrets_path = current_dir / "secrets.toml"
#         if secrets_path.exists():
#             with open(secrets_path, 'r', encoding='utf-8') as f:
#                 # Parse TOML file
#                 import toml
#                 secrets = toml.load(f)
#                 if 'OPENAI_API_KEY' in secrets:
#                     return secrets['OPENAI_API_KEY']
#     except Exception as e:
#         st.sidebar.warning(f"Could not load local secrets: {str(e)}")

#     # 2. Try to get from environment variable
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if api_key:
#         return api_key
    
#     return None

def get_openai_api_key():
    """Get OpenAI API key from environment variables."""
    # Load .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.sidebar.warning("OpenAI API key not found. Please set it in your .env file.")
        return None
        
    return api_key

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        # Get API key
        api_key = get_openai_api_key()
        
        # If no API key found, show input in sidebar
        if not api_key:
            with st.sidebar:
                st.markdown("## ⚙️ ตั้งค่า OpenAI API Key")
                
                st.info("""
                🔑 API Key ไม่พบในระบบ กรุณาตั้งค่าด้วยวิธีใดวิธีหนึ่ง:
                1. สร้างไฟล์ `secrets.toml` ใน folder ChatBot
                2. ตั้งค่า Environment Variable: OPENAI_API_KEY
                3. ใส่ API Key ในช่องด้านล่าง
                """)
                
                api_key = st.text_input(
                    "กรอก OpenAI API Key:",
                    type="password",
                    help="รับ API key ได้ที่ https://platform.openai.com/account/api-keys"
                )

                col1, col2 = st.columns(2)
                save_to_file = col1.checkbox("บันทึกลงไฟล์ secrets.toml", True)
                
                if col2.button("💾 บันทึก API Key", use_container_width=True):
                    if api_key:
                        if save_to_file:
                            try:
                                current_dir = Path(__file__).parent
                                secrets_path = current_dir / "secrets.toml"
                                
                                import toml
                                with open(secrets_path, 'w', encoding='utf-8') as f:
                                    toml.dump({'OPENAI_API_KEY': api_key}, f)
                                st.success("✅ บันทึก API Key ลงในไฟล์ secrets.toml แล้ว!")
                            except Exception as e:
                                st.error(f"❌ ไม่สามารถบันทึกไฟล์ได้: {str(e)}")
                        
                        st.session_state.api_key = api_key
                        st.success("✅ บันทึก API Key ในเซสชันแล้ว!")
                        st.rerun()
                    else:
                        st.error("❌ กรุณากรอก API Key")
                        return
        
        if api_key:
            try:
                st.session_state.chatbot = LegalChatBot(api_key)
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น ChatBot: {str(e)}")
                return

def display_chat_message(role: str, content: str, refs: List[Dict] = None):
    """Display chat message with proper formatting."""
    if role == "user":
        st.markdown(f'<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 5px 0;">'
                   f'<strong>คุณ:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;">'
                   f'<strong>AI:</strong><br>{content}</div>', unsafe_allow_html=True)
        if refs:
            with st.expander("ดูมาตราที่เกี่ยวข้อง"):
                for ref in refs:
                    st.markdown(f"""
                    **มาตรา {ref['section_number']}**
                    Similarity: {ref['similarity']:.4f}
                    ```
                    {ref['content']}
                    ```
                    """)

def display_token_usage(input_tokens: int, output_tokens: int, total_tokens: Dict[str, int]):
    """Display token usage information."""
    st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;">
        <strong>🔄 Token Usage:</strong><br>
        - Input Tokens Used: {input_tokens}<br>
        - Output Tokens Used: {output_tokens}<br>
        - Total Input Tokens Used: {total_tokens['input']}<br>
        - Total Output Tokens Used: {total_tokens['output']}<br>
        - Total Tokens Used: {total_tokens['input'] + total_tokens['output']}
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Thai Legal ChatBot",
        page_icon="⚖️",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .chat-container { 
            margin-bottom: 20px; 
        }
        .reference-text { 
            font-size: 0.8em; 
            color: #666; 
        }
        .stButton > button {
            width: 100%;
        }
        .main-title {
            text-align: center;
            padding: 1rem;
            color: #1E88E5;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-box.ready {
            background-color: #E8F5E9;
            border: 1px solid #4CAF50;
        }
        .status-box.not-ready {
            background-color: #FFEBEE;
            border: 1px solid #F44336;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🤖 ระบบให้คำปรึกษากฎหมายอัตโนมัติ</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()

    # Show system status
    with st.sidebar:
        st.markdown("### 📊 สถานะระบบ")
        if 'chatbot' in st.session_state:
            st.markdown("""
            <div class="status-box ready">
                ✅ ระบบพร้อมใช้งาน<br>
                💬 สามารถสอบถามได้ตามต้องการ
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box not-ready">
                ❌ ระบบยังไม่พร้อมใช้งาน<br>
                ⚙️ กรุณาตั้งค่า API Key
            </div>
            """, unsafe_allow_html=True)

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("refs")
            )

    # Chat input
    with st.container():
        query = st.chat_input(
            "💭 พิมพ์คำถามของคุณที่นี่...",
            disabled='chatbot' not in st.session_state
        )
        
        if query:
            # Display user message
            display_chat_message("user", query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Get chatbot response
            try:
                with st.spinner("🔍 กำลังค้นหาข้อมูลและวิเคราะห์..."):
                    response, refs, input_tokens, output_tokens = st.session_state.chatbot.process_query(query)
                    
                # Display assistant response
                display_chat_message("assistant", response, refs)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "refs": refs
                })
                
                # Display token usage
                display_token_usage(input_tokens, output_tokens, st.session_state.chatbot.total_tokens_used)
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ การจัดการ")
        if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
            st.session_state.messages = []
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.conversation_history = []
            st.rerun()
        
        if st.button("⚠️ ล้างการตั้งค่าทั้งหมด", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
