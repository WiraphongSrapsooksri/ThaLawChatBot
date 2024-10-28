from docx import Document
from sentence_transformers import SentenceTransformer
import json
import re
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

class LegalDocumentProcessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize processor with embedding model
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.sections = []
            self.embeddings = None
            print("ระบบพร้อมใช้งาน - โมเดลโหลดสำเร็จ")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
            raise
        
    def read_docx(self, file_path: str) -> List[Dict]:
        """
        อ่านไฟล์ .docx และแยกเนื้อหาตามมาตรา
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ไม่พบไฟล์ที่ {file_path}")
                
            print(f"กำลังอ่านไฟล์: {file_path}")
            doc = Document(file_path)
            current_section = {
                'id': None,
                'title': '',
                'content': [],
                'subsections': []
            }
            sections = []
            
            # Pattern สำหรับจับมาตราและรายละเอียด
            section_pattern = re.compile(r'มาตรา\s+(\d+)')
            detail_pattern = re.compile(r'รายละเอียดคำที่พิมพ์\s+รายการที่\s+(\d+)/(\d+)')
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                    
                # ตรวจจับมาตราใหม่
                section_match = section_pattern.match(text)
                detail_match = detail_pattern.match(text)
                
                if section_match:
                    # บันทึกมาตราก่อนหน้า
                    if current_section['id']:
                        sections.append(current_section)
                    
                    # สร้างมาตราใหม่
                    section_id = section_match.group(1)
                    current_section = {
                        'id': f'มาตรา {section_id}',
                        'title': text,
                        'content': [text],
                        'subsections': []
                    }
                elif detail_match:
                    # จัดการกับรายละเอียดคำที่พิมพ์
                    detail_num = f"{detail_match.group(1)}/{detail_match.group(2)}"
                    if current_section['id']:
                        current_section['detail_number'] = detail_num
                else:
                    if current_section['id']:
                        current_section['content'].append(text)
                    else:
                        # เก็บส่วนหัวเอกสาร
                        sections.append({
                            'id': 'header',
                            'title': 'กฎหมายเช่าซื้อ',
                            'content': [text],
                            'subsections': []
                        })
            
            # บันทึกมาตราสุดท้าย
            if current_section['id']:
                sections.append(current_section)
                
            self.sections = sections
            print(f"อ่านข้อมูลสำเร็จ: พบ {len(sections)} มาตรา")
            return sections
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")
            raise

    def create_embeddings(self) -> Dict[str, np.ndarray]:
        """
        สร้าง embeddings สำหรับแต่ละส่วน
        """
        try:
            print("กำลังสร้าง embeddings...")
            embeddings = {}
            
            for section in self.sections:
                # สร้าง embedding สำหรับเนื้อหาทั้งมาตรา
                full_text = ' '.join(section['content'])
                section_embedding = self.model.encode(full_text)
                
                embeddings[section['id']] = {
                    'full_text': full_text,
                    'embedding': section_embedding
                }
                
                # สร้าง embedding สำหรับแต่ละย่อหน้า
                for i, paragraph in enumerate(section['content']):
                    if paragraph.strip():
                        para_embedding = self.model.encode(paragraph)
                        embeddings[f"{section['id']}_p{i}"] = {
                            'full_text': paragraph,
                            'embedding': para_embedding
                        }
            
            self.embeddings = embeddings
            print(f"สร้าง embeddings สำเร็จ: {len(embeddings)} รายการ")
            return embeddings
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการสร้าง embeddings: {str(e)}")
            raise

# ใช้งานกับไฟล์เป้าหมาย
def main():
    try:
        # สร้าง processor
        processor = LegalDocumentProcessor()
        
        # กำหนดพาธของไฟล์
        file_path = "Doc/กฎหมายเช่าซื้อ.docx"
        
        # อ่านไฟล์ .docx
        sections = processor.read_docx(file_path)
        
        # สร้าง embeddings
        embeddings = processor.create_embeddings()
        
        # บันทึกข้อมูล
        output_file = "legal_embeddings_rental.pkl"
        processor.save_processed_data(output_file)
        print(f"บันทึกข้อมูลลงไฟล์ {output_file} สำเร็จ")
        
        # ทดสอบการค้นหา
        test_queries = [
            "การเช่าซื้อคืออะไร",
            "สิทธิของผู้เช่าซื้อ",
            "หน้าที่ของผู้ให้เช่าซื้อ"
        ]
        
        print("\nทดสอบการค้นหา:")
        for query in test_queries:
            print(f"\nคำถาม: {query}")
            results = processor.find_similar_sections(query)
            for result in results:
                print(f"\nมาตรา: {result['section']}")
                print(f"ความเกี่ยวข้อง: {result['score']:.3f}")
                print(f"เนื้อหา: {result['text']}")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำงาน: {str(e)}")

if __name__ == "__main__":
    main()