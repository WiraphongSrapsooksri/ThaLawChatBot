import json
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Any
import os

class LegalDataConverter:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize converter with the specified model
        """
        self.model = SentenceTransformer(model_name)
        self.data = None
        self.vectors = {}
    
    def docx_to_json(self, file_path: str, output_json_path: str = 'law_data.json') -> Dict:
        """
        แปลงไฟล์ .docx เป็น JSON
        """
        doc = Document(file_path)
        legal_data = {
            "document_info": {
                "title": "กฎหมายเช่าซื้อ",
                "type": "พระราชบัญญัติ",
                "sections": []
            }
        }
        
        current_section = None
        section_pattern = re.compile(r'มาตรา\s+(\d+)')
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            section_match = section_pattern.match(text)
            
            if section_match:
                # บันทึกมาตราก่อนหน้า
                if current_section:
                    legal_data["document_info"]["sections"].append(current_section)
                
                # สร้างมาตราใหม่
                section_number = section_match.group(1)
                current_section = {
                    "section_number": f"มาตรา {section_number}",
                    "content": text,
                    "paragraphs": [text],
                    "referenced_sections": [],
                    "keywords": [],
                    "metadata": {
                        "is_current": True,
                        "last_updated": "2024",
                        "version": "1.0"
                    }
                }
            elif current_section:
                current_section["paragraphs"].append(text)
                current_section["content"] += f"\n{text}"
                
                # ตรวจหาการอ้างอิงมาตราอื่น
                ref_matches = re.finditer(r'มาตรา\s+(\d+)', text)
                for match in ref_matches:
                    ref_section = f"มาตรา {match.group(1)}"
                    if ref_section != current_section["section_number"]:
                        current_section["referenced_sections"].append(ref_section)
        
        # บันทึกมาตราสุดท้าย
        if current_section:
            legal_data["document_info"]["sections"].append(current_section)
        
        # บันทึก JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(legal_data, f, ensure_ascii=False, indent=2)
        
        self.data = legal_data
        return legal_data
    
    def create_vectors(self, json_data: Dict = None) -> Dict:
        """
        สร้าง vectors จากข้อมูล JSON
        """
        if json_data is None:
            json_data = self.data
        
        if json_data is None:
            raise ValueError("ไม่พบข้อมูล JSON สำหรับการสร้าง vectors")
        
        vectors = {
            "document_vectors": {
                "sections": {},
                "paragraphs": {},
                "combined": {}
            }
        }
        
        # สร้าง vectors สำหรับแต่ละมาตรา
        for section in json_data["document_info"]["sections"]:
            section_id = section["section_number"]
            
            # Vector สำหรับทั้งมาตรา
            section_vector = self.model.encode(section["content"])
            vectors["document_vectors"]["sections"][section_id] = {
                "vector": section_vector.tolist(),
                "text": section["content"]
            }
            
            # Vector สำหรับแต่ละย่อหน้า
            vectors["document_vectors"]["paragraphs"][section_id] = []
            for i, para in enumerate(section["paragraphs"]):
                para_vector = self.model.encode(para)
                vectors["document_vectors"]["paragraphs"][section_id].append({
                    "paragraph_index": i,
                    "vector": para_vector.tolist(),
                    "text": para
                })
        
        self.vectors = vectors
        return vectors
    
    def save_vectors(self, output_path: str = 'law_vectors.json'):
        """
        บันทึก vectors ลงไฟล์
        """
        if self.vectors:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f, ensure_ascii=False, indent=2)
    
    def find_similar_sections(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        ค้นหามาตราที่เกี่ยวข้องโดยใช้ vectors
        """
        query_vector = self.model.encode(query)
        similarities = {}
        
        for section_id, section_data in self.vectors["document_vectors"]["sections"].items():
            section_vector = np.array(section_data["vector"])
            similarity = np.dot(query_vector, section_vector)
            similarities[section_id] = {
                "score": float(similarity),
                "text": section_data["text"]
            }
        
        # เรียงลำดับผลลัพธ์
        sorted_results = sorted(similarities.items(), 
                              key=lambda x: x[1]["score"], 
                              reverse=True)[:top_k]
        
        return [{"section": k, **v} for k, v in sorted_results]

def main():
    try:
        # สร้าง converter
        converter = LegalDataConverter()
        
        # แปลง DOCX เป็น JSON
        print("กำลังแปลงไฟล์ DOCX เป็น JSON...")
        json_data = converter.docx_to_json("Doc/กฎหมายเช่าซื้อ.docx", "rental_law.json")
        print("แปลงเป็น JSON สำเร็จ")
        
        # สร้าง vectors
        print("\nกำลังสร้าง vectors...")
        vectors = converter.create_vectors(json_data)
        converter.save_vectors("rental_law_vectors.json")
        print("สร้างและบันทึก vectors สำเร็จ")
        
        # ทดสอบการค้นหา
        print("\nทดสอบการค้นหา:")
        test_queries = [
            "การเช่าซื้อคืออะไร",
            "สิทธิของผู้เช่าซื้อ",
            "การผิดนัดชำระค่าเช่าซื้อ"
        ]
        
        for query in test_queries:
            print(f"\nคำถาม: {query}")
            results = converter.find_similar_sections(query)
            for result in results:
                print(f"\nมาตรา: {result['section']}")
                print(f"ความเกี่ยวข้อง: {result['score']:.3f}")
                print(f"เนื้อหา: {result['text'][:200]}...")  # แสดงแค่ 200 ตัวอักษรแรก
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main()