from docx import Document
import json
import re
from typing import Dict, List, Any

class LawDocumentConverter:
    def docx_to_json(self, docx_path: str, output_path: str) -> Dict[str, Any]:
        """
        แปลงไฟล์ DOCX เป็น JSON ที่มีโครงสร้างตามมาตรากฎหมาย
        
        Args:
            docx_path (str): path ของไฟล์ DOCX ที่ต้องการแปลง
            output_path (str): path ที่ต้องการบันทึกไฟล์ JSON
        
        Returns:
            Dict ที่เก็บเนื้อหากฎหมายในรูปแบบที่มีโครงสร้าง
        """
        doc = Document(docx_path)
        law_content = {
            "title": "",
            "sections": []
        }
        
        current_section = None
        
        # วนลูปอ่านแต่ละย่อหน้าในเอกสาร
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # ตรวจจับหัวข้อมาตรา (มาตรา X)
            if re.match(r'^มาตรา \d+', text):
                if current_section:
                    law_content["sections"].append(current_section)
                
                section_number = re.search(r'\d+', text).group()
                current_section = {
                    "section_number": section_number,
                    "title": text,  # เก็บหัวข้อมาตราเต็ม
                    "content": [],  # เก็บเนื้อหาเป็น list
                    "raw_text": text  # เก็บข้อความดิบทั้งหมดของมาตรานี้
                }
            
            # ตรวจจับชื่อกฎหมาย
            elif text.startswith("กฎหมาย"):
                law_content["title"] = text
            
            # เพิ่มเนื้อหาเข้าไปในมาตราปัจจุบัน
            elif current_section:
                current_section["content"].append(text)
                current_section["raw_text"] = current_section["raw_text"] + "\n" + text
        
        # เพิ่มมาตราสุดท้าย
        if current_section:
            law_content["sections"].append(current_section)
        
        # บันทึกไฟล์ JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(law_content, f, ensure_ascii=False, indent=2)
        
        return law_content

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    converter = LawDocumentConverter()
    json_data = converter.docx_to_json("Doc\กฎหมายเช่าซื้อ.docx", "law_content.json")
    
    # แสดงตัวอย่างผลลัพธ์
    print(f"ชื่อกฎหมาย: {json_data['title']}")
    print(f"จำนวนมาตราทั้งหมด: {len(json_data['sections'])}")
    
    # แสดงตัวอย่างมาตราแรก
    if json_data['sections']:
        first_section = json_data['sections'][0]
        print(f"\nตัวอย่างมาตราแรก:")
        print(f"เลขมาตรา: {first_section['section_number']}")
        print(f"เนื้อหา: {first_section['raw_text']}")