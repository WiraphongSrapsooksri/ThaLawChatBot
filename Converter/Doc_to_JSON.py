import os
from docx import Document
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class LegalReference:
    index: str
    law_type: str
    law_number: str

@dataclass
class LegalSection:
    section_number: str
    section_content: str
    subsections: List[Dict[str, str]]
    reference: Optional[LegalReference] = None

class DocxLegalConverter:
    def __init__(self):
        self.current_reference = None
        
    def read_docx(self, file_path: str) -> str:
        """Read content from a DOCX file and convert to plain text."""
        try:
            doc = Document(file_path)
            paragraphs = []
            
            for para in doc.paragraphs:
                # Skip empty paragraphs
                if para.text.strip():
                    # Handle different styles/formatting
                    text = para.text.strip()
                    
                    # Check if it's a section header or special formatting
                    if any(run.bold for run in para.runs):
                        # Add extra newline for sections
                        paragraphs.append(f"\n{text}\n")
                    else:
                        paragraphs.append(text)
            
            # Join with double newlines to maintain paragraph separation
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            raise Exception(f"Error reading DOCX file: {str(e)}")

    def extract_law_info(self, text: str) -> Dict:
        """Extract law type and number information."""
        law_pattern = r"กฎหมาย\s*(.*?)\s*\[([^\]]+)\]"
        match = re.search(law_pattern, text)
        
        if match:
            return {
                "law_type": match.group(1).strip(),
                "law_number": f"[{match.group(2)}]"
            }
        return {"law_type": "Unknown", "law_number": ""}

    def extract_reference_number(self, text: str) -> Optional[str]:
        """Extract reference number (e.g., '5/115')."""
        pattern = r"รายละเอียดคำศัพท์\s+รายการที่\s+(\d+/\d+)"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def parse_section(self, text: str, current_reference: Optional[LegalReference] = None) -> LegalSection:
            """Parse individual sections with reference information."""
            # Extract section number - improved pattern matching
            section_pattern = r'มาตรา\s+(\d+(?:/\d+)?(?:\s*[ก-ฮ])?(?:\[\d+\])?)'
            section_match = re.search(section_pattern, text)
            
            # Extract only the numeric part and remove any trailing/leading whitespace
            if section_match:
                # Remove any non-numeric characters except '/' for section numbers like "1/1"
                section_number = re.sub(r'[^\d/]', '', section_match.group(1)).strip()
            else:
                section_number = ""
            
            # Split content into main content and subsections
            lines = text.split('\n')
            main_content = []
            subsections = []
            current_subsection = []
            
            # Find the main content line (starting with มาตรา)
            for line in lines:
                if line.strip().startswith('มาตรา'):
                    main_content.append(line.strip())
                    break
            
            # Process remaining lines
            for line in lines[lines.index(main_content[0]) + 1:]:
                if re.match(r'^\s*[\(\[]([0-9ก-ฮ]+)[\)\]]', line):
                    if current_subsection:
                        subsections.append({
                            "number": current_subsection[0].strip(),
                            "content": " ".join(current_subsection[1:]).strip()
                        })
                    current_subsection = [line]
                elif current_subsection:
                    current_subsection.append(line.strip())
                else:
                    if line.strip():
                        main_content.append(line.strip())
            
            # Add last subsection if exists
            if current_subsection:
                subsections.append({
                    "number": current_subsection[0].strip(),
                    "content": " ".join(current_subsection[1:]).strip()
                })
                
            return LegalSection(
                section_number=section_number,
                section_content=" ".join(main_content),
                subsections=subsections,
                reference=current_reference
            )


    def convert_to_json(self, text: str) -> Dict:
        """Convert legal document text to structured JSON with enhanced metadata."""
        sections = []
        current_law_info = None
        current_reference = None
        
        # Split text into blocks
        blocks = text.split('\n\n')
        
        for block in blocks:
            if not block.strip():
                continue
                
            # Check for law information
            if "กฎหมาย" in block:
                current_law_info = self.extract_law_info(block)
            
            # Check for reference number
            ref_number = self.extract_reference_number(block)
            if ref_number and current_law_info:
                current_reference = LegalReference(
                    index=ref_number,
                    law_type=current_law_info["law_type"],
                    law_number=current_law_info["law_number"]
                )
            
            # Check for section content
            if block.strip().startswith('มาตรา'):
                section = self.parse_section(block, current_reference)
                section_dict = {
                    "section_number": section.section_number,
                    "content": section.section_content,
                    "subsections": section.subsections,
                }
                
                if section.reference:
                    section_dict["reference"] = {
                        "index": section.reference.index,
                        "law_type": section.reference.law_type,
                        "law_number": section.reference.law_number
                    }
                
                sections.append(section_dict)
        
        # Create document metadata
        metadata = {
            "title": "กฎหมายเช่าซื้อของประเทศไทย",
            "conversion_date": datetime.now().isoformat(),
            "laws_referenced": []
        }
        
        # Collect unique law references
        law_refs = set()
        for section in sections:
            if "reference" in section:
                law_ref = f"{section['reference']['law_type']} {section['reference']['law_number']}"
                law_refs.add(law_ref)
        
        metadata["laws_referenced"] = sorted(list(law_refs))
        
        return {
            "metadata": metadata,
            "sections": sections
        }

    def process_docx_file(self, input_path: str, output_path: str) -> None:
        """Process a DOCX file and save the result as JSON."""
        try:
            # Read DOCX content
            content = self.read_docx(input_path)
            
            # Convert to structured JSON
            json_data = self.convert_to_json(content)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            print(f"Successfully converted {input_path} to {output_path}")
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")

def main():
    # Example usage
    try:
        converter = DocxLegalConverter()
        
        # Set your input and output paths
        input_path = "../Doc/กฎหมายเช่าซื้อ.docx"
        output_path = "../DataConverter/legal_document.json"
        
        # Process the file
        converter.process_docx_file(input_path, output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()