import re
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
PDF_FILE_PATH = "Income_Tax_Training_File.pdf"
FAISS_INDEX_PATH = "tax_knowledge_base.index"
CHUNKS_DATA_PATH = "tax_chunks_and_metadata.json"
MODEL_NAME = 'all-mpnet-base-v2' # Using a more accurate model for better retrieval quality

def extract_text_from_pdf(file_path):
    """Extracts raw text from a PDF file."""
    print(f"Reading text from '{file_path}'...")
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        print("✅ Text extracted successfully.")
        return text
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{file_path}'")
        return None

def logical_chunking(text):
    """
    Splits the raw text into logical, self-contained chunks based on the document's hierarchical structure.
    This version uses a more robust, stateful parsing method.
    """
    print("Performing enhanced logical chunking...")
    
    chunks_with_metadata = []
    # This regex identifies the main numbered sections (e.g., "1. Tax Deductions")
    section_pattern = re.compile(r'^\d\.\s[A-Z][a-zA-Z\s]+', re.MULTILINE)
    # This regex identifies the main bulleted subsections (e.g., "• Section 80C")
    subsection_pattern = re.compile(r'^•\s[A-Z][a-zA-Z0-9\s\(\)\-]+:', re.MULTILINE)

    # Find all main section titles and their starting positions
    sections = [(match.group(0), match.start()) for match in section_pattern.finditer(text)]

    for i, (section_title, start_pos) in enumerate(sections):
        # Determine the content of the entire section
        end_pos = sections[i+1][1] if i + 1 < len(sections) else len(text)
        section_content = text[start_pos:end_pos]

        # Find all subsection titles and their starting positions within this section
        subsections = [(match.group(0), match.start()) for match in subsection_pattern.finditer(section_content)]

        if not subsections:
            # If a section has no bulleted subsections, treat the whole section as one chunk
            chunks_with_metadata.append({
                "main_section": section_title.strip(),
                "sub_section": section_title.strip(),
                "content": section_content.strip()
            })
            continue

        # The text before the first subsection is the section's introduction
        intro_text_end = subsections[0][1]
        section_intro = section_content[:intro_text_end].strip()

        for j, (subsection_title, sub_start_pos) in enumerate(subsections):
            # Determine the content of this subsection
            sub_end_pos = subsections[j+1][1] if j + 1 < len(subsections) else len(section_content)
            subsection_content = section_content[sub_start_pos:sub_end_pos].strip()

            # Combine intro text with the first subsection for better context
            final_content = (section_intro + "\n\n" + subsection_content) if j == 0 else subsection_content

            chunks_with_metadata.append({
                "main_section": section_title.strip(),
                "sub_section": subsection_title.replace("•", "").strip(),
                "content": final_content
            })

    print(f"✅ Document split into {len(chunks_with_metadata)} logical chunks.")
    return chunks_with_metadata

def create_and_save_embeddings(chunks_with_metadata):
    """
    Creates vector embeddings for each chunk and saves them to a FAISS index.
    Also saves the text chunks and metadata for later retrieval.
    """
    if not chunks_with_metadata:
        print("No chunks were created. Halting embedding process.")
        return

    print(f"Loading sentence transformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    corpus = [chunk['content'] for chunk in chunks_with_metadata]
    
    print("Encoding chunks into vector embeddings... (This may take a moment)")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(index)
    ids = np.array(range(len(corpus))).astype('int64')
    index.add_with_ids(embeddings, ids)
    
    print(f"✅ Embeddings created. FAISS index has {index.ntotal} vectors.")
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to '{FAISS_INDEX_PATH}'")
    
    with open(CHUNKS_DATA_PATH, 'w') as f:
        json.dump(chunks_with_metadata, f, indent=2)
    print(f"✅ Chunks and metadata saved to '{CHUNKS_DATA_PATH}'")


if __name__ == "__main__":
    raw_text = extract_text_from_pdf(PDF_FILE_PATH)
    
    if raw_text:
        chunks = logical_chunking(raw_text)
        
        print("\n--- Sample Chunks Verification ---")
        for sample_chunk in chunks[:3]:
            print(json.dumps(sample_chunk, indent=2))
            print("-" * 20)
        
        create_and_save_embeddings(chunks)
        
        print("\n🎉 Knowledge base creation complete!")
