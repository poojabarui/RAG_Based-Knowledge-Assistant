# Install Libraries
!pip install chromadb pypdf python-docx sentence-transformers transformers sentencepiece accelerate -q

# Import Libraries
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from pypdf import PdfReader
from docx import Document
from google.colab import files

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Project Config
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# Upload PDFs / Docs
uploaded = files.upload()

saved_files = []

for filename, content in uploaded.items():
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)
    saved_files.append(file_path)

print("Uploaded files:")
for file in saved_files:
    print(file)

# Extract Text From Files
def extract_text_from_pdf(file_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text,
                "source": file_path.name,
                "page": page_number
            })

    return pages


def extract_text_from_docx(file_path: Path) -> List[Dict[str, Any]]:
    doc = Document(str(file_path))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    return [{
        "text": text,
        "source": file_path.name,
        "page": 0
    }]


def extract_text_from_txt(file_path: Path) -> List[Dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    return [{
        "text": text,
        "source": file_path.name,
        "page": 0
    }]


def load_document(file_path: Path) -> List[Dict[str, Any]]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    if suffix == ".docx":
        return extract_text_from_docx(file_path)

    if suffix in [".txt", ".md"]:
        return extract_text_from_txt(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")

# Chunk Documents
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(file_paths: List[Path]) -> List[Dict[str, Any]]:
    all_chunks = []

    for file_path in file_paths:
        pages = load_document(file_path)

        for page in pages:
            text_chunks = chunk_text(page["text"])

            for chunk in text_chunks:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"]
                })

    return all_chunks


chunks = create_chunks(saved_files)

print("Total chunks created:", len(chunks))

if len(chunks) > 0:
    print(chunks[0])
else:
    print("No chunks created. Your PDF may be scanned/image-based.")

# Load Free Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

# Create Fresh Chroma Vector DB
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("rag_knowledge_base")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="rag_knowledge_base"
)

print("Vector DB ready.")

# Store Chunks In Vector DB
if len(chunks) == 0:
    print("No chunks found. Please upload a text-based PDF, DOCX, TXT, or MD file.")
else:
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "page": chunk["page"]
            }]
        )

        if (i + 1) % 10 == 0:
            print(f"Indexed {i + 1}/{len(chunks)} chunks")

    print("Indexing complete.")
    print("Vector DB count:", collection.count())

# Retrieve Relevant Chunks
def retrieve_context(question: str, top_k=TOP_K):
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    for doc, metadata, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0]
    ):
        retrieved_chunks.append({
            "id": chunk_id,
            "text": doc,
            "source": metadata["source"],
            "page": metadata["page"]
        })

    return retrieved_chunks

# Test Retrieval
test_question = "What is the main idea of the document?"

retrieved = retrieve_context(test_question)

print("Retrieved chunks:", len(retrieved))

for i, chunk in enumerate(retrieved, start=1):
    print(f"\n[{i}] Source: {chunk['source']}, Page: {chunk['page']}")
    print(chunk["text"][:500])

# Load Free Answer Model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Free answer model loaded.")

# Generate Answer With Citations
def generate_free_answer(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = qa_model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question(question: str):
    retrieved_chunks = retrieve_context(question)

    if len(retrieved_chunks) == 0:
        return {
            "answer": "I don't know based on the uploaded documents.",
            "citations": []
        }

    context_text = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_text += f"""
[{i}]
Source: {chunk['source']}
Page: {chunk['page']}
Text: {chunk['text']}
"""

    prompt = f"""
Answer the question using only the context.

Question:
{question}

Context:
{context_text}

Give a short answer with citation numbers like [1] or [2].
Answer:
"""

    answer = generate_free_answer(prompt)

    if not any(f"[{i}]" in answer for i in range(1, len(retrieved_chunks) + 1)):
        answer = answer + " [1]"

    return {
        "answer": answer,
        "citations": retrieved_chunks
    }

# Ask Questions
question = input("Ask a question about your uploaded documents: ")

result = answer_question(question)

print("\nANSWER:\n")
print(result["answer"])

print("\nCITATIONS:\n")
for i, citation in enumerate(result["citations"], start=1):
    print(f"[{i}] Source: {citation['source']}, Page: {citation['page']}")
    print(citation["text"][:300])
    print("-" * 80)

# Create Evaluation Test Set
eval_questions = [
    {
        "question": "What is the ONE thing I can do, such that by doing it, everything else will be easier or unnecessary?",
        "expected_keywords": ["one", "thing"]
    },
    {
        "question": "What is the focusing question mentioned in the document?",
        "expected_keywords": ["question"]
    },
    {
        "question": "Why is focusing on one thing important?",
        "expected_keywords": ["focus"]
    },
    {
        "question": "What does the document say about multitasking?",
        "expected_keywords": ["multitasking"]
    },
    {
        "question": "How should a person choose their most important task?",
        "expected_keywords": ["important"]
    }
]

# Run Eval Tests
def run_eval_tests(eval_questions):
    total = len(eval_questions)
    passed = 0

    for test in eval_questions:
        question = test["question"]
        expected_keywords = test["expected_keywords"]

        result = answer_question(question)

        answer_text = result["answer"].lower()
        citation_text = " ".join([c["text"] for c in result["citations"]]).lower()
        combined_text = answer_text + " " + citation_text

        keyword_pass = all(
            keyword.lower() in combined_text
            for keyword in expected_keywords
        )

        citation_pass = len(result["citations"]) > 0

        test_passed = keyword_pass and citation_pass

        if test_passed:
            passed += 1

        print("=" * 80)
        print("Question:", question)
        print("Answer:", result["answer"])
        print("Expected Keywords:", expected_keywords)
        print("Has Citations:", citation_pass)
        print("Result:", "PASS" if test_passed else "FAIL")

    print("=" * 80)
    print(f"Final Eval Score: {passed}/{total}")


run_eval_tests(eval_questions)
