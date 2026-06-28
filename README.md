# RAG_Based-Knowledge-Assistant
RAG Knowledge Assistant is a free document question-answering system built in Google Colab. It allows users to upload PDF, DOCX, TXT, or Markdown files, ask natural language questions, and receive answers with source citations. The project uses Sentence Transformers for embeddings, ChromaDB for vector search, and Hugging Face FLAN-T5 for answer generation without requiring paid OpenAI API credits.


## Features

- Upload PDF, DOCX, TXT, and Markdown files
- Extract document text
- Split text into chunks
- Generate free local embeddings using Sentence Transformers
- Store and search chunks using ChromaDB
- Ask questions about uploaded documents
- Show citations with source and page number
- Run simple evaluation tests

## Tech Stack

- Python
- Google Colab
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- FLAN-T5
- PyPDF
- python-docx

## Project Structure

```text
rag_based-knowledge-assistant/
├── README.md
├── requirements.txt
├── rag_knowledge_assistant.ipynb
├── .gitignore
├── sample_docs/
│   └── sample.txt
└── screenshots/
    ├── upload.png
    ├── retrieval.png
    └── eval-result.png
