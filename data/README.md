# Sample Data Directory

This directory is where you should place your documents for the RAG system to ingest.

## Supported File Types

The RAG system uses LlamaIndex's SimpleDirectoryReader, which supports various file types including:
- Text files (.txt)
- PDF files (.pdf)
- Markdown files (.md)
- HTML files (.html)
- JSON files (.json)
- CSV files (.csv)
- And more

## Example Usage

1. Place your documents in this directory
2. Run the server or RAG workflow
3. The documents will be automatically ingested and made available for querying

## Sample Document

For testing purposes, you can create a simple text file in this directory:

```
# DeepSeek AI

DeepSeek is an AI research company focused on developing large language models.

## DeepSeekR1

DeepSeekR1 is a large language model trained on a diverse corpus of text data. 
It was trained using a combination of supervised learning and reinforcement learning from human feedback (RLHF).

The model architecture is based on a transformer decoder with several billion parameters.
Training was conducted on a large cluster of GPUs over several months.

Key features of DeepSeekR1 include:
- Strong reasoning capabilities
- Code generation and understanding
- Multi-turn conversation abilities
- Knowledge cutoff date of 2023
```

Save this as `deepseek_info.txt` in the data directory to test the RAG system with the example query in the main script.

