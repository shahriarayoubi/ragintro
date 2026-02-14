"""
Project 1: RAG with Semantic Search (OpenAI Embeddings)

"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def load_knowledge_base(knowledge_dir: str = "knowledge") -> dict:
    """
    STEP 1: PREPARE DOCUMENTS - Load from external files
    """
    print("\n" + "="*70)
    print("STEP 1: Loading documents from files")
    print("="*70)

    knowledge_base = {}
    knowledge_path = Path(__file__).parent / knowledge_dir

    if not knowledge_path.exists():
        print(f"Knowledge directory '{knowledge_path}' not found. Please create it and add some text files.")
        return knowledge_base
    
    for file in knowledge_path.glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            knowledge_base[file.stem] = content
            print(f"Loaded '{file.name}' with {len(content.split())} words.")


    return knowledge_base


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)

    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Formula: cosine_similarity = (A . B) / (||A|| * ||B||)
    where:
    - A . B is the dot product of vectors A and B
    - ||A|| is the magnitude (length) of vector A
    - ||B|| is the magnitude (length) of vector B
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def create_embeddings_cache(knowledge_base: dict) -> dict:

    print("\n" + "="*70)
    print("STEP 2: Creating embeddings for all documents")
    print("="*70)
    print("Converting each document to a 1536-dimensional vector...\n")

    embeddings_cache = {}

    for doc_name, content in knowledge_base.items():
        #print(f"Creating embedding for '{doc_name}'...")
        embeddings_cache[doc_name] = get_embedding(content)

    return embeddings_cache


# def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:

#     print("\n" + "="*70)
#     print("STEP 3: Semantic search with cosine similarity")
#     print("="*70)
    
#     return similarities[:top_k]


# def rag_query(user_question, knowledge_base, embeddings_cache):

#     return response.choices[0].message.content


def main():
    """
    Run example RAG queries with semantic search
    """
    print("=" * 70)
    print("RAG Assistant - Semantic Search with Embeddings")
    print("=" * 70)

    knowledge_base = load_knowledge_base()

    #print(knowledge_base["office_locations"])

    create_embeddings_cache(knowledge_base)

if __name__ == "__main__":
    main()


