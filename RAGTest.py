import requests
import json
import faiss
import numpy as np
from pypdf import PdfReader
import glob
import os


class PDFRAGChatbot:
    def __init__(self, model="gemma3:4b", host="http://localhost:11434"):
        self.model = model
        self.host = host
        self.index = None
        self.docs = []  # [(chunk_text, source_info), ...]

    # --- Embeddings ---
    def embed(self, text):
        response = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "input": text}
        )
        return np.array(response.json()["embedding"], dtype="float32")

    # --- PDF Loader + Chunking ---
    def load_pdf(self, file_path, chunk_size=500, overlap=50):
        reader = PdfReader(file_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append((chunk, f"{os.path.basename(file_path)} - page {page_num+1}"))
        print(f"📄 Loaded {len(chunks)} chunks from {file_path}")
        return chunks

    # --- Build FAISS Index from Folder ---
    def build_index_from_folder(self, folder_path):
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            print(f"❌ No PDFs found in folder: {folder_path}")
            return

        all_chunks = []
        for pdf in pdf_files:
            chunks = self.load_pdf(pdf)
            all_chunks.extend(chunks)

        self.docs = all_chunks
        embeddings = [self.embed(chunk[0]) for chunk in all_chunks]
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        print(f"✅ Built FAISS index with {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")

    # --- Retrieve Relevant Chunks ---
    def retrieve(self, query, k=5):
        query_vec = self.embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        return [self.docs[i] for i in indices[0]]

    # --- Generate Answer ---
    def generate(self, query, context_text):
        prompt = f"Answer the question using the following context:\n\n{context_text}\n\nQuestion: {query}"
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    # --- Chat with RAG ---
    def chat(self, query, k=5):
        context_docs = self.retrieve(query, k)
        context_text = "\n".join([f"({src}) {txt}" for txt, src in context_docs])
        answer = self.generate(query, context_text)

        print(f"\n🤖 {self.model}:\n{answer}\n")

    # --- Run interactive loop ---
    def run(self, folder_path="./pdfs"):
        self.build_index_from_folder(folder_path)

        if not self.index:
            print("❌ Could not start chatbot (no PDFs indexed).")
            return

        print(f"\n🤖 RAG Chatbot running with {self.model}. Type 'exit' or 'quit' to stop.\n")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit", "done"]:
                print("👋 Goodbye!")
                break
            self.chat(query)


if __name__ == "__main__":
    rag_bot = PDFRAGChatbot(model="gemma3:4b")
    rag_bot.run(folder_path="./pdfs")  # put all your PDFs inside ./pdfs folder
