import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class SimpleEmbeddings:
    """Simple wrapper for sentence-transformers"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        """Make the object callable for compatibility"""
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)

class RAGEngine:
    def __init__(self, documents_folder="rules_documents"):
        self.documents_folder = documents_folder
        self.vectorstore = None
        self.embeddings = SimpleEmbeddings()
        
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
            print(f"✅ Created folder: {documents_folder}")
        
        self.load_documents()
    
    def load_documents(self):
        """
        Load all PDF, TXT, and DOCX files from the documents folder
        """
        print(f"📂 Loading documents from: {self.documents_folder}")
        
        all_documents = []
        file_count = 0
        
        # Supported file types - FIXED FOR WINDOWS
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,  # Changed from UnstructuredWordDocumentLoader
        }
        
        # Load all files
        for filename in os.listdir(self.documents_folder):
            file_path = os.path.join(self.documents_folder, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in loaders:
                try:
                    loader = loaders[file_ext](file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    file_count += 1
                    print(f"   ✅ Loaded: {filename} ({len(documents)} pages)")
                except Exception as e:
                    print(f"   ❌ Error loading {filename}: {e}")
        
        if file_count == 0:
            print("⚠️  No documents found. Please add PDF, TXT, or DOCX files to the rules_documents folder.")
            return
        
        # Split documents into chunks
        print(f"📝 Splitting {len(all_documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(all_documents)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Create vector store
        print("🔍 Creating vector embeddings (this may take a minute)...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"✅ RAG Engine ready! Loaded {file_count} documents with {len(chunks)} searchable chunks")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.vectorstore:
            return []
        
        # Search for similar documents
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'score': float(score)
            })
        
        return formatted_results
    
    def get_context_for_question(self, question: str, max_chunks: int = 3) -> str:
        """
        Get relevant context from documents for a given question
        
        Args:
            question: User's question
            max_chunks: Maximum number of document chunks to retrieve
            
        Returns:
            Formatted context string with citations
        """
        if not self.vectorstore:
            return "No rule documents loaded."
        
        results = self.search(question, k=max_chunks)
        
        if not results:
            return "No relevant information found in rule documents."
        
        # Build context with citations
        context_parts = []
        for i, result in enumerate(results, 1):
            source = os.path.basename(result['source'])
            page = result['page']
            content = result['content']
            
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{content}\n"
            )
        
        return "\n".join(context_parts)


# Test the RAG engine
if __name__ == "__main__":
    print("🚀 Testing RAG Engine...")
    
    # Initialize
    rag = RAGEngine()
    
    # Test search
    test_question = "What is the definition of a machine gun?"
    print(f"\n🔍 Test Question: {test_question}")
    
    context = rag.get_context_for_question(test_question)
    print(f"\n📄 Retrieved Context:\n{context}")