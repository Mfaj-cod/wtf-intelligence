import chromadb
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


class Ingestor:
    def __init__(self, data_path='./data/research/', glob_pattern='**/*.txt', collection_name='wtf_research', db_path='./chroma_db', embedding_model='nomic-embed-text'):
        self.data_path = data_path
        self.glob_pattern = glob_pattern
        self.collection_name = collection_name
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.docs = []
        self.chunks = []
        self.embeddings = None
        self.collection = None

    def load_documents(self):
        try:
            data_dir = Path(self.data_path)
            if not data_dir.exists():
                print(f"Research directory does not exist: {data_dir}", flush=True)
                return []

            files = list(data_dir.rglob("*.txt"))
            print(f"Found {len(files)} text files.", flush=True)

            documents = []
            for file_path in files:
                print(f"Reading: {file_path}", flush=True)

                text = file_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "document": file_path.name,
                        }
                    )
                )

                print(
                    f"Loaded {file_path.name}: {len(text)} characters",
                    flush=True
                )

            self.docs = documents
            print(
                f"Loaded {len(self.docs)} documents successfully.",
                flush=True
            )

            return self.docs

        except Exception as exc:
            print(f"Error loading documents: {exc}", flush=True)
            return []

    def split_documents(self, chunk_size=512, chunk_overlap=50):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.chunks = splitter.split_documents(self.docs)
            print(f'Split {len(self.chunks)} chunks')
            return self.chunks
        except Exception as exc:
            print(f'Error splitting documents: {exc}')
            return []

    def init_embedding_model(self):
        try:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            print(f'Initialized embedding model: {self.embedding_model}')
            return self.embeddings
        except Exception as exc:
            print(f'Error initializing embedding model: {exc}')
            return None

    def init_collection(self):
        try:
            client = chromadb.PersistentClient(path=self.db_path)
            self.collection = client.get_or_create_collection(self.collection_name)
            print(f'Connected to collection: {self.collection_name}')
            return self.collection
        except Exception as exc:
            print(f'Error initializing Chroma collection: {exc}')
            return None

    def index_chunks(self):
        if not self.embeddings:
            print('Embedding model not initialized.')
            return 0

        if not self.collection:
            print('Collection not initialized.')
            return 0

        indexed_count = 0
        try:
            for i, chunk in enumerate(self.chunks):
                try:
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    self.collection.add(
                        ids=[f'chunk_{i}'],
                        embeddings=[embedding],
                        documents=[chunk.page_content],
                        metadatas=[{'source': chunk.metadata.get('source', 'unknown')}],
                    )
                    indexed_count += 1
                except Exception as exc:
                    print(f'Error indexing chunk {i}: {exc}')
            print(f'Indexed {indexed_count} chunks successfully')
            return indexed_count
        except Exception as exc:
            print(f'Error during indexing process: {exc}')
            return indexed_count

    def run(self):
        try:
            self.load_documents()
            if not self.docs:
                print('No documents found to ingest.')
                return

            self.split_documents()
            if not self.chunks:
                print('No chunks created from documents.')
                return

            self.init_embedding_model()
            self.init_collection()
            self.index_chunks()
        except Exception as exc:
            print(f'Unexpected error during ingestion: {exc}')


# if __name__ == '__main__':
#     ingestor = Ingestor()
#     ingestor.run()
