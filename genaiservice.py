# import os
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter

# class GenAIService:
#     def __init__(self):
#         self.index_path = "faiss_index"
#         self.embeddings = OpenAIEmbeddings()
#         self.index = None

#     def load_index(self):
#         if os.path.exists(self.index_path):
#             self.index = FAISS.load_local(self.index_path, self.embeddings)
#         else:
#             self.index = FAISS(embedding_function=self.embeddings)

#     def process_data(self, data):\
#         loader = CSVLoader()
#         data = loader.load(data)

#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         chunks = text_splitter.split_documents(data)

#         for chunk in chunks:
#             query = chunk.page_content
#             results = self.index.similarity_search(query)

#             if not results:
#                 self.index.add_texts([chunk])
#                 print(f"Chunk added to index: {chunk.page_content}")

#         self.index.save_local(self.index_path)

#     def run(self, data):
#         self.load_index()
#         self.process_data(data)
