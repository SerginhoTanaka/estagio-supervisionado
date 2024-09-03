# import os
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter

# # Defina o caminho do índice
# index_path = "faiss_index"

# # Carregar embeddings
# embeddings = OpenAIEmbeddings()

# # Verifique se o índice já existe
# if os.path.exists(index_path):
#     # Carregar índice existente
#     index = FAISS.load_local(index_path, embeddings)
# else:
#     index = FAISS(embedding_function=embeddings)  # Crie um novo índice se não existir

# # Carregar CSV
# loader = CSVLoader('caminho/para/seu/arquivo.csv')
# data = loader.load()

# # Dividir texto
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# chunks = text_splitter.split_documents(data)

# # Adicionar chunks ao índice se não existirem
# for chunk in chunks:
#     query = chunk.page_content  # ou outra representação do chunk
#     results = index.similarity_search(query)
    
#     if not results:
#         # Se não houver resultados, o chunk não está no índice
#         index.add_texts([chunk])  # Adiciona o chunk ao índice
#         print(f"Chunk adicionado ao índice: {chunk.page_content}")

# # Salvar o índice atualizado
# index.save_local(index_path)