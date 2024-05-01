# from google.colab import drive
# drive.mount('/content/drive')

csv_file_path = 'anlamsal_arama_data.csv'
from tqdm import tqdm
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def load_docs_from_csv(csv_file):
    df = pd.read_csv(csv_file, sep='|')
    documents = []
    for index, row in df.iterrows():
        url = row['url'].strip()
        title = row['title'].strip()
        text = row['text'].strip()
        document = Document(url=url, title=title, text=text, page_content=text)
        documents.append(document)
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def semantic_search(queries_df, vectordb):
    results = []

    for query in tqdm(queries_df['sorgu'], desc="Processing queries"):
        matching_docs = vectordb.similarity_search(query)
        top_5 = matching_docs[:5]

        # Eğer 5'ten az sonuç dönerse, eksik sonuçları boş stringlerle doldur
        top_results = [doc.page_content for doc in top_5] + [""] * (5 - len(top_5))
        results.append(top_results)

    results_df = pd.DataFrame(results, columns=['1st', '2nd', '3rd', '4th', '5th'])
    return results_df

# CSV dosyasından belgeleri yükle
documents = load_docs_from_csv(csv_file_path)

# Belgeleri parçala
docs = split_docs(documents)
print(len(docs))

# Gömme modelini yükle
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vektör veritabanını oluştur
db = Chroma.from_documents(docs, embeddings)

# Anlamsal arama sorgularını işle
queries_df = pd.read_csv('task2_yarismaci_test_data.csv', sep='|')
semantic_results = semantic_search(queries_df, db)

# Sonuçları .csv dosyasına yaz
semantic_results.to_csv('anlamsal_arama.csv', sep='|', index=False)
