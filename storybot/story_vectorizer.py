from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the cleaned CSV file
df = pd.read_csv("../data/children_stories.csv").head(10) #load only the first 10 stories

# Set up embeddings and database location
embeddings = OllamaEmbeddings(model="gemma:2b")
db_location = "./vector-base"
add_documents = not os.path.exists(db_location)

# If vector DB is not created, embed and store documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        page_content = str(row["Names"]).strip() + " " + str(row["Description"]).strip()
        document = Document(
            page_content=page_content,
            metadata={"category": str(row["Category"]).strip(), "id": str(i)}
        )
        documents.append(document)
        ids.append(str(i))

# Set up Chroma vector store
vector_store = Chroma(
    collection_name="stories",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# # checking if vector base works - retrieve a few stories
# sample_docs = vector_store.get(include=["documents"], ids=["0", "1", "2"])
# for i, doc in enumerate(sample_docs["documents"]):
#     print(f"\n--- Story {i + 1} ---\n{doc}")

# Set up retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)