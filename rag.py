import sys
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


## RAG Quickstart with comments based on https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
#### INDEXING ####
query = "How to make cafe con leche"

# Load Documents
# Example of online source where we only parse the html contents regarding the css classes: "post-content", "post-title" and "post-header"
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Caf%C3%A9_con_leche",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mw-body-content", "mw-page-title-main", "mw-normal-catlinks")
        )
    ),
)
docs = loader.load()

# Testing doc retrival first 100 chars
#print(docs[0].page_content[:100])

# Split
# Because the whole document wouldnt fit as the context of a prompt, and it wouldnt procude good results either,
# we split it into chunks that then are going to be retrieved based on their similiartiy to the original question.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#print("Split size: ",len(splits),"\nPage content: ", splits[0].page_content[:100],"\nMetadata: ",splits[0].metadata)

# Embed
# This is our way to index the chunks for retrieval
# You have to check in your openai API subscription which embedding models you have available.
# You should already have saved the environment variable OPENAI_API_KEY in your system.
model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstoredb = Chroma.from_documents(persist_directory="persist", documents=splits, 
                                    embedding=model)

#Commented to reuse vectorstoredb. Need to implement proper logic

#vectorstoredb = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())

#print(vectorstoredb)

#### RETRIEVAL and GENERATION ####
# Retriever logic: Dependning on the application and how your data is stored you may need different approaches
# Info about all approaches https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/
# We well use the most basic and the most common type of Retriever, which is the VectorStoreRetriever,
# which uses the similarity search capabilities of a vector store to facilitate retrieval.
# Any VectorStore can easily be turned into a Retriever with VectorStore.as_retriever():

retriever = vectorstoredb.as_retriever()
retrieved_docs = retriever.invoke(query)

#print("Similar docs: ",len(retrieved_docs), "\nFirst doc: ",retrieved_docs[0].page_content[:100])

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
# 4 mini

prompt = f"""Question: {query}
Context: {format_docs(retrieved_docs)}
"""

try:
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are context-answering machine for a RAG app for local files. Don't use any other outside information other than the one in the context. Keep the answers as short and concise as possible."},
            {"role": "user", "content": f"Question: {query}\nContext: {format_docs(retrieved_docs)}"}
        ]
    )

    content = response.choices[0].message.content
    print(content)
except Exception as e:
    print(f"An error occurred: {e}")
