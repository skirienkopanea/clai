import sys
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


## RAG Quickstart with comments based on https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/
#### INDEXING ####
query = "What are the colors of the italian flag"


# Load Documents

# Example of online source where we only parse the html contents regarding the css classes: "post-content", "post-title" and "post-header"

loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Sala_del_Tricolore",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mw-body-content", "mw-page-title-main", "mw-normal-catlinks")
        )
    ),
)

docs = loader.load()

loader = DirectoryLoader("data/", glob="*.txt", recursive=True)
docs = docs + loader.load()


# Testing doc retrival first 100 chars
#print(docs[0].page_content[:100])

# Split
# Because the whole document wouldnt fit as the context of a prompt, and it wouldnt procude good results either,
# we split it into chunks that then are going to be retrieved based on their similiartiy to the original question.

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=100)
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

# similar results above minimum threshold

retriever = vectorstoredb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.1,
                   'k':30}
                   )
retrieved_docs = retriever.invoke(query)

# Get the most relevant doc from each source
sources = []
final_docs = []
for doc in retrieved_docs:
    if doc.metadata["source"] not in sources:
        final_docs.append(doc)
        sources.append(doc.metadata["source"])
# Diverse results
"""
retriever = vectorstoredb.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 3, 'fetch_k': 10, 'lambda_mult': 0.3}
            )
retrieved_docs = retriever.invoke(query)            
"""


#print("Similar docs: ",len(final_docs), "\nFirst doc: ",final_docs[0].page_content[:100])

# Post-processing
def context_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Post-processing
def source_docs(docs):
    
    return "\n\n".join(doc.page_content + "\nSource: " + doc.metadata["source"]for doc in docs)

# Chain
# 4 mini

try:
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are context-answering machine for a RAG app for local files. Don't use any other outside information other than the one in the context. Keep the answers as short and concise as possible."},
            {"role": "user", "content": f"Question: {query}\nContext: {context_docs(final_docs)}"}
        ]
    )

    content = response.choices[0].message.content

    if len(final_docs):
        print(content,f"\n\nContext:\n{source_docs(final_docs)}", )
    else: print("I don't know")
except Exception as e:
    print(f"An error occurred: {e}")
