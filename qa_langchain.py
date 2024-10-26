from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


import os

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("state_of_the_union.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
)

questions = [
    "Who is the speaker",
    "What did the president say about Ketanji Brown Jackson",
    "What are the threats to America",
    "Who are mentioned in the speech",
    "Who is the vice president",
    "How many projects were announced",
]

for query in questions:
    print("Query:", query)
    print("Answer:", index.query(query, llm=llm))