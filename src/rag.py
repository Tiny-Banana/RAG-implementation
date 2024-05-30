import os
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import  SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import uuid
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

os.environ['COHERE_API_KEY'] = "1pUAePYpYJWfTe5IZi7sgT4G11uIhppYpPwW3rsE"
SUMMARY_CACHE_PATH = "./data/cache/summary.json"

def load_summary():
    if os.path.exists(SUMMARY_CACHE_PATH):
        with open(SUMMARY_CACHE_PATH, 'r') as f:
            return json.load(f)
    else:
        return None
    
def save_summary(summaries):
    """Save summaries and embeddings to cache."""
    with open(SUMMARY_CACHE_PATH, 'w') as f:
        json.dump(summaries, f)

def answer_query(question):
    ### Load
    loader = DirectoryLoader("./data/raw", glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()

    ### LLM
    llm = ChatCohere(model="command-r", format="json", temperature=0)
    embd = CohereEmbeddings(model="embed-english-light-v3.0")

    ### Summary embedding
    summaries = load_summary()

    if summaries is None:
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | llm
            | StrOutputParser()
        )
        summaries = chain.batch(docs, {"max_concurrency": 5})
        save_summary(summaries)

    # The vectorstore to use to index the child chunks
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="summaries", embedding_function=embd)

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    ### Retrieval Grader
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],)

    retrieval_grader = prompt | llm | JsonOutputParser()
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

    ### Generate
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)


    ### Hallucination Grader
    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    print(hallucination_grader.invoke({"documents": docs, "generation": generation}))

    ### Answer Grader
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    print(answer_grader.invoke({"question": question, "generation": generation}))

    return generation

answer_query("What is the connection between lang and lamu?")