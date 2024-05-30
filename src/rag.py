import os
import json
import uuid
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

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
            | PromptTemplate(template="Summarize the following document:\n\n{doc}", input_variables=["doc"])
            | llm
            | StrOutputParser()
        )
        summaries = chain.batch(docs, {"max_concurrency": 5})
        save_summary(summaries)

    # The vectorstore to use to index the child chunks
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

    rag_chain = prompt | llm | StrOutputParser()

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

    ### State
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    ### Nodes
    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = vectorstore.similarity_search(question)
        print(documents)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "supported"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generate

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "supported": END,
        },
    )

    app = workflow.compile()

    # Test
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")

    print("Question: " + value["question"])
    print("Answer: " + value["generation"])
    print("Sources: ")
    for doc in value["documents"]:
        print(doc.metadata)

answer_query("What are the physical characteristics of lamu?")