__import__('pysqlite3')
import os
import sys
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import chromadb
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain_community.document_transformers import LongContextReorder

load_dotenv()
os.environ['COHERE_API_KEY'] = os.getenv('API_KEY')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def populate_chroma_db():
    ### Load
    loader_1 = DirectoryLoader("../../data/raw/topic1", glob="./*.txt", loader_cls=TextLoader)
    docs1 = loader_1.load()
    loader_2 = DirectoryLoader("../../data/raw/topic2", glob="./*.txt", loader_cls=TextLoader)
    docs2 = loader_2.load()

    ### Embedding
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    # Make splits
    splits1 = text_splitter.split_documents(docs1)
    splits2 = text_splitter.split_documents(docs2)

    client_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory="chromadb",
        anonymized_telemetry=False,
    )

    Chroma.from_documents(
        collection_name="project_store_topic1",
        documents=splits1,
        persist_directory="chromadb",
        client_settings=client_settings,
        embedding=CohereEmbeddings(),
    )

    Chroma.from_documents(
        collection_name="project_store_topic2",
        documents=splits2,
        persist_directory="chromadb",
        client_settings=client_settings,
        embedding=CohereEmbeddings(),
    )

def to_doc(docs):
    return [doc.to_document() for doc in docs]

def answer_query(question):
    ### LLM
    llm = ChatCohere(model="command-r", format="json", temperature=0)

    client_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory="chromadb",
        anonymized_telemetry=False,
    )

    db_1 = Chroma(
        collection_name="project_store_topic1",
        persist_directory="chromadb",
        client_settings=client_settings,
        embedding_function=CohereEmbeddings(),
    )
    db_2 = Chroma(
        collection_name="project_store_topic2",
        persist_directory="chromadb",
        client_settings=client_settings,
        embedding_function=CohereEmbeddings(),
    )

    retriever_1 = db_1.as_retriever(
        search_type="similarity", search_kwargs={"k": 5} # TODO this is setting itself to 2 im not sure why
    )
    retriever_2 = db_2.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    lotr = MergerRetriever(retrievers=[retriever_1, retriever_2])


    # TODO These are the process used in the DocumentCompressorPipeline. I think di na kailangan yung
    #  filter_ordered_by_retriever since others don't implement it. Medyo mabagal kasi yung retrieval process
    #  I guess test it out and see if it's worth.

    filter = EmbeddingsRedundantFilter(embeddings=CohereEmbeddings())
    filter_ordered_by_retriever = EmbeddingsClusteringFilter(
        embeddings=CohereEmbeddings(),
        num_clusters=2,
        num_closest=1,
        sorted=True,
    )
    reordering = LongContextReorder()

    pipeline = DocumentCompressorPipeline(transformers=[filter, filter_ordered_by_retriever, reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )

    ### MultiQueryRetriever
    mq_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions. Output in a bullet list. Just output the bullet list and nothing else. Not even an intro text.
    Original question: {question}""",)
    generate_queries = mq_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    retriever = MultiQueryRetriever(
        retriever=compression_retriever, llm_chain=generate_queries
    )

    retrieve_todoc = retriever | to_doc

    # retriever = MultiQueryRetriever.from_llm(
    # retriever=vectorstore.as_retriever(search_kwargs={"k": 1}), llm=llm
    # )
        
    ### Generate
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, and 
        if there is no context provided, mention it. 
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

    ### Question Generator
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant trying to help a user
        generate 3 question that will help them understand the the following pieces of information below. Just output
        the question and nothing else.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    question_generator = prompt | llm | StrOutputParser()

    ### State
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """
        retry: int
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
        documents = retrieve_todoc.invoke({"question":question})
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
        retry =  state["retry"]

        if retry is None:
            retry = 0
        else:
            retry = retry + 1

        # RAG generation
        if (retry >= 2):  
             generation = "RAGBot cannot produce a coherent response based on the corpus."
        else:
             generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation, "retry": retry}

    def generate_question(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE QUESTION---")
        documents = state["documents"]

        generation = "The database does not have a useful answer for that question. I have generated more relevant questions " \
                     "that you may ask: \n" + question_generator.invoke({"context": documents})
        return {"documents": documents, "question": state["question"], "generation": generation, "retry": state["retry"]}

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry = state["retry"]

        # Check if retry count exceeds the limit
        if (retry >= 2):   
            print("Maximum retry limit reached. Stopping...")
            return "stop"
        
        print("---CHECK HALLUCINATIONS---")
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes" or grade == "1" or grade == 1:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes" or grade == "1" or grade == 1:
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "supported"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "question irrelevant"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("generate_question", generate_question)  # generate

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate_question", END)
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "question irrelevant": "generate_question",
            "supported": END,
            "stop": END,
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
    return value["generation"]

# populate_chroma_db()