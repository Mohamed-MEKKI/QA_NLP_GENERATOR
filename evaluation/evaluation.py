import pandas as pd
import os
import discord
import asyncio
import json
import nest_asyncio
import numpy as np

from discord.ext import commands
from embedder import Emmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from llm.meditron import BaseModel
from chunker.splitters import Chunker
from config import variables

from database.pinecone import pinecone_storage
from generation_pipeline.query_transformation import generate_prompt

from generation_pipeline.query_transformation import data_format

from generation_pipeline.rag_generation_pipeline import generate_answer

from retrievers.retrievers import Retrievers
from torch import cuda, device
from dotenv import load_dotenv, find_dotenv


def load_local_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_pipeline(retriever, huggingface_wrapper):
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | huggingface_wrapper
        | StrOutputParser()
    )
    return rag_chain


def main():
    
    _ = load_dotenv(find_dotenv())

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    hf_api_key = os.getenv("HF_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")    

    #Load the modal
    meditron_model = BaseModel(model_id=variables.model_id , access_token=hf_api_key)
    
    data = load_local_dataset(os.path.dirname(os.path.realpath(__file__)) + '/source/output.json')
    # TODO: REMOVE THIS SLICING
    data_test = data[:500]

    chunker = Chunker(meditron_model.len_tokens_text)
    documents = chunker.format_document(data_test)


    dataset = data_format(documents)
    dataset.head()

    ##Embedding pipeline
    ####################

    #embedding_model = 'neuml/pubmedbert-base-embeddings'
    device = device("cuda" if cuda.is_available() else "cpu")
    emb = Emmbeddings(variables.embedding_id, device)   
    emb.print_stats("hello world")


    ##Pinecone storage configurations - create index, store the embeddings in Pinecone, show the stats of the index
    ################################
    p = pinecone_storage(emb)
    #TODO: Remove hardcoded pinecone index Name, store in constants
    ind = p.create_index("project-qa")
    #p.store_embeddings(dataset)
    p.show_stats("project-qa")


    ##Create VectorStore using PineconeVector
    ####################
    vectorstore = PineconeVectorStore(ind, emb.embed_model(), "text")

    
    return vectorstore, dataset, meditron_model.meditron(), chunker.splitter_function(), emb.embed_model()


if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    with open(dir_path+'evaluation_dataset.json', 'w') as file:
        eval_data = json.load(file)
    
    questions = list()
    ground_truth_answer = list()
    
    for question_answer_pair in eval_data:
        questions.append(question_answer_pair["question"])
        ground_truth_answer.append(question_answer_pair["answer"])
    
    # Call Rag Pipeline
    generated_answers = list()

    #fetch data, chunk and embed
    vectorstore, dataset, meditron_model_obj, splitter_function, embed_model = main()
    retriever = Retrievers(dataset, vectorstore)
    ensemble_retriever = retriever.ensemble_retriever(3)

    new_transformed_questions = [generate_prompt(ques) for ques in question]

    our_answers = rag_pipeline(ensemble_retriever, meditron_model_obj).batch(new_transformed_questions)[0]

    contexts = [ ensemble_retriever.get_relevant_documents(transformed_question) for transformed_question in new_transformed_questions ]
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")

    bleu_simple = bleu.compute(predictions=our_answers, references=ground_truth_answers)
    rogue_simple = rouge.compute(predictions=our_answers, references=ground_truth_answers)

    bertscore_simple = bertscore.compute(predictions=our_answers, references=ground_truth_answers, lang="en")
    bertscore_simple_averaged={}

    for key in bertscore_simple.keys():
        if key!='hashcode':
            bertscore_simple_averaged[key]=np.mean(bertscore_simple[key])

    print("Bleu Simple: ")
    print(bleu_simple)
    print("Rogue Simple: ")
    print(rogue_simple)
    print("Bert Score: ")
    print(bertscore_simple_averaged)


    



