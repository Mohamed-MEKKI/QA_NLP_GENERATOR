import pandas as pd
import os
import discord
import asyncio
import json
import nest_asyncio

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


def main(question):
    
    _ = load_dotenv(find_dotenv())


    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    hf_api_key = os.getenv("HF_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")    

    #Load the modal
    meditron_model =  BaseModel(model_id=variables.model_id , access_token=hf_api_key)

    #question format
    new_question = generate_prompt(question, openai_api_key)
    
    print(new_question)
    
    data = load_local_dataset(os.path.dirname(os.path.realpath(__file__)) + '/source/output.json')
    # TODO: REMOVE THIS SLICING
    data_test = data[:500]

    chunker = Chunker(meditron_model.len_tokens_text)
    documents = chunker.format_document(data_test)


    dataset = data_format(documents)
    dataset.head()

    ##Embedding pipeline
    ####################
    from torch import device
    #embedding_model = 'neuml/pubmedbert-base-embeddings'
    device = device("cuda" if cuda.is_available() else "cpu")
    emb = Emmbeddings(variables.embedding_id, device)   
    emb.print_stats("hello world")


    ##Pinecone storage configurations - create index, store the embeddings in Pinecone, show the stats of the index
    ################################
    p = pinecone_storage(pinecone_api_key,emb)
    #TODO: Remove hardcoded pinecone index Name, store in constants
    ind = p.create_index("project-qa")
    #p.store_embeddings(dataset)
    p.show_stats("project-qa")


    ##Create VectorStore using PineconeVector
    ####################
    vectorstore = PineconeVectorStore(ind, emb.embed_model(), "text")

    
    # return vectorstore, dataset, meditron_model.meditron(), chunker.splitter_function(), emb.embed_model()
    ####################


    ##Initialise and test the Retreievers
    ####################
    retriever = Retrievers(dataset, vectorstore)
    retriever.doc_search("hello world", 3)
    retriever.bm25_retriever(k=3)
    retriever.bm25_doc_search("world", 3)


    ##Pinecone Retriever
    ####################
    pinecone_retriever = retriever.vectorstore_retriever(3)


    ##Ensemble Retriever
    ####################
    ensemble_retriever = retriever.ensemble_retriever(3)

    ##Compression Retriever
    ####################
    compression_retriever=retriever.reranker_retrieval(meditron_model.meditron(), k=3)

    #Filter
    ####################
    filter = retriever.filter_retrieval(emb_model=emb.embed_model(), splitter=chunker.splitter_function(), ensemble_retriever=ensemble_retriever)

    #Initiate different pipelines
    ####################    

    #simple retriever
    simple_rag = rag_pipeline(pinecone_retriever, meditron_model.meditron()).invoke(new_question)

    #Hybird search
    advanced_rag = rag_pipeline(ensemble_retriever, meditron_model.meditron()).invoke(new_question)

    #Reranking
    advanced_rag_compression = rag_pipeline(compression_retriever, meditron_model.meditron()).invoke(new_question)

    #Reranking + filtering
    advanced_rag_compression_with_filter = rag_pipeline(filter, meditron_model.meditron()).invoke(new_question)
    
    return simple_rag, advanced_rag, advanced_rag_compression, advanced_rag_compression_with_filter
   


if __name__ == "__main__":

    BOT_TOKEN = 'MTIxMjgzMzIzMTM1MTcwOTgyNw.G7dDnN.lQmnOrtmz15hMtDPVkiEgiLrjgCnSyDOEaVRHc'
    CHANNEL_ID = 1212847612315701338

    bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

    nest_asyncio.apply()


    @bot.event
    async def on_ready():
        print("Hello! MediBot is ready!")
        channel = bot.get_channel(CHANNEL_ID)
        await channel.send("Hello! MediBot is ready!")


    user_queries = {}


    @bot.command()
    async def hello(ctx):
        await ctx.send("Hello!")


    @bot.command()
    async def query(ctx, *, message):
        user_queries[ctx.author.id] = message
        print(message)
        simple_rag, advanced_rag, advanced_rag_compression, advanced_rag_compression_with_filter=main(message)
        #choose the type of response
        await ctx.send(advanced_rag_compression_with_filter)


    async def mymain(BOT_TOKEN):
        bot.run(BOT_TOKEN)


    loop = asyncio.get_event_loop()
    loop.run_until_complete(mymain(BOT_TOKEN))

    ##PS: To run the rag pipeline (function main) without the bot:
    #main()
