from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document

from llm.meditron import BaseModel

class Retrievers(BaseModel):
    def __init__(self, dataset, vectorstore):
        self.vectorstore = vectorstore
        self.dataset = dataset

    def vectorstore_retriever(self, k: int):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def bm25_retriever(self, k: int):
        return BM25Retriever.from_texts(list(self.dataset["text"]), k=k)

    def bm25_doc_search(self, query, k: int):
        return self.bm25_retriever(k).get_relevant_documents(query)

    def ensemble_retriever(self, k: int):
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever(k), self.vectorstore_retriever(k)], weights=[0.5, 0.5])

    def doc_search(self, query: str, k: int):
        return self.vectorstore.similarity_search(
            k=k,  # returns top k most relevant chunks of text
            query=query,  # the search query
        )

    def reranker_retrieval(self, llm,k: int):
        compressor = LLMChainExtractor.from_llm(llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.ensemble_retriever(k)
        )
        return compression_retriever
    
    def filter_retrieval(self, emb_model, ensemble_retriever, splitter):
        from langchain.retrievers.document_compressors import EmbeddingsFilter


        embeddings_filter = EmbeddingsFilter(embeddings=emb_model, similarity_threshold=0.5)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=ensemble_retriever
        )

        from langchain.retrievers.document_compressors import DocumentCompressorPipeline
        from langchain_community.document_transformers import EmbeddingsRedundantFilter

        redundant_filter = EmbeddingsRedundantFilter(embeddings=emb_model)
        relevant_filter = EmbeddingsFilter(embeddings=emb_model, similarity_threshold=0.50)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        filter_retrieval = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
        )

        return filter_retrieval
