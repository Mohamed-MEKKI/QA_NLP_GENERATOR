import torch

from llm.meditron import BaseModel
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



class generate_answer(BaseModel):
    def __init__(self, api_key):
        #self.retriever = retriever
        self.model = BaseModel(model_id="epfl-llm/meditron-7b", access_token=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_pipeline(self,retriever):
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
            | self.model.meditron()
            | StrOutputParser()
        )

        return rag_chain
