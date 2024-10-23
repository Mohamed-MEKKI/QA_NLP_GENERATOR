import os

from torch import cuda, device
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


class Emmbeddings:

    def __init__(self, embedding_model: str,device):
        #define the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )

    def embed_model(self):
        return self.embeddings
        
    def embed_function(self, documents):
        if not documents:
            return ValueError
        return self.embeddings.embed_documents(documents)

    def print_stats(self, documents):
        if not documents:
            return ValueError
        # print results
        print("number of docs:", len(self.embed_function(documents)))
        print("dimension of docs:", len(self.embed_function(documents)[0]))


if __name__ == "__main__":
    embedding_model = 'neuml/pubmedbert-base-embeddings'
    device = device("cuda" if cuda.is_available() else "cpu")

    emb = Emmbeddings(embedding_model, device)   
    #embed_function = emb.embed_function("hello world")
    #print(embed_function)
    #emb.print_stats("hello world")
    #p=emb.embed_query()
    #print(p("hello world"))
