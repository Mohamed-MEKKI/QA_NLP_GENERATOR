"""## *Pinecone storage configuration*"""

from embedder import Emmbeddings
from torch import cuda, device
import pinecone
import time
import os
import tqdm
import logging as log

class pinecone_storage:
    def __init__(self, access_token,embeddings):
        #Establish pinecone connection
        #self.pinecone = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), environment='us-central1-gcp')
        self.pinecone = pinecone.Pinecone(api_key=access_token, environment='us-central1-gcp')
        self.embeddings = embeddings

    def create_index(self, index_name):
        # get list of existing indexes
        existing_indexes = [
            index_info["name"] for index_info in self.pinecone.list_indexes()
        ]

        # check if index already exists (it shouldn't if this is first time)
        if index_name not in existing_indexes:
            # if does not exist, create index
            self.pinecone.create_index(
                index_name,
                dimension=768,  # dimensionality of minilm
                metric='cosine',
                spec=pinecone.PodSpec(environment='us-central1-gcp')
            )
            # wait for index to be initialized
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
            #create index
            index_name="project-qa"
        log.info(f"Index {index_name} created")
        return self.pinecone.Index(index_name)
    
    def delete_index(self, index_name):
        self.pinecone.delete_index(index_name)
        log.info(f"Index {index_name} deleted")

    def show_stats(self,index_name):
        self.pinecone_index = self.pinecone.Index(index_name)
        self.pinecone_index.describe_index_stats()


    def store_embeddings(self,dataset):

        batch_size = 32
        # to be changed according to our data final shape

        for i in tqdm(range(0, len(dataset), batch_size)):
            i_end = min(len(dataset), i + batch_size)
            batch = dataset.iloc[i:i_end]
            ids = [f"{x['id']}" for i, x in batch.iterrows()]
            texts = [x['text'] for i, x in batch.iterrows()]
            embeds = self.embeddings.embed_documents(texts)
            # get metadata to store in Pinecone
            metadata = [
                {'text': x['text'],
                 'authors': x['authors']} for i, x in batch.iterrows()
            ]
            print(metadata)
            self.pinecone_index.upsert(vectors=zip(ids, embeds, metadata))


if __name__ == "__main__":
    #define the embedding model
    embedding_model = 'neuml/pubmedbert-base-embeddings'
    device = device("cuda" if cuda.is_available() else "cpu")

    emb = Emmbeddings(embedding_model, device)   
    embed_function = emb.embed_function("hello world")
    #print(embed_function)
    emb.print_stats("hello world")
    embedding_model = 'neuml/pubmedbert-base-embeddings'
    pc = pinecone_storage(embed_function)
    pc.create_index(embed_function, "test_index")
    #using T4-GPU for computation
    #device = device("cuda" if cuda.is_available() else "cpu")