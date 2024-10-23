from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm


class Chunker:

    def __init__(self, len_tokens_text):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=50,
            length_function=len_tokens_text,
            separators=['\n\n', '\n', ' ', '']
        )
    
    def splitter_function(self):
        return self.splitter

    def chunk_text(self, text):
        return self.splitter.split_text(text)

    def format_document(self, data_test):
        documents = []
        for doc in tqdm(data_test):
            author = doc['authors']
            date = doc['publication_date']
            uid = doc['PMID']
            chunks = self.chunk_text(doc['abstract'])
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f'{uid}-{i}',
                    'text': chunk,
                    'date': date,
                    'authors': author
                })
        return documents

    def len_documents(self, documents):
        return len(self.format_document(documents))
