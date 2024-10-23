import pandas as pd
from langchain.docstore.document import Document


def get_documents(data_test):
    documents = [doc["text"] for doc in data_test["document"]]
    questions = [doc["authors"] for quest in data_test["authors"]]
    answers = [doc["date"] for ans in data_test["date"]]
    documents = list(set(documents))
    return documents


def data_format(documents):
    dataset = pd.DataFrame(documents)
    return dataset


def show_data(dataset):
    print(dataset.head())
