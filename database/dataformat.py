import json
import re

# Load data
with open('/', 'r') as file:
    data = json.load(file)

data_test = data[:500]

print(data[0])
print(len(data))


from langchain.chains.summarize import load_summarize_chain

def summarize(doc: Document, meditron, chain_type="base"):

    chain = load_summarize_chain(meditron, chain_type="refine")

    summary = chain.run(doc[0])
    return summary

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
