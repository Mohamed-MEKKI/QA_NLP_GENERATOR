# QASystem-INLPT-WS2023

---

# ***Question Answering System for Medical Data using RAG***

---

### Question Answering System for the Medical Data: 
Abstracts concerning “intelligence” from 2013 till 2023 in the [PubMed Articles](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle/abstract%5D&filter=simsearch1.fha&filter=years.2013-2023&sort=date)

---

### Group member details:

| Name                        | Matriculation Number | GitHub Username | Email Addresses                                                 |
|-----------------------------|----------------------|-----------------|-----------------------------------------------------------------|
| Karthik Thirthahalli Ganesh | 3770883              | karthikaf       | tgkarthik007@gmail.com, karthik@stud.uni-heidelberg.de          |
| Mohamed Mekki               | 4733974              | Mohamed-MEKKI   | mohamed.mekki@edu.ece.fr, mohamed.mekki@stud.uni-heidelberg.de  | 
| Raghav Kundra               | 4732299              | entunidotdeb    | raghavkundrasan@gmail.com, raghav.kundra@stud.uni-heidelberg.de |
| Waheb Zaidi                 | 3768236              | Zaldl           | zaidiwahebhasan@gmail.com, waheb.zaidi@stud.uni-heidelberg.de   | 

### Advisor: Nicolas Reuter (reuter@informatik.uni-heidelberg.de)

---

### Get Started

0. Ensure you have python's 3.10 or more version. 
(Preferrable version 3.10.X as we are able to use all mentioned libraries in requirements file with this version on kaggle) 
```bash
python --version
```

```
OPENAI_API_KEY = "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"
HF_API_KEY = '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'
PINECONE_API_KEY = '73849156-4645-4fab-a129-8f5ee01dc8dd'
```

1. Create a virtual environment 
```bash
python -m venv ve
```

2. Activate virtual environment 
```bash
source ve/bin/activate
```

3. Clone the repository

```bash
git clone https://github.com/karthikaf/QASystem-INLPT-WS2023.git

```
4. Join the Discord server with this link: [Discord](https://discord.gg/KzMuHzyQ)
5. Then you need to Create an .env file with [openai api key](https://platform.openai.com/api-keys), [Pinecone api key](https://app.pinecone.io/), [Hugging Face Access Token](https://huggingface.co/settings/tokens)
Run these in your terminal
```bash
cd QASystem-INLPT-WS2023
touch .env
```
```bash
echo PINECONE_API_KEY="<INSERT-YOUR-KEY-HERE>" >> .env
echo HF_API_KEY="<INSERT-YOUR-KEY-HERE>" >> .env
echo OPENAI_API_KEY="<INSERT-YOUR-KEY-HERE>" >> .env
```   


6. Unzip the dataset to obtain dataset in JSON
```bash
unzip ./src/source/output.json.zip -d ./QASystem-INLPT-WS2023/src/source/
```   

7. Install requirements.
```bash
pip instll -r requirements.txt
```

8. Run the main function in the src folder
 ```bash
python ./src/main.py
```
9. Start your questions in the chatbot always with `!Query`

---

### Dataset

  - Source: [PubMed Articles](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle/abstract%5D&filter=simsearch1.fha&filter=years.2013-2023&sort=date)
  - Format: 
    - PMID
    - Abstract
    - Publication Date
    - Authors
    - Example:
![data.jpg](images/data.jpg)

---

Note: 
- GPU is required for the computation of the RAG model.
- In case of malfunction when executing the main.py, run only the function `main` inside the main.py file.
- We have also uploaded the ipynb notebook files (with and without evaluation) for the code in the src folder.
