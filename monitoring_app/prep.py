import os
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from dotenv import load_dotenv
import json

from db import init_db

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL_LOCAL")
MODEL_NAME = os.getenv("MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

# BASE_URL = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"


def fetch_documents():
    print("Fetching documents...")
    with open('/home/purity/llm-zoomcamp/project/documents-with-ids.json', 'r') as file:  # Replace 'data.json' with the path to your file
        documents = json.load(file)
    print(f"Fetched {len(documents)} documents")
    return documents


def fetch_ground_truth():
    print("Fetching ground truth data...")
    df_ground_truth = pd.read_csv('/home/purity/llm-zoomcamp/project/ground-truth-data.csv')
    ground_truth = df_ground_truth.to_dict(orient='records')
    print(f"Fetched {len(ground_truth)} ground truth records")
    return ground_truth


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def setup_elasticsearch():
    print("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL,
    request_timeout=30,)

    index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "heading": {"type": "text"},
            "body": {"type": "text"},
            "question": {"type": "text"},
            "Title": {"type": "keyword"} ,
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Elasticsearch index '{INDEX_NAME}' created")

    return es_client

def clean_doc(documents):
    new_data = []

    for d in documents:
        # Check if the required key is present
        if 'question' not in d:
            continue

        # If all conditions pass, add the dictionary to the filtered list
        new_data.append(d)

    return new_data

def index_documents(es_client, documents, model):
    print("Indexing documents...")
    for doc in tqdm(documents):
        question = doc['question']
        text = doc['body']
        doc['question_text_vector'] = model.encode(question + ' ' + text)
        es_client.index(index=INDEX_NAME, document=doc)
    print(f"Indexed {len(documents)} documents")


def main():
    # you may consider to comment <start>
    # if you just want to init the db or didn't want to re-index
    print("Starting the indexing process...")

    documents = fetch_documents()
    ground_truth = fetch_ground_truth()
    # merging the ground truth dict and data dict on 'id' to get the questions on the data dict
    for i in documents:
        for j in ground_truth:
            if j['document'] == i['id']:
                # Add the `question` key and value from dict1 to dict2
                i['question'] = j['question']

    new_documents = clean_doc(documents) #getting rid of documents without questions
    
    model = load_model()
    es_client = setup_elasticsearch()
    index_documents(es_client, new_documents, model)
    # you may consider to comment <end>

    print("Initializing database...")
    init_db()

    print("Indexing process completed successfully!")


if __name__ == "__main__":
    main()