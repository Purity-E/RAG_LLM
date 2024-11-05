import os
import time
import json

from openai import OpenAI

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")


es_client = Elasticsearch(ELASTIC_URL)
ollama_client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


#building the elastic search model
def elastic_search(query, title, index_name="data-roles"):
    search_query = {
        "size": 2,  # Retrieve only the top result based on the highest score
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["heading^4", "body"],  # Boost 'heading' field
                            "type": "best_fields",  # Best field matching strategy
                            "fuzziness": "AUTO"  # Allow slight variations in text
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "Title": title  # Make sure 'Title' field is indexed as a keyword
                        }
                    }
                ]
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs


def elastic_search_knn(field, vector, title):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "Title": title
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["body", "question", "Title", "id"]
    }

    es_results = es_client.search(
        index="data-roles-questions",
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs


def build_prompt(query, search_results):
    prompt_template = """
You're a career assistant that helps people know more about data careers. Answer the QUESTION based on the CONTEXT from the knowledge base.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"title: {doc['Title']}\n, content: {doc['body']}\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model_choice):
    start_time = time.time()
    if model_choice.startswith('ollama/'):
        response = ollama_client.chat.completions.create(
            model=model_choice.split('/')[-1],
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return answer, tokens, response_time


def evaluate_relevance(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, 'ollama/gemma:2b')
    
    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens



def get_answer(query, course, model_choice, search_type):
    if search_type == 'Vector':
        vector = model.encode(query)
        search_results = elastic_search_knn('question_text_vector', vector, course)
    else:
        search_results = elastic_search(query, course)

    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model_choice)
    
    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)
 
    return {
        'answer': answer,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'model_used': model_choice,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
    }