import streamlit as st
import time

from elasticsearch import Elasticsearch
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

es_client = Elasticsearch('http://localhost:9200') 


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

#building the prompt
def build_prompt(query, search_results):
    prompt_template = """
You're a career assistant that helps people know more about data careers. Answer the QUESTION based on the CONTEXT from the knowledge.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f" content: {doc['body']}\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

#building RAG
def llm(prompt):
    response = client.chat.completions.create(
        model='gemma:2b',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query, title):
    search_results = elastic_search(query, title)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

#building streamlit UI
def main():
    st.title("Data Careers")

    # Add a dropdown for article selection
    article_list = ["Roles in a Data Team", "Building a Data Science Team", "Data Science Manager vs Data Science Expert", "Data Engineers Aren't Plumbers",
                    "Starting a Career as a Data Scientist", "Guidelines to Get a Data Engineer Job Against the Odds"]  # Replace with dynamic list if available
    selected_article = st.selectbox("Select an article that you want to get more from:", article_list)
    

    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        with st.spinner('Processing...'):
            output = rag(user_input, selected_article)
            st.success("Completed!")
            st.write(output)

if __name__ == "__main__":
    main()