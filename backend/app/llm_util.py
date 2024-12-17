from openai import OpenAI
import os 

api_key = "" # TODO: Add your OpenAI API key here
client = OpenAI(api_key=api_key)


def generate_embeddings(client, text: str, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def generate_embeddings_ollama(embed_model, text: str):
    embedding = embed_model.embed_query(text)
    return embedding

def generate_openai(client, query: str, context: str):
    user_prompt = f""" You need to answer the question in the sentences as same as in the pdf content. .
    Given below is the context and question of the user.
    context = {context}
    question = {query}
    if the answer is not in the pdf answer "i don't know what the hell you are asking about"
    answer in the same language as in the question.    
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful QA assistant, skilled in answering questions with given context."},
            {"role": "user", "content": user_prompt}
        ]
    )
    answer = completion.choices[0].message.content
    return answer 

def generate_ollama(llm, query: str, context: str):
    response = llm.generate(
        context=context,
        query=query,
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response.choices[0].text