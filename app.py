import pandas as pd
import numpy as np
import openai
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = st.secrets['keys']

def load_data_and_embeddings():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    # Convert string representations of embeddings back into numpy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=','))
    embeddings = np.array(df['Question_Embedding'].tolist())
    return df, embeddings

def get_openai_embedding(question):
    response = openai.Embedding.create(input=[question], engine="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

def process_question(user_question):
    # Generate embedding for user question
    user_embedding = get_openai_embedding(user_question).reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding, embeddings)
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[0][most_similar_index]

    # Set a similarity threshold
    threshold = 0.7  # Adjust as needed

    if similarity_score > threshold:
        answer = df['Answer'][most_similar_index]
        return f"Answer: {answer}\nSimilarity Score: {similarity_score:.2f}"
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

def clear_input():
    st.session_state.user_question = ""
    
def main():
    st.title("Health Question Answering")

    user_question = st.text_input("Ask your health question")
    submit = st.button("Submit")

    clear = st.button("Clear")

    if submit and user_question:
        answer = process_question(user_question)
        st.write(answer)

    if clear:
        clear_input()

# Load data and embeddings
df, embeddings = load_data_and_embeddings()

# Run the Streamlit app
if __name__ == '__main__':
    main()
