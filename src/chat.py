import openai
from vector_store import get_top_matches
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def query_chatbot(user_query):
    """Retrieves context and generates a chatbot response."""
    top_matches = get_top_matches(user_query)
    
    context = "\n".join(top_matches)  # Placeholder - needs actual text retrieval
    prompt = f"Use the following information to answer:\n{context}\nUser: {user_query}\nBot:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Bot:", query_chatbot(user_input))
