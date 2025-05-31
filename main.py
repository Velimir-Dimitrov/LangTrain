import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get your API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the Gemini model
# You can choose different models like "gemini-pro" or "gemini-1.5-flash"
# Check Google AI Studio or Gemini API docs for available models.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that writes short, creative stories."),
    ("human", "Write a story about a {animal} who finds a {magical_item}.")
])

# Create a simple chain
chain = prompt | llm

# Invoke the chain with inputs
inputs = {"animal": "dragon", "magical_item": "glowing crystal"}
response = chain.invoke(inputs)

print(response.content)

# Another example: simple direct invocation
print("\n--- Direct Invocation ---")
direct_response = llm.invoke("Tell me a fun fact about giraffes.")
print(direct_response.content)