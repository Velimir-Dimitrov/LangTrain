import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables into the program from .env file
load_dotenv()

# Get your API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")


# Initialize the Gemini model since others like ChatGPT and Grok require subscription
# Pick the model "gemini-1.5-flash" since it's the free one at the moment
# Provide the API Key variable and store the model in a variable
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)


# Quick test with 'predict' (an older method, primarily for demonstration
# or compatibility with older LangChain versions. 'invoke' is the modern standard.)
result = llm.predict("Your opinion on Bulgaria")
print(result)


# Another example: simple direct invocation (replacing "predict")
print("\n--- Direct Invocation ---")
direct_response = llm.invoke("Tell me a fun fact about giraffes.")
print(direct_response.content)


# Define a prompt template (our "script" for the AI)
# We use "system" to set AI's persona, "human" to define user's input
# Other popular roles like: ("ai", ...) or ("assistant", ...) provide conversational history for multi-turn interactions
# We use {placeholders} that will be filled in later
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that writes short, creative stories."),
    ("human", "Write a story about a {animal} who finds a {magical_item}.")
])

# Create a simple chain by taking the output from "prompt" and "piping" (the '|' operator) it into the 'llm'
# This "pipe"operator comes from LangChain Expression Language (LCEL)
chain = prompt | llm

# Old option would be using the class "LLMChain", import from langchain.chains instead LCEL
# chain = LLMChain(prompt=prompt, llm=llm)

# Invoke the chain with inputs on the "blanks"
inputs = {"animal": "dragon", "magical_item": "glowing crystal"}
response = chain.invoke(inputs)
print(response.content)


