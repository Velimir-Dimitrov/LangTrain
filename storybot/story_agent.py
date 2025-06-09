from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from story_vectorizer import retriever # Import vector search retriever

# Load the language model from Ollama (locally hosted LLM)
model = OllamaLLM(model="gemma:2b")

# Define the structured prompt that will be sent to the LLM
# Keeping this clear helps improve model understanding and consistency
template = """
You are an expert in answering questions about children's stories.

Below are some relevant story excerpts:

{stories}

Answer the following question based on the above stories:

{question}
"""

# Compile the prompt into a LangChain "prompt template"
prompt = ChatPromptTemplate.from_template(template)

# Chain together prompt → model to form the end-to-end pipeline
chain = prompt | model

# Main interactive loop: takes user input and returns LLM response
while True:
    question = input("Ask your question ('E' to exit): ").strip() #strip for cleaner query
    if question.lower() == "e":
        break

    # Retrieve relevant stories from the vector store
    # Extract story content from each document for the prompt
    docs = retriever.invoke(question)
    stories = "\n\n".join(doc.page_content for doc in docs)

    if not stories:
        print("No matching stories found.")
        continue

    # Send the prompt to the model and display the response
    result = chain.invoke({"stories": stories, "question": question})
    print(result)
