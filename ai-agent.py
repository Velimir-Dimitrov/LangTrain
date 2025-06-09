from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma:2b")

template = """
You are an expert in answering questions about stories

Here are some relevant story: {stories}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Ask your question ('E' to exit): ")
    if question.lower() == "e":
        break

    docs = retriever.invoke(question)
    stories = "\n\n".join(doc.page_content for doc in docs)

    if not stories:
        print("⚠️ No matching stories found.")
        continue

    result = chain.invoke({"stories": stories, "question": question})
    print(result)

