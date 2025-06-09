# Playing around with LangChain

## What was done in the 'experiment' phase?
1. Secured API key and linked with the Gemini model
2. Using invoke() and the older equivalent predict() 
3. Concept of prompt templates and roles (system, human).
4. Using LangChain Expression Language (LCEL) instead "LLMChain" class

## Storybot phase:
1. Built a small vector database using Chroma and LangChain
2. Embedded some stories for children using Ollama + gemma:2b
3. Created an AI agent to answer questions about the stories

## Data Source: 
The stories in this project come from the public dataset:
**[Highly Rated Children Books and Stories – by Thomas Konstantin on Kaggle](https://www.kaggle.com/datasets/thomaskonstantin/highly-rated-children-books-and-stories)**
Used under the terms and conditions defined by the Kaggle dataset page.

## Requirements:
```text
python-dotenv~=1.1.0
langchain-google-genai~=2.1.5
langchain~=0.3.25
langchain-core==0.3.63
langchain-chroma==0.2.4
langchain-ollama==0.3.3
langchain-text-splitters==0.3.8
pandas==2.3.0
