from langchain_ollama import OllamaLLM

model = OllamaLLM(model='llama3.2')

response = model.invoke('하늘이')
print(response)
