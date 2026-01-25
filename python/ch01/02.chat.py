from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

model = ChatOllama(model='llama3.2')
prompt = [HumanMessage('프랑스의 수도는 어디인가요?')]

response = model.invoke(prompt)
print(response.content)
