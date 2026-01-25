from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# 구성 요소

template = ChatPromptTemplate.from_messages(
    [
        ('system', '당신은 친절한 어시스턴트입니다.'),
        ('human', '{question}'),
    ]
)

model = ChatOpenAI(model='gpt-3.5-turbo')

# 함수로 결합한다
# 데코레이터 @chain을 추가해 작성한 함수에 Runnable 인터페이스를 추가한다

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


# 사용한다

response = chatbot.invoke({'question': '거대 언어 모델은 어디서 제공하나요?'})
print(response.content)