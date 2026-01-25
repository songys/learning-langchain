from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class Joke(BaseModel):
    setup: str = Field(description='농담의 설정')
    punchline: str = Field(description='농담의 포인트')


model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
model = model.with_structured_output(Joke)

result = model.invoke('고양이에 대한 농담을 만들어 주세요.')
print(result)
