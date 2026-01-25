from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

# 샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.'),
    HumanMessage(content='안녕하세요! 나는 민혁입니다.'),
    AIMessage(content='안녕하세요!'),
    HumanMessage(content='바닐라 아이스크림을 좋아해요.'),
    AIMessage(content='좋네요!'),
    HumanMessage(content='2 + 2는 얼마죠?'),
    AIMessage(content='4입니다.'),
    HumanMessage(content='고마워요.'),
    AIMessage(content='천만에요!'),
    HumanMessage(content='즐거운가요?'),
    AIMessage(content='예!'),
]

# 축약 설정
trimmer = trim_messages(
    max_tokens=65,
    strategy='last',
    token_counter=ChatOpenAI(model='gpt-4o-mini'),
    include_system=True,
    allow_partial=False,
    start_on='human',
)

# 축약 적용
trimmed = trimmer.invoke(messages)
print(trimmed)
