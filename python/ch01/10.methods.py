from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()

completion = model.invoke('반가워요!')
# 안녕하세요! 만나서 반가워요. 무엇을 도와드릴까요?

completions = model.batch(['반가워요!', '잘 있어요!'])
# ['안녕하세요! 만나서 반가워요! 어떻게 도와드릴까요?', '감사합니다! 제가 여기 있어서 도와줄 수 있는 일이 있으면 말씀해주세요. ^^']

for token in model.stream('잘 있어요!'):
    print(token)
# 잘
# 가
# !
