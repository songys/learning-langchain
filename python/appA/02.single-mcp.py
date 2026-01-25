from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o-mini")

# MCP 서버 설정
server_params = StdioServerParameters(
    command="npx",
    args=["@playwright/mcp@latest", "--headless"],
)
prompt = "https://news.ycombinator.com/front에서 주요 뉴스 5개와 조회수가 가장 높은 뉴스 5개를 정리해주세요."

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": prompt})
            
            messages = agent_response.get('messages', [])
            for message in messages:
                if hasattr(message, 'content') :
                    print(message.content)
                    

# 비동기 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
