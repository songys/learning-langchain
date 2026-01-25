from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

prompt = """
https://news.ycombinator.com/front에서 주요 뉴스 5개와 조회수가 가장 높은 뉴스 5개를 정리해서 .md 파일로 저장해 주세요.
파일명은 [오늘 날짜]-HN News.md로 저장해 주세요.
각 뉴스의 제목과 링크, 작성일(YYYY-MM-DD)를 포함해 주세요.
"""
        

async def main():
    model = ChatOpenAI(model="gpt-4o-mini")

    async with MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest", "--headless"],
                "transport": "stdio",
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/mnt/d/projects/learning-langchain/python/appA"
                ]
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        agent_response = await agent.ainvoke({"messages": prompt})
        
        messages = agent_response.get('messages', [])
        for message in messages:
            if hasattr(message, 'content') :
                print(message.content)

if __name__ == "__main__":
    asyncio.run(main())
