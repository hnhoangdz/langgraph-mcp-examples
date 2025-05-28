import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=OPENAI_API_KEY)


async def main():
    client = MultiServerMCPClient(
        {
            "test_agent": {
                "command": "uv",
                "args": ["run", "mcp_lg_tool"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print(tools)
    resources = await client.get_prompt("test_agent", "research-assistant")
    print(resources)

    agent = create_react_agent(llm, tools)
    resp = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "give me short introduction about To Lam"}
            ]
        }
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
