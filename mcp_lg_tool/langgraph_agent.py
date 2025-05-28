from dotenv import load_dotenv
_ = load_dotenv()
import os
import operator
import json
import requests
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, Graph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
os.environ["LANGCHAIN_VERBOSE"] = "true"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

@tool
def calculate_operators(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

tools = [calculate_operators, TavilySearchResults(max_results=4)]

class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("print_summary", self.print_summary)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge(
            "action",
            "print_summary"
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def print_summary(self, state: AgentState):
        print("Agent Summary")
        print("--------------")
        print(f"state print_summary: {state}")
        print(f"System: {self.system}")
        print(f"Tools: {', '.join([t.name for t in self.tools.values()])}")
        print("--------------")
    
    def exists_action(self, state: AgentState):
        print(f"State exists_action: {state}")
        
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        print(f"State call_openai: {state}")
        
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        print(f"State take_action: {state}")
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo")  #reduce inference cost
# abot = Agent(model, tools, system=prompt)
# print(type(abot.graph))
# abot.graph.get_graph().draw_png('langgraph_agent.png')
# messages = [HumanMessage(content="What is the weather in sf? and calculate 1+2?")]
# result = abot.graph.invoke({"messages": messages})

# print("Final Result:", result)