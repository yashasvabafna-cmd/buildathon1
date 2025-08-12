from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import pprint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain.chat_models import init_chat_model
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from promptstore import orderPrompt, conversationPrompt, agentPrompt
from Classes import Item, Order
from utils import makeRetriever, get_context

import warnings
warnings.filterwarnings("ignore")

# Setup and definitions

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # intermediate_steps: Annotated[list, add]


parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("testmenu100.csv")

# Defining chains and tools

llm = init_chat_model("ollama:llama3.1")
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
retriever = makeRetriever(menu, search_type="similarity", k=10)

@tool
def extract_order(user_input: str) -> Order:
    """Extract structured order JSON from user input."""
    try:
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        return result.model_dump_json()
    except Exception as e:
        return f"Error parsing order: {str(e)}"
    
@tool
def menu_query(user_input: str) -> str:
    """Answer questions about the menu."""
    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": []
    })
    return ai_response.content

tools = [extract_order, menu_query]

# llm = ChatNVIDIA(
#     model="qwen/qwen2_5-7b-instruct",
#     nvidia_api_key="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")
# print(llm.invoke("hello").content)

agent = create_react_agent(model=llm, tools=tools, prompt=agentPrompt)
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)


# use ChatNVIDIA when bug is resolved

def agent_node(state: State):
    messages = state["messages"]
    # print(state["messages"])
    # print(state["intermediate_steps"])
    response = agent.invoke({"messages": messages})
    # print(response["agent_scratchpad"])
    return {"messages": response["messages"]}

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

agent_graph = graph.compile()

state = {
    "messages": [],
    # "intermediate_steps": []
}

config = {"configurable": {"thread_id": "def234"}}

while True:
    user_input = input("You: ")

    if user_input.lower() in {"quit", "exit"}:
        print("Chatbot: Goodbye!")
        break
    
    state["messages"].append({"role": "user", "content": user_input})

    for update in agent_graph.stream(state, config=config):
        for step, output in update.items():
            if "messages" in output:
                for m in output["messages"]:
                    if isinstance(m, (AIMessage, ToolMessage)):
                        m.pretty_print()