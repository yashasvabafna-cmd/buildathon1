from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

import pandas as pd
import pprint

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from promptstore import orderPrompt, conversationPrompt, agentPrompt
from classes import Item, Order
from utils import makeRetriever, get_context
from misc.sample_input import sample_sequences

import warnings
warnings.filterwarnings("ignore")
# from dotenv import load_dotenv
# load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
memory= MemorySaver()



class State(TypedDict):
    messages: Annotated[list, add_messages]

state = State(messages=[])

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("datafiles/testmenu100.csv")


# from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = init_chat_model("ollama:llama3.1")
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
retriever = makeRetriever(menu, search_type="similarity", k=10)


@tool
def extract_order(user_input: str) -> Order:
   """
    Extracts a single, structured order JSON from user input.
    ALWAYS use this tool when the user is explicitly placing an order.
    The function must be called only once per user turn,
    and its purpose is to parse the entire user request into a single order."""
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
    """Answer questions about the menu. If the user asks about the menu, this tool will be used to answer the question. Repeat the order if the user asks about the order.(Extract it from the chat history)"""
    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": []
    })
    return ai_response.content

@tool
def order_summary(order_history: Annotated[list, list.__add__]) -> str:
    """Summarize the order. This tool summarizes the order by looking at the JSON output from the extract_order tool in the order_history. The user is asking to summarize the order or 'what did I order?'"""
    if order_history:
        
        most_recent_order = order_history[-1]
        return f"Your current order is: {most_recent_order}. Would you like to confirm this order?"
    else:
        return "No order found. Please place an order first."

tools = [extract_order, menu_query, order_summary]

llm_with_tools=llm.bind_tools(tools)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

## Graph
def update_order_history(state: State):
    last_message = state['messages'][-1]
    if last_message.name == 'extract_order':
        order_json = last_message.content
        return {"order_history": [order_json]}
    return None

## Node Condition
def tool_calling_llm(state: State):
   return {"messages": [llm_with_tools.invoke([state['messages'][-1]])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("update_order_history", update_order_history)
builder.add_node("tools", ToolNode(tools))

## Edges
# Define the single, unambiguous entry point
builder.add_edge(START, "tool_calling_llm")

# Define the conditional edges from the 'tool_calling_llm'
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    }
)

# Define the conditional edges from the 'tools' node
builder.add_conditional_edges(
    "tools",
    lambda state: state['messages'][-1].name == "extract_order",
    {
        True: "update_order_history",
        False: "tool_calling_llm"
    }
)

# Define the edge from the 'update_order_history' node
builder.add_edge("update_order_history", "tool_calling_llm")

# graph building and saving the memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"1"}}
# messages = ['I want 3 Burgers, 1 burger from this should have extra cheese and another should have a beef patty and a ginger beer']

# for chunk in graph.stream(state, config=config,stream_mode="values"):
#     print(chunk['messages'][-1].content)
    # last_message = chunk['messages'][-1]
    # if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
    #     for tool_call in last_message.tool_calls:
    #         if tool_call['name'] == 'order_summary':
    #               print("The agent is calling the order_summary tool.", message=last_message.content)

chat = True
if not chat: print("Using preset inputs.")

i = 0
j = 0
with open(f"chat_history_{datetime.now().strftime("%m:%d::%H:%M")}.txt", "x") as f:

    while True:
        if chat:
            user_input = input("You: ")
        else:
            if i >= len(sample_sequences):
                print("No more sample inputs.")
                break
            user_input = sample_sequences[i][j]
            j += 1
            if j >= len(sample_sequences[i]):
                i += 1
                j = 0
                f.write("\n\nNew Sequence\n")
        
        if not chat:
            f.write(f"User: {user_input}\n")

        if user_input.lower() in {"quit", "exit"}:
            print("Chatbot: Goodbye!")
            break
        
        state["messages"].append({"role": "user", "content": user_input})

        for update in graph.stream(state, config=config):
            for step, output in update.items():
                if not output:
                    if chat:
                        print(f"Step {step} has no output.")
                    else:
                        f.write(f"Step {step} has no output.\n")
                elif "messages" in output:
                    for m in output["messages"]:
                        if not isinstance(m, HumanMessage) and hasattr(m, "content"):
                            pretty_str = m.pretty_repr()
                            if not chat:
                                f.writelines(f"{pretty_str}\n")
                            else:
                                print(pretty_str)
                                # print(f"{m.name} - {m.content}")
                else:
                    if chat:
                        print(f"Step {step} has no messages. Raw output: {output}")
                    else:
                        f.write(f"Step {step} has no messages.\nRaw output: {output}\n")
        
        if not chat:
            print(f"progress - {i}, {j}")

                # if last_message.tool_calls[0]['name'] == "extract_order":
            #     messages.append(last_message)
    # for update in graph.stream(state, config=config):
    #         for step, output in update.items():
    #             if "messages" in output:
    #                 for m in output["messages"]:
    #                     if isinstance(m, (AIMessage, ToolMessage)):
    #                         m.pretty_print()                 

            # if last_message.tool_calls[0]['name'] == "extract_order":
            #     messages.append(last_message)