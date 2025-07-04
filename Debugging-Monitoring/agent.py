from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START,END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "AgenticGraphLLM Project"
os.environ["LANGSMITH_TRACING"] = "true"

from langchain.chat_models import init_chat_model
llm = init_chat_model('groq:llama3-8b-8192')
llm

class State(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]


def make_tool_graph():
    ## Graph with tools

    @tool
    def add(a:int, b:int)->int:
        """Adding two numbers"""
        return a+b
    tools = [add]
    tool_node = ToolNode([add])
    llm_with_tools = llm.bind_tools([add])

    def call_llm_model(state:State):
        return {"messages":[llm_with_tools.invoke(state["messages"])]}
    
    ## Build the Graph
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest (message) from the assistant is a tool call -> tools_condition routes to tool
        # If the latest (message) from the assistant is not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools", "tool_calling_llm")

    ## Compile the graph
    graph_bldr = builder.compile()
    return graph_bldr

tool_agent = make_tool_graph()