from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

# 1. Define the schema for the agent's memory state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define the core node (the thinking brain of the agent)
def call_model(state: AgentState):
    # Initializes the LLM model
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build the graph structure
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)

# Set execution flow: START -> agent -> END
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 4. Compile the graph into an executable application
graph = workflow.compile()
