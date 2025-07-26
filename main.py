import os
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define the messages state e.g.:
# 1) State = {"messages": [{"role": "user", "content": "Hello, how are you?"}] ///// metadata = add_messages()}
# 2) State = {"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "system", "content": "I'm good, thank you!"}]}
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None

# Pydantic type enforcer
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description = "Classify if the message requires an emotional (therapist) or logical response."
    )

graph_builder = StateGraph(State)

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the message as either:
            - 'emotional': if it asks for emotional/therapist support, deals with feelings or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions 
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    
    return {"message_type": result.message_type}

def route_message(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist_agent"}
    return {"next": "logical_agent"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
    {
        "role": "system",
        "content": """You are a compassionate therapist. Focus on the emotional aspects of the question. 
        Show empathy, validate the users feelings, and help them process their emotions.
        Ask thoughtful questions to help them explore thier feelings more openly.
        Avoid giving logical solutions unless explicitly asked. 
        """
    },
    {
        "role": "user",
        "content": last_message.content
    }
    ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
    {
        "role": "system",
        "content": """You are a purely logical assistant. Focus only on facts and information. 
        Provide clear, concise answers based on logic and evidence.
        Do not address emotions or provide emotional support.
        Be direct and straightforward in your responses.
        """
    },
    {
        "role": "user",
        "content": last_message.content
    }
    ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("route_message", route_message)
graph_builder.add_node("therapist_agent", therapist_agent)
graph_builder.add_node("logical_agent", logical_agent)

graph_builder.add_edge(START,"classify_message")
graph_builder.add_edge("classify_message","route_message")

graph_builder.add_conditional_edges(
    "route_message",
    lambda state: state.get("next"),
    {"therapist_agent": "therapist_agent", "logical_agent": "logical_agent"}
    )

graph_builder.add_edge("therapist_agent",END)
graph_builder.add_edge("logical_agent",END)

graph = graph_builder.compile()


def run_chatbot():

    # Initialise state
    state = {"messages": [], "message_type": None, "next": None}
    
    while True:
        user_input = input("Enter a message: ")
        if user_input == "exit":
            print("Bye!")
            break
        
        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        
        # state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})]
        state = graph.invoke(state)

        if state.get("messages") and len(state.get("messages")) > 0:
            last_message = state.get("messages")[-1]
            print(f"Assisstant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()