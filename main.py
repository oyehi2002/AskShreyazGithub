import os
from typing import TypedDict, Sequence, Annotated
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import add as add_messages
from dotenv import load_dotenv


load_dotenv()


print("Current working directory:", os.getcwd())
print("USER_AGENT from .env:", os.getenv('USER_AGENT'))
print("USER_AGENT in os.environ:", os.environ.get('USER_AGENT', 'NOT SET'))

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)


class RAGstate(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
}

websites = ["https://github.com/oyehi2002/stockanalysisAI", "https://github.com/oyehi2002/AIreserarchagentb2b",
            "https://github.com/oyehi2002/autoorganizeAI", "https://github.com/oyehi2002/AIautoimagecap", "https://github.com/oyehi2002/secmsg",]

loader = WebBaseLoader(web_paths=websites, header_template=headers)
loaded_file = loader.load()
textsplit = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
embed = textsplit.split_documents(loaded_file)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=embed, embedding=embeddings)
# print(vectorstore._collection.count())
# print(vectorstore._collection.get())- debug statements for sake of seeing whats in our vectorstore

retriver = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})


@tool
def retriver_tool(query: str) -> str:
    '''Retrieve relevant information from Shreya Ramraika's Github using the vectorstore retriever.'''

    ans = retriver.invoke(query)

    if not ans:
        return f"No relevant info found in Shreya Ramraika's github"

    results = []
    for info in ans:
        results.append(info.page_content)
    return "\n\n".join(results)


tools = [retriver_tool]
llm = llm.bind_tools(tools)


def should_continue(state: RAGstate):
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # printing debug statements so we know what AI is doing behind the scenes
        print("DEBUG: Tool calls found, continuing to tools")
        return "tools"

    # If no tool calls, end the conversation
    print("DEBUG: No tool calls, ending")
    return "end"


def agent_start(state: RAGstate) -> RAGstate:

    system_prompt = """
You are an intelligent AI assistant who answers questions about Shreya Ramraika's github based on the websites loaded into your knowledge base.
Use the retriever tool available to answer questions about the question people would ask about her code and data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)]+messages
    msg = llm.invoke(messages)
    return {'messages': [msg]}


tool_node = ToolNode(tools=tools)

graph = StateGraph(RAGstate)
graph.add_node("start", agent_start)
graph.add_node("tools", tool_node)

graph.set_entry_point("start")
graph.add_conditional_edges("start", should_continue, {
                            "tools": "tools", "end": END})
graph.add_edge("tools", "start")

ready_bot = graph.compile()

while True:
    user_input = input(
        "\n (Type exit/quit to end) \nPlease enter what you wanna know about Shreya Ramraika's github: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    messages = [HumanMessage(content=user_input)]

    try:
        result = ready_bot.invoke(
            {"messages": messages},
            config={"recursion_limit": 10}
        )
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again with a different question.")
