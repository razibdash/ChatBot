import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Initialize the ChatGroq model
model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
)



# Define the messages to send to the chatbot
messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

response = model(messagesToTheChatbot)


print("\n----------\n")

print("My favorite color is blue.")

print("\n----------\n")
print(response.content)

print("\n----------\n")

response = model.invoke([
    HumanMessage(content="What is my favorite color?")
])

print("\n----------\n")

print("What is my favorite color?")

print("\n----------\n")
print(response.content)

print("\n----------\n")
# Save the chat history to a file
chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    model, 
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is red.")],
    config=session1,
)

print("\n----------\n")

print("My favorite color is red.")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What's my favorite color? (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session2,
)

print("\n----------\n")

print("What's my favorite color? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What's my favorite color? (in session1 again)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi name is Julio.")],
    config=session2,
)

print("\n----------\n")

print("Mi name is Julio. (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=session2,
)

print("\n----------\n")

print("What is my name? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("What is my favorite color? (in session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

# Define a chain that limits the memory of messages to the last 2 messages
def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | model
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)

print("\n----------\n")

print("My favorite vehicles are Vespa scooters. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)

print("\n----------\n")

print("My favorite city is San Francisco. (in session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite color?")],
    },
    config=session1,
)

print("\n----------\n")

print("what is my favorite color? (chatbot with memory limited to the last 3 messages)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="what is my favorite color?")],
    config=session1,
)

print("\n----------\n")

print("what is my favorite color? (chatbot with unlimited memory)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")