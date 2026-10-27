

"""Build a simple AI tutor with conversation memory and limited context."""

from langchain.chat_models import init_chat_model


# Why: The system message stays at index 0 so the tutor always keeps its instructions.
system_message = """
You are a helpful tutor.

Answer clearly and concisely.
Keep answers short and easy to understand.
"""

# Why: A list lets us keep both user messages and AI replies in conversation order.
messages = [
    ("system", system_message),
]

# Why: Keeping only recent messages prevents the prompt from growing forever.
questions = []

llm = init_chat_model(
    "ollama:llama3.2",
    temperature=0.2,
)

print("Tutor: Ask me anything. Commands: /reset, /count, bye")

while True:
    user_message = input("You: ").strip()

    if user_message.lower() == "bye":
        print("Tutor: Goodbye!")

        if questions:
            print(f"You asked {len(questions)} questions.")
            print(f"First question: {questions[0]}")
            print(f"Last question: {questions[-1]}")

        break

    if user_message.lower() == "/reset":
        messages = [messages[0]]
        print("Tutor: Memory cleared.")
        continue

    if user_message.lower() == "/count":
        print(f"Tutor: {len(messages)} messages in memory.")
        continue

    questions.append(user_message)

    # Add the user's message to the conversation.
    messages.append(("human", user_message))

    response = llm.invoke(messages)

    # Add the AI response so the tutor can remember its own previous answers.
    messages.append(("assistant", response.text))

    print(f"Tutor: {response.text}")
    print(
        f"       (tokens sent: "
        f"{response.usage_metadata['input_tokens']})"
    )

    # Keep the system message + only the last 8 conversation messages.
    messages = [messages[0]] + messages[1:][-8:]
