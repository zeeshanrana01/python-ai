
"""Create a business assistant with personas, trusted facts, and conversation memory."""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Demo business facts: replace these with your Module 1 business details.
facts = {
    "business_name": "Zeenexus AI",
    "description": "An AI and software solutions business",
    "currency": "PKR",
    "services": ["AI chatbot", "website development", "AI consultation"],
    "support_hours": "Monday to Friday, 9 AM to 5 PM",
    "prices": {
        "ai chatbot": 15000,
        "website development": 25000,
        "ai consultation": 5000,
    },
}

personas = {
    "friendly": "Warm and cheerful, uses simple words",
    "formal": "Professional, respectful, and concise",
    "sales": "Enthusiastic, highlights benefits, and suggests one relevant extra service",
}

# Why: Central configuration makes the model and memory limit easy to change.
config = {
    "model_name": "ollama:llama3.2",
    "temperature": 0.2,
    "memory_size": 8,
    "default_persona": "friendly",
}

llm = init_chat_model(
    config["model_name"],
    temperature=config["temperature"],
)

current_persona = config["default_persona"]


def build_system_message(persona_name):
    """Build the tutor instructions using verified business facts."""
    return f"""
You are a business assistant for {facts['business_name']}.

Business description: {facts['description']}
Services: {', '.join(facts['services'])}
Support hours: {facts['support_hours']}
Currency: {facts['currency']}
Prices: {facts['prices']}

Your personality: {personas[persona_name]}.

IMPORTANT RULES:
- Only use these facts to answer questions about the business.
- Never invent services, prices, policies, or business details.
- If information is missing, politely say you do not know.
- Keep answers concise.
- Use the selected personality without changing the facts.
"""


# Why: The system message stays at index 0 when memory is reset or trimmed.
messages = [SystemMessage(content=build_system_message(current_persona))]

# Why: A separate list lets us track user questions without counting AI replies.
questions = []

print("Bot ready. Commands: /persona friendly|formal|sales, /price <item>, /add <item> <price>, bye")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    command = user_input.lower()

    if command == "bye":
        print("Bot: Goodbye!")
        print(f"Questions asked: {len(questions)}")

        if questions:
            print(f"First question: {questions[0]}")
            print(f"Last question: {questions[-1]}")

        break

    if command.startswith("/persona"):
        parts = user_input.split(maxsplit=1)
        requested_persona = (
            parts[1].strip().lower() if len(parts) > 1 else ""
        )

        # .get() safely falls back if the requested persona is unknown.
        current_persona = (
            requested_persona
            if requested_persona in personas
            else config["default_persona"]
        )

        # Update the system instructions immediately.
        messages[0].content = build_system_message(current_persona)

        print(f"Bot: Style changed to: {personas[current_persona]}")
        continue

    if command.startswith("/price"):
        parts = user_input.split(maxsplit=1)

        if len(parts) < 2 or not parts[1].strip():
            print("Bot: Please provide an item. Example: /price AI chatbot")
            continue

        item = parts[1].strip().lower()
        price = facts["prices"].get(item)

        # Why: Looking up prices directly prevents the AI from inventing prices.
        if price is not None:
            print(f"Bot: {item.title()} costs {price:,.2f} {facts['currency']}.")
        else:
            print(f"Bot: Sorry, {item} is not on our menu.")

        continue

    if command.startswith("/add"):
        parts = user_input.split()

        if len(parts) < 3:
            print("Bot: Use /add <item> <price>. Example: /add logo design 3000")
            continue

        try:
            price = float(parts[-1])
            item = " ".join(parts[1:-1]).strip().lower()

            if not item or price < 0:
                raise ValueError

        except ValueError:
            print("Bot: Please provide a valid item and non-negative price.")
            continue

        facts["prices"][item] = price

        # Rebuild the system message so the AI receives the updated facts.
        messages[0].content = build_system_message(current_persona)

        print(f"Bot: Added {item.title()} at {price:,.2f} {facts['currency']}.")
        continue

    questions.append(user_input)
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.text))

    print(f"Bot: {response.text}")

    # Keep the system message plus the most recent conversation messages.
    messages = [messages[0]] + messages[1:][-config["memory_size"]:]
