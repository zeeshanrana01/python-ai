
"""
Lesson 2.8: First AI Agent

A tool-using Sweet Crumbs assistant that reads prices and FAQs,
checks order status, and places orders only after customer approval.
Conversation messages are kept in memory while the program is running.
"""

import json
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import ValidationError


# Why: Python is the trusted source for prices, not the language model.
PRICES = {
    "brownie": 300,
    "cupcake": 250,
    "donut": 200,
    "chocolate cake": 1500,
}

ORDER_STATUSES = {
    "A101": "being prepared",
    "A102": "out for delivery",
    "A103": "delivered",
}

NEXT_ORDER_NUMBER = 104
MAX_STEPS = 5

FAQ_FILE = Path(__file__).resolve().parent / "faq.json"


@tool(parse_docstring=True)
def get_price(item: str) -> str:
    """Look up the menu price for an item.

    Args:
        item: The menu item name, such as brownie or cupcake.
    """
    item = item.strip().casefold()

    if item not in PRICES:
        menu = ", ".join(sorted(PRICES))
        return f"'{item}' is not on the menu. Available items: {menu}."

    return f"{item} costs {PRICES[item]} PKR"


@tool(parse_docstring=True)
def order_status(order_id: str) -> str:
    """Check the current status of an order.

    Args:
        order_id: The order ID, such as A102.
    """
    order_id = order_id.strip().upper()
    status = ORDER_STATUSES.get(order_id)

    if status is None:
        return f"Order {order_id} was not found."

    return f"Order {order_id} is {status}"


@tool(parse_docstring=True)
def place_order(item: str, quantity: int, customer: str) -> str:
    """Place a new order only after the user approves it.

    Args:
        item: The menu item to order.
        quantity: The positive number of items requested.
        customer: The customer's name.
    """
    global NEXT_ORDER_NUMBER

    item = item.strip().casefold()
    customer = customer.strip()

    if item not in PRICES:
        return f"Order not placed: '{item}' is not on the menu."

    # Why: Validate tool arguments before asking for approval or changing data.
    if quantity <= 0:
        return "Order not placed: quantity must be greater than zero."

    if not customer:
        return "Order not placed: a customer name is required."

    total = PRICES[item] * quantity

    print(
        f"  [approve] {quantity} x {item} for {customer} "
        f"= {total} PKR. Place it? (yes/no) ",
        end="",
        flush=True,
    )
    approval = input().strip().casefold()

    if approval not in {"yes", "y"}:
        return "Order cancelled. No order was placed and no money was spent."

    order_id = f"A{NEXT_ORDER_NUMBER}"
    NEXT_ORDER_NUMBER += 1

    ORDER_STATUSES[order_id] = "confirmed"

    return (
        f"Order {order_id} placed: {quantity} x {item}, "
        f"total {total} PKR for {customer}"
    )


@tool(parse_docstring=True)
def get_faq(question: str) -> str:
    """Read the owner's FAQ file to help answer a customer question.

    Args:
        question: The customer's question about the business.
    """
    if not FAQ_FILE.exists():
        return (
            f"FAQ file not found at {FAQ_FILE}. "
            "Please create code/faq.json using the FAQ from Lesson 2.6."
        )

    try:
        with FAQ_FILE.open("r", encoding="utf-8") as file:
            faq_data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        return f"Could not read the FAQ file: {error}"

    # Why: The model should answer business-policy questions from the owner's
    # saved information rather than inventing opening hours or service rules.
    return (
        f"Customer question: {question}\n"
        f"Owner's FAQ data:\n{json.dumps(faq_data, ensure_ascii=False, indent=2)}"
    )


# The dictionary lets us safely find tools by the names requested by the model.
TOOLS = {
    get_price.name: get_price,
    order_status.name: order_status,
    place_order.name: place_order,
    get_faq.name: get_faq,
}

SYSTEM_PROMPT = """
You are the Sweet Crumbs business assistant.

Rules:
- Use get_price for menu price questions. Never invent prices.
- Use order_status when a customer asks about an existing order.
- Use get_faq for business policy, services, delivery, or opening-hour questions.
- Use place_order only when the customer clearly wants to place an order.
- Before placing an order, make sure you have the item, positive quantity,
  and customer name. Ask the user for missing details instead of guessing.
- place_order itself requires explicit yes/no approval. Never claim an order
  was placed unless the tool confirms it.
- If an order is cancelled, clearly say that it was not placed.
- Be concise, friendly, and accurate.
"""


def run_agent_turn(
    model_with_tools,
    messages: list,
) -> None:
    """Run one user turn, handling tool calls and the step limit."""
    for step in range(MAX_STEPS):
        try:
            response = model_with_tools.invoke(messages)
        except Exception as error:
            print(f"Agent: I couldn't contact the AI model ({type(error).__name__}).")
            return

        messages.append(response)

        # No tool calls means the model has produced its final reply.
        if not response.tool_calls:
            print(f"Agent: {response.content}")
            break

        for call in response.tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            call_id = call.get("id") or f"missing-id-{step}-{name}"

            if not isinstance(args, dict):
                result = "Tool error: arguments must be a JSON object."
            elif name not in TOOLS:
                result = f"Tool error: unknown tool '{name}'."
            else:
                try:
                    result = TOOLS[name].invoke(args)
                    if not isinstance(result, str):
                        result = str(result)
                except ValidationError as error:
                    result = f"Tool error: invalid arguments. {error}"
                except (TypeError, ValueError) as error:
                    result = f"Tool error: incorrect arguments. {error}"
                except Exception as error:
                    result = f"Tool error: {type(error).__name__}: {error}"

            # Every call, including a failed one, is visible for debugging.
            print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result}")

            # Why: ToolMessage must use the matching tool_call_id so the model
            # can connect each tool result to the call that requested it.
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id,
                )
            )

    else:
        # This runs only if all MAX_STEPS iterations finish without a break.
        print("Agent: I reached my step limit. Please try a simpler question.")


def main() -> None:
    """Start the interactive agent and preserve conversation history."""
    model = init_chat_model(
        "ollama:llama3.2",
        temperature=0,
    )

    # bind_tools gives the model the tool descriptions and argument schemas.
    model_with_tools = model.bind_tools(list(TOOLS.values()))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]

    print(
        "Sweet Crumbs Agent: ask about prices, orders, or place an order. "
        "Type 'bye' to exit."
    )

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Goodbye!")
            break

        if user_text.casefold() == "bye":
            print("Agent: Goodbye!")
            break

        if not user_text:
            continue

        # Keep earlier messages so follow-up questions have conversational memory.
        messages.append(HumanMessage(content=user_text))
        run_agent_turn(model_with_tools, messages)


if __name__ == "__main__":
    main()
