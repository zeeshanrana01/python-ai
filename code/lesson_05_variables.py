"""
This program creates an AI model profile card
and estimates the cost of a chat based on token usage.
"""

# A model profile makes it easier to compare important AI model characteristics.
model_name = "llama3.2"
context_window = 128000
price_per_1k_tokens = 0.5
runs_locally = True

print("--- AI Model Profile Card ---")

print(f"Model: {model_name} {type(model_name)}")
print(f"Context window: {context_window} tokens {type(context_window)}")
print(f"Price per 1K tokens: {price_per_1k_tokens} PKR {type(price_per_1k_tokens)}")
print(f"Runs locally: {runs_locally} {type(runs_locally)}")

# Converting tokens to an integer allows the calculation to use the user's actual chat size.
tokens_per_chat = int(input("Tokens per chat: "))

estimated_cost = round(
    (tokens_per_chat / 1000) * price_per_1k_tokens,
    2
)

print(f"Estimated cost: {estimated_cost} PKR")

# Monthly usage helps the client understand the potential cost at a larger scale.
chats_per_day = int(input("Chats per day: "))

monthly_cost = round(
    estimated_cost * chats_per_day * 30,
    2
)

print(f"Estimated monthly cost: {monthly_cost} PKR")