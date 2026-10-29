
"""
Lesson 2.4: Batch Writer

Generate one-sentence AI descriptions for business products,
skip products with zero prices, and report token usage and time.
"""

from langchain.chat_models import init_chat_model

# Business products and their prices in PKR.
products = {
    "Chocolate Cake": 1500,
    "Cupcake": 250,
    "Brownie": 300,
    "Donut": 200,
    "Cookies": 180,
    "Old Stock": 0,
}

# Why: Keep the model configuration in one place so it is easy to change.
config = {
    "model_name": "ollama:llama3.2",
    "temperature": 0.7,
}

# Why: Create the model once, not repeatedly inside the loop.
llm = init_chat_model(
    config["model_name"],
    temperature=config["temperature"],
)

# Why: Accumulators must start at zero before processing any products.
total_tokens = 0
total_seconds = 0.0

results = []

for name, price in products.items():
    if price == 0:
        print(f"Skipping {name.lower()} (no price)")
        continue

    prompt = f"""
Write exactly one appealing sentence describing this product:
Product: {name}
Price: {price} PKR

Do not invent ingredients or unsupported business claims.
Return only the description.
"""

    response = llm.invoke(prompt)

    # Extract the generated description.
    description = response.content
    results.append({
        "item": name,
        "price": price,
        "text": description,
    })

    # Collect token usage when the model provides it.
    usage = response.usage_metadata or {}
    tokens = usage.get("total_tokens", 0)
    total_tokens += tokens

    # Ollama may report generation time in nanoseconds.
    metadata = response.response_metadata or {}
    ollama_seconds = metadata.get("total_duration", 0) / 1_000_000_000
    total_seconds += ollama_seconds

    # Why: Skipping zero-price products prevents unnecessary AI calls.
    # The continue above also ensures they cost no generation tokens.

print("=" * 50)

for number, item in enumerate(results, start=1):
    print(f"{number}. {item['item']} ({item['price']} PKR)")
    print(f"   {item['text']}")

print("=" * 50)
print(
    f"{len(results)} descriptions, "
    f"{total_tokens} tokens, "
    f"{total_seconds:.1f} seconds"
)
