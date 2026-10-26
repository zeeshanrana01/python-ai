
"""Route AI requests to different models based on task type and token length."""

task = input("Task (code / private / chat): ").strip().lower()
tokens = int(input("Estimated tokens: "))

# Why: Private data should stay local to improve privacy and avoid cloud costs.
if task == "private":
    model = "llama3.2 (local, Ollama)"
    price = 0

# Why: Code questions need a stronger model because they can require deeper reasoning.
elif task == "code":
    model = "claude (cloud, strong at code)"
    price = 0.80

else:
    model = "gemini (cloud, cheap)"
    price = 0.20


if tokens > 4000:
    print("Warning: long request, consider shortening it.")
else:
    print("Length is fine.")

cost = (tokens / 1000) * price

print(f"Chosen model: {model}")
print(f"Estimated cost: {cost:.2f} PKR")