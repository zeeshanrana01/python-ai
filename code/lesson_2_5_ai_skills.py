
"""
Lesson 2.5: AI Skills Toolkit

Reusable AI skills for summarizing text, translating content,
and replying professionally to bakery customer reviews.
"""

from langchain.chat_models import init_chat_model


# Why: Create both models once so every skill can reuse them.
precise_model = init_chat_model(
    "ollama:llama3.2",
    temperature=0.1,
)

creative_model = init_chat_model(
    "ollama:llama3.2",
    temperature=0.8,
)


def ask_ai(
    prompt: str,
    system: str = "You are a helpful assistant. Follow the user's instructions.",
    creative_mode: bool = False,
) -> str:
    """Send a prompt to the selected model and return its text response."""
    # Why: Choose a model based on the task instead of creating one per request.
    model = creative_model if creative_mode else precise_model
    response = model.invoke(
        [
            ("system", system),
            ("human", prompt),
        ]
    )
    return response.text


def summarize(text: str) -> str:
    """Summarize bakery-related text into a few clear sentences."""
    return ask_ai(
        f"Summarize this text in 2 short sentences:\n{text}",
        system="You summarize text accurately. Do not add unsupported facts.",
    )


def translate(text: str, language: str = "Urdu") -> str:
    """Translate text into the requested language."""
    return ask_ai(
        f"Translate the following text into {language}. Return only the translation:\n{text}",
        system="You are an accurate translator. Preserve the original meaning.",
    )


def reply_to_review(review: str) -> str:
    """Write a polite and professional reply to a bakery customer review."""
    return ask_ai(
        f"Write a short, friendly reply to this customer review:\n{review}",
        system=(
            "You represent Sweet Crumbs bakery. Be polite and professional. "
            "Thank customers for positive feedback. Acknowledge complaints "
            "without making promises that are not provided."
        ),
        creative_mode=True,
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple word-based approximation."""
    # Why: This utility works locally without spending tokens on an AI call.
    return len(text.split())


# Why: Assertions check the utility function without needing an AI connection.
assert estimate_tokens("") == 0
assert estimate_tokens("Cake is delicious") == 3
assert estimate_tokens("Your order is ready") == 4


def main() -> None:
    """Run the interactive Sweet Crumbs AI skills menu."""
    usage = {
        "summarize": 0,
        "translate": 0,
        "reply": 0,
        "batch": 0,
    }

    print("Sweet Crumbs AI Toolkit. Commands: /summarize, /translate, /reply, /batch, bye")

    while True:
        command = input("\nCommand: ").strip().lower()

        if command == "bye":
            break

        elif command == "/summarize":
            text = input("Text to summarize: ").strip()
            if not text:
                print("Please enter some text.")
                continue

            print("AI:", summarize(text))
            usage["summarize"] += 1

        elif command == "/translate":
            text = input("Text to translate: ").strip()
            if not text:
                print("Please enter some text.")
                continue

            language = input("Language (Enter for Urdu): ").strip()
            if not language:
                language = "Urdu"

            print("AI:", translate(text, language))
            usage["translate"] += 1

        elif command == "/reply":
            review = input("Customer review: ").strip()
            if not review:
                print("Please enter a review.")
                continue

            print("AI:", reply_to_review(review))
            print(f"    (review was about {estimate_tokens(review)} tokens)")
            usage["reply"] += 1

        elif command == "/batch":
            reviews = [
                "The chocolate cake was delicious!",
                "My cupcakes were great, but delivery was late.",
                "Good brownies and friendly service.",
            ]

            # Why: enumerate gives each generated batch result a clear number.
            for number, review in enumerate(reviews, start=1):
                print(f"\n{number}. Review: {review}")
                print("   AI:", reply_to_review(review))

            usage["batch"] += 1

        else:
            print("Unknown command. Try /summarize, /translate, /reply, /batch or bye.")

    print("\nSkills used today:")
    for skill, count in usage.items():
        print(f"  {skill}: {count}")


if __name__ == "__main__":
    main()
