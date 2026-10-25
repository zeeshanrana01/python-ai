"""Clean customer questions and build a safe AI prompt."""

name = input("Your name: ")
question = input("Your question: ")

# Why: strip removes accidental spaces that customers may type.
clean_name = name.strip().title()

# Why: capitalize gives the question a consistent format before sending it to AI.
clean_question = question.strip().capitalize()

characters = len(clean_question)
estimated_tokens = round(characters / 4)

mentions_python = "python" in clean_question.lower()

prompt = f"""You are a friendly Python tutor.
Greet {clean_name} by name and answer in 3 short lines.
Question: {clean_question[:150]}
"""

print("=" * 40)
print(f"Cleaned question: {clean_question}")
print(f"Characters: {characters}, about {estimated_tokens} tokens")
print(f"Mentions Python: {mentions_python}")
print("=" * 40)
print(prompt)