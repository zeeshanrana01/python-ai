from langchain.chat_models import init_chat_model

# llm = init_chat_model("llama3.2", model_provider="ollama")

# response = llm.invoke([{"role": "user", "content": "Explain AI in 2 lines for a beginner."}])
# print(response.text)

# print(response.response_metadata["model_name"])  # prints the model name used for the response
# print(response.response_metadata["model_provider"])  # prints the model provider used for the response
# print(response.usage_metadata["input_tokens"])  # prints the number of input tokens used for the response
# print(response.usage_metadata["output_tokens"])  # prints the number of output tokens used for the response
# # print(response.usage_metadata["total_tokens"])  # prints the total number of tokens used for the response
# print(response.response_metadata["total_duration"])  # prints the time taken to generate the response

# seconds = response.response_metadata["total_duration"] / 1_000_000_000
# total_tokens = response.usage_metadata["total_tokens"]
# print(f"{total_tokens} tokens in {seconds:.1f} seconds")



# system= "You are the friendly assistant of Sweet Crumbs bakery in Lahore. Answer in 2 short lines."

# llm =  init_chat_model ("llama3.2" , model_provider="ollama", temperature = 0.3)
# print("Sweet Crumbs Bot: Ask me anything. Type 'bye' to exit.")

# while True:
#     message = input("you: ").strip()

#     if message.lower() == "bye":
#         print("Sweet Crumbs Bot: Goodbye! Have a sweet day!")
#         break
#     elif "refund" in message.lower():
#         print("bot:mangager will call you later")
#     else:
#         response = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": message}])
#         print(f"Bot: {response.text}")
#         print(f"     ({response.usage_metadata['output_tokens']} tokens)")


"""Make the first AI tutor call and display token usage and response time."""


from langchain.chat_models import init_chat_model


topic = input("Topic: ").strip()
level = input("Level (beginner or expert): ").strip()

# Why: The system message tells the AI exactly what role and response style to use.
system_message = f"""
You are a helpful tutor.

The student's level is: {level}

Explain the topic at the student's level.
If the level is beginner, use simple English and keep the answer short.
If the level is expert, give a more technical explanation.

Keep the answer concise and focused.
"""

# Why: A low temperature makes the tutor's answers more consistent and focused.
tutor = init_chat_model(
    "ollama:llama3.2",
    temperature=0.2,
)



response = tutor.invoke(
    [
        ("system", system_message),
        ("human", topic),
    ]
)



print("=" * 40)
print(response.text)
print("=" * 40)

print(response.response_metadata["model_name"])  # prints the model name used for the response
print(response.response_metadata["model_provider"])  # prints the model provider used for the response
print(response.usage_metadata["input_tokens"])  # prints the number of input tokens used for the response
print(response.usage_metadata["output_tokens"])  # prints the number of output tokens used for the response

print(response.response_metadata["total_duration"])  # prints the time taken to generate the response

seconds = response.response_metadata["total_duration"] / 1_000_000_000
total_tokens = response.usage_metadata["total_tokens"]
print(f"{total_tokens} tokens in {seconds:.1f} seconds")