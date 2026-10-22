"""
This program collects business information and creates
an AI prompt for suggesting practical business solutions.
"""

# Collecting specific business details makes the AI suggestions more relevant.
name = input("Enter your name: ")
business = input("Enter your business type: ")
problem = input("Enter one problem your business faces: ")

# A budget helps the AI suggest solutions that are realistic for the business.
budget = int(input("Enter your monthly budget in PKR: "))

print(f"Welcome, {name}! Let's create an AI prompt for your business.")

prompt = (
    f"Suggest 3 AI solutions for a {business} that struggles with {problem}. "
    f"Budget: {budget} PKR per month."
)

print("\nYour AI prompt:")
print(prompt)