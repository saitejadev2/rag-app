from generation.generator import Generator


generator = Generator()

context = """
FastAPI supports security mechanisms such as OAuth2
and API keys.
"""

query = "How does FastAPI handle authentication?"

answer = generator.generate(
    query=query,
    context=context
)

print(answer)