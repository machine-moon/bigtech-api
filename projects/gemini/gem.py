import google.generativeai as genai
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
    api_key = config["api_key"]

genai.configure(api_key=api_key)


model = genai.GenerativeModel("gemini-pro")
# response = model.generate_content('tell me a joke')
# print(response.text)

# Option 1: Using environment variable
# genai.configure() will automatically detect it

# Option 2: Using configuration file
# genai.configure(api_key=api_key)
############################################################################################################################################


def is_prime(n):
    """
    Check if a number is prime.

    Args:
      n: The number to check.

    Returns:
      True if n is prime, False otherwise.
    """

    if n <= 1:
        return False

    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False

    return True


def generate_primes():
    """
    Generate prime numbers forever.

    Yields:
      Prime numbers.
    """

    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1


if __name__ == "__main__":
    # Print the first 100 prime numbers.

    for prime in generate_primes():
        print(prime)
       
