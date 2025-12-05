import google.generativeai as genai
import os

try:
    # Load the API key directly from the file, like env_utils.py does
    with open("safe/api.key", "r") as f:
        api_key = f.read().strip()
        print("Loaded API key from safe/api.key")

    genai.configure(api_key=api_key)
    print("API key configured...")
    print("Attempting to connect to Gemini (60 second timeout)...")

    # Use a simple, reliable model for the test
    model = genai.GenerativeModel('gemini-pro')

    # Set an explicit timeout so it doesn't hang forever
    response = model.generate_content("What is 2+2?", request_options={"timeout": 60})

    print("\n--- SUCCESS! ---")
    print("API responded:")
    print(response.text)

except Exception as e:
    print("\n--- TEST FAILED ---")
    print(f"An error occurred: {e}")