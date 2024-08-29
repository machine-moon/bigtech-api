import google.generativeai as genai
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
    api_key = config["api_key"]

genai.configure(api_key=api_key)


model = genai.GenerativeModel("gemini-pro")



def start():
    user_input = ''
    while user_input != 'exit':
        user_input = input('User: ')
        if user_input != 'exit':
            print('Gemini:\n')
            response = model.generate_content(user_input)
            print(response.text)
        else:
            print('Goodbye!')
            

# Option 1: Using environment variable
# genai.configure() will automatically detect it

# Option 2: Using configuration file
# genai.configure(api_key=api_key)
############################################################################################################################################

start()
    







       
