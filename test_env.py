from dotenv import load_dotenv 
import os 
load_dotenv() 
key = os.getenv('GOOGLE_API_KEY') 
print('Key loaded:', key is not None)