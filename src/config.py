import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")