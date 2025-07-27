import os
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load variables from your .env file
load_dotenv() 

# 1. Load the Gemini API Key from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# 2. Configure your local server details
SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"
DATA_URL = f"http://127.0.0.1:8080/getmydata?sessionId={SESSION_ID}"


def get_financial_data():
    """Fetches the financial data from your local server."""
    print("Fetching financial data from local server...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        print("Data fetched successfully!")
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        if err.response.status_code == 401:
            print("Error 401: Unauthorized. Have you logged in with this session ID in your browser first?")
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: Could not connect to the server at {DATA_URL}.")
        print("Please ensure your Go server is running.")
    return None

def generate_insights(financial_data):
    """Sends the data to Gemini and returns the insights."""
    print("Generating insights with Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are a helpful and friendly financial assistant. Your goal is to analyze a user's financial data and provide clear, actionable insights. Do not be overly cautious or generic.

    Analyze the following financial data which is in JSON format:
    ```json
    {json.dumps(financial_data, indent=2)}
    ```

    Based on this data, please provide:
    1.  A one-sentence summary of the user's overall financial health.
    2.  Three clear, bulleted insights about their assets, liabilities, or investment patterns.
    3.  One actionable recommendation for them to consider next.
    
    Present the output in a clean, readable format.
    """
    
    try:
        response = model.generate_content(prompt)
        print("Insights generated!")
        return response.text
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return None

if __name__ == "__main__":
    data = get_financial_data()
    if data:
        insights = generate_insights(data)
        if insights:
            print("\n💡 Your Financial Insights from Gemini 💡")
            print("------------------------------------------")
            print(insights)