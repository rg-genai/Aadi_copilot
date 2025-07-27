import os
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from data_transformer import transform_financial_data

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"
DATA_URL = f"http://127.0.0.1:8080/getmydata?sessionId={SESSION_ID}"

# --- DATA FETCHING ---
def get_financial_data():
    """Fetches financial data from the local server."""
    print("Connecting to server to fetch user's financial data...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        print("✅ Data fetched successfully. Initializing financial advisor...")
        return response.json()
    except requests.exceptions.RequestException:
        print(f"❌ Connection Error: Could not connect to the server at {DATA_URL}.")
        print("   Please ensure your Go server is running and you have logged in via the browser.")
        return None
    
def analyze_financial_data(data):
    """
    Takes the raw JSON from the server and creates a structured
    Financial Profile summary for the LLM.
    """
    try:
        profile = {}

        # Net Worth, Assets, Liabilities
        net_worth_data = data.get("fetch_net_worth", {}).get("netWorthResponse", {})
        profile["net_worth"] = int(net_worth_data.get("totalNetWorthValue", {}).get("units", 0))
        
        total_assets = 0
        total_liabilities = 0
        for asset in net_worth_data.get("assetValues", []):
            total_assets += int(asset.get("value", {}).get("units", 0))
        for liability in net_worth_data.get("liabilityValues", []):
             # Liabilities are often negative, take absolute value
            total_liabilities += abs(int(liability.get("value", {}).get("units", 0)))
            
        profile["total_assets"] = total_assets
        profile["total_liabilities"] = total_liabilities
        profile["debt_to_asset_ratio"] = round(total_liabilities / total_assets, 2) if total_assets > 0 else 0

        # Credit Score
        credit_report = data.get("fetch_credit_report", {}).get("creditReports", [{}])[0]
        profile["credit_score"] = int(credit_report.get("creditReportData", {}).get("score", {}).get("bureauScore", 0))

        # Investment Summary
        investments = []
        mf_analytics = data.get("fetch_net_worth", {}).get("mfSchemeAnalytics", {}).get("schemeAnalytics", [])
        for mf in mf_analytics:
            detail = mf.get("schemeDetail", {})
            analytics = mf.get("enrichedAnalytics", {}).get("analytics", {}).get("schemeDetails", {})
            investments.append({
                "name": detail.get("nameData", {}).get("longName"),
                "type": detail.get("assetClass"),
                "provider": detail.get("amc", "").replace("_MUTUAL_FUND", "").replace("_", " ").title(),
                "value": int(analytics.get("currentValue", {}).get("units", 0))
            })
        
        epf_balance = data.get("fetch_net_worth",{}).get("netWorthResponse",{}).get("assetValues",[])
        for asset in epf_balance:
            if asset.get("netWorthAttribute") == "ASSET_TYPE_EPF":
                investments.append({
                    "name": "Employee Provident Fund (EPF)",
                    "type": "DEBT",
                    "provider": "EPFO",
                    "value": int(asset.get("value",{}).get("units",0))
                })

        profile["investments"] = investments
        
        # Create a formatted string summary
        summary_text = f"""
--- USER FINANCIAL PROFILE ---
Net Worth: INR {profile['net_worth']:,}
Credit Score: {profile['credit_score']}

Assets & Liabilities:
- Total Assets: INR {profile['total_assets']:,}
- Total Liabilities: INR {profile['total_liabilities']:,}
- Debt-to-Asset Ratio: {profile['debt_to_asset_ratio']}

Investment Holdings Summary:
"""
        for inv in profile["investments"]:
            summary_text += f"- {inv['name']} ({inv['type']}): INR {inv['value']:,}\n"
        summary_text += "---------------------------\n"
        
        return summary_text

    except Exception as e:
        print(f"Error creating financial profile: {e}")
        return "Error: Could not summarize the user's financial data."
    
def get_detailed_answer(question, raw_data):
    """Creates a one-time prompt with the full raw data to answer a specific, detailed question."""
    print("AI is drilling down into the raw data for a detailed answer...")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # This prompt focuses only on answering the specific question using the full data
    prompt = f"""
    You are a data retrieval expert. Your task is to answer the user's specific question based ONLY on the full JSON data provided. Be precise and thorough.

    Full JSON Data:
    ```json
    {json.dumps(raw_data, indent=2)}
    ```

    User's Question: "{question}"

    Your Answer:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred while getting the detailed answer: {e}")
        return "Sorry, I encountered an error while retrieving the specific details."

# --- CORE CHATBOT LOGIC ---
def create_system_prompt(structured_data):
    """Creates the prompt using the new structured data."""

    # Now we can be much more precise with our prompt!
    holdings_summary = structured_data["holdings"][['asset_name', 'category', 'current_value']].to_string()
    total_portfolio_value = structured_data["holdings"]['current_value'].sum()

    return f"""
    You are "Aadi", a highly experienced AI financial co-pilot.

    Here is a summary of the user's financial holdings:
    - Total Portfolio Value: INR {total_portfolio_value:,.0f}

    Holdings Details:
    {holdings_summary}

    Your task is to act as a helpful and proactive financial advisor. Start by greeting the user and giving them a brief overview of their portfolio.
    """

def run_chat_session():
    """Initializes and runs the conversational loop."""

    raw_financial_data = get_financial_data()
    if not raw_financial_data:
        return

    structured_data = transform_financial_data(raw_financial_data)
    print("\n--- Transformed Holdings ---")
    print(structured_data["holdings"].head())
    print("\n--- Transformed Transactions ---")
    print(structured_data["transactions"].head())
    print("----------------------------\n")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    system_prompt = create_system_prompt(structured_data)

    chat = model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]}
    ])
    
    initial_response = chat.send_message("Let's begin.")
    print("\n--- Fi Financial Advisor ---")
    print(f"Aadi: {initial_response.text}")
    print("--------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Advisor: It was a pleasure assisting you. Have a great day!")
            break
            
        print("Aadi is thinking...")
        response = chat.send_message(user_input)
        
        # --- THIS IS THE CORRECTED LINE ---
        # We check if the keyword is 'in' the response, which is more robust.
        if "[get_details]" in response.text:
            detailed_answer = get_detailed_answer(user_input, raw_financial_data)
            print(f"Aadi: {detailed_answer}")
            # Add the detailed answer to the conversation history so the AI remembers it
            chat.history.append({'role': 'user', 'parts': [user_input]})
            chat.history.append({'role': 'model', 'parts': [detailed_answer]})
        else:
            print(f"Aadi: {response.text}")

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    run_chat_session()