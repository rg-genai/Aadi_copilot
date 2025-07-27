import os
import json
import re
import requests
import pandas as pd
import yfinance as yf
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from scipy.optimize import fsolve

# --- NEW: PyPortfolioOpt Imports ---
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# --- LOCAL MODULES ---
# We import this for standalone testing of the agent
from data_transformer import transform_financial_data

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# --- KNOWLEDGE BASE CONFIGURATION ---
FAISS_INDEX_PATH = "tax_knowledge_base.index"
CHUNKS_DATA_PATH = "tax_chunks_and_metadata.json"
EMBEDDING_MODEL = 'all-mpnet-base-v2'


class PersonaAgent:
    """
    A specialist agent responsible for analyzing a user's financial data
    and classifying them into a detailed financial persona.
    """
    def __init__(self):
        print("🤖 Persona Agent initialized.")
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def _calculate_key_metrics(self, structured_data):
        """
        Calculates essential financial ratios from the structured data.
        """
        metrics = {}
        transactions_df = structured_data["transactions"]
        holdings_df = structured_data["holdings"]
        summary = structured_data["summary"]

        # 1. Income & Savings Rate (based on the last 2-3 months for accuracy)
        recent_transactions = transactions_df[transactions_df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
        monthly_income = recent_transactions[recent_transactions['description'].str.contains("SALARY", case=False)]['amount'].mean()
        
        # Calculate monthly investments (MF & Stock buys)
        monthly_investments = recent_transactions[
            (recent_transactions['type'] == 'BUY') & 
            (recent_transactions['source'].isin(['MUTUAL_FUND', 'STOCK']))
        ]['amount'].abs().sum() / 3 # Average over 3 months
        
        if monthly_income > 0:
            metrics['avg_monthly_income'] = f"{monthly_income:,.0f}"
            metrics['savings_rate_percent'] = f"{(monthly_investments / monthly_income) * 100:.2f}%" if monthly_income > 0 else "0%"
        
        # 2. Asset Allocation
        asset_allocation = holdings_df.groupby('asset_class')['current_value'].sum()
        total_assets = summary.get("Total Assets (INR)", 1) # Avoid division by zero
        metrics['asset_allocation'] = {
            "Equity_Percent": f"{(asset_allocation.get('EQUITY', 0) / total_assets) * 100:.2f}%",
            "Debt_Percent": f"{(asset_allocation.get('DEBT', 0) / total_assets) * 100:.2f}%",
            "Cash_Percent": f"{(asset_allocation.get('CASH', 0) / total_assets) * 100:.2f}%",
            "Commodity_Percent": f"{(asset_allocation.get('COMMODITY', 0) / total_assets) * 100:.2f}%",
        }

        # 3. Debt-to-Asset Ratio
        total_liabilities = summary.get("Total Liabilities (INR)", 0)
        metrics['debt_to_asset_ratio'] = f"{total_liabilities / total_assets:.2f}" if total_assets > 0 else "0"

        return metrics

    def run(self, structured_data):
        """
        Executes the persona analysis.
        """
        print("Persona Agent: Analyzing financial data to generate persona...")
        
        key_metrics = self._calculate_key_metrics(structured_data)
        
        prompt = f"""
        **TASK: Financial Persona Analysis**
        You are an expert financial analyst. Your task is to analyze the following key financial metrics and summary for a user and then classify them into a specific, descriptive "Financial Persona".

        **User's Financial Data:**
        - Key Metrics Summary: {json.dumps(structured_data['summary'], indent=2)}
        - Calculated Ratios: {json.dumps(key_metrics, indent=2)}
        - Holdings Overview: {structured_data['holdings'][['asset_name', 'category', 'current_value']].to_string()}

        **Instructions:**
        1.  **Analyze the Data:** Look at the net worth, credit score, debt-to-asset ratio, savings rate, and asset allocation.
        2.  **Choose a Persona Title:** Select a descriptive and insightful title for this user's financial character.
        3.  **Write a Summary:** In 2-3 sentences, provide a concise summary that justifies your chosen persona title, referencing specific data points.
        4.  **Format the Output:** Return the result ONLY as a valid JSON object with two keys: "title" and "summary".

        **BEGIN ANALYSIS AND PROVIDE THE JSON OUTPUT.**
        """
        
        try:
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            persona = json.loads(cleaned_response)
            print(f"✅ Persona generated successfully: {persona['title']}")
            return persona
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Persona Agent Error: Could not parse AI response. Error: {e}")
            return {"title": "Analysis Incomplete", "summary": "Could not generate a financial persona."}

class InvestmentAgent:
    """
    A specialist agent for analyzing the user's investment portfolio.
    This version performs deep analysis, calculates returns, and runs portfolio optimization.
    """
    def __init__(self):
        print("📈 Investment Agent initialized.")
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        # --- ENHANCED TICKER MAP ---
        self.ticker_map = {
            # Indian Stocks
            "RELIANCE INDUSTRIES LTD": "RELIANCE.NS", "TATA CONSULTANCY SERVICES LTD": "TCS.NS",
            "HDFC BANK LTD": "HDFCBANK.NS", "ICICI BANK LTD.": "ICICIBANK.NS",
            "BHARTI AIRTEL LIMITED": "BHARTIARTL.NS", "TATA MOTORS LTD.": "TATAMOTORS.NS",
            "ASIAN PAINTS LTD.": "ASIANPAINT.NS", "TITAN COMPANY LTD": "TITAN.NS",
            "BAJAJ FINANCE LIMITED": "BAJFINANCE.NS",
            # US Stocks
            "Apple Inc.": "AAPL", "Microsoft Corporation": "MSFT", 
            "NVIDIA Corporation": "NVDA", "Alphabet Inc.": "GOOGL",
            # Indian Mutual Funds (Using more reliable identifiers where possible)
            "Parag Parikh Flexi Cap Fund": "0P0000XWJ2.BO", # Still might be unreliable, yfinance has poor MF support
            "Parag Parikh Flexi Cap Fund - Direct - Growth": "0P0000XWJ2.BO",
            "ICICI Prudential Technology Fund": "0P0000732H.BO",
            "UTI Nifty 50 Index Fund": "0P0001D479.BO",
            "Nippon India Small Cap Fund - Direct - Growth": "0P0000XWW4.BO",
            "Axis Midcap Fund - Direct Plan - Growth": "0P0000O6N7.BO",
            "Mirae Asset Tax Saver Fund - Direct Plan - Growth": "0P00018544.BO",
            "Mirae Asset ELSS Tax Saver Fund - Direct Plan - Growth": "0P00018544.BO",
            "Tata Digital India Fund - Direct - Growth": "0P0001BBU4.BO",
        }
        self.isin_map = {
            "INE002A01018": "RELIANCE INDUSTRIES LTD", "INE467B01029": "TATA CONSULTANCY SERVICES LTD",
            "INE040A01034": "HDFC BANK LTD", "INE009A01021": "ICICI BANK LTD.",
            "INE075A01022": "BHARTI AIRTEL LIMITED", "INE155A01022": "TATA MOTORS LTD.",
            "INE021A01026": "ASIAN PAINTS LTD.", "INE280A01028": "TITAN COMPANY LTD",
            "INE296A01024": "BAJAJ FINANCE LIMITED"
        }

    def _get_live_prices(self, tickers):
        """Fetches live prices for a list of stock tickers."""
        if not tickers: return pd.Series(dtype=float)
        print(f"Investment Agent: Fetching live market data for {len(tickers)} tickers...")
        try:
            # Use yf.Tickers for more robust fetching
            data = yf.Tickers(tickers).history(period="2d", auto_adjust=True)
            if data.empty: return pd.Series(dtype=float)
            
            live_prices = data['Close'].iloc[-1]
            return live_prices
        except Exception as e:
            print(f"❌ Investment Agent Error (yfinance): {e}")
            return pd.Series(dtype=float)

    def _calculate_stock_returns(self, holdings_df, transactions_df, live_prices):
        """Calculates average buy price and returns for each stock holding."""
        stock_analysis = []
        stock_holdings = holdings_df[holdings_df['category'] == 'STOCKS_IND']
        
        for _, holding in stock_holdings.iterrows():
            stock_name = holding['asset_name']
            isin = next((k for k, v in self.isin_map.items() if v == stock_name), None)
            ticker = self.ticker_map.get(stock_name)

            if not ticker or ticker not in live_prices.index or pd.isna(live_prices[ticker]):
                stock_analysis.append(f"- **{stock_name}:** Live price data is currently unavailable for analysis.")
                continue

            # Ensure we have an ISIN to find transactions
            if not isin:
                 stock_analysis.append(f"- **{stock_name}:** Holdings found, but cannot match to transactions without an ISIN.")
                 continue

            buy_txns = transactions_df[
                (transactions_df['description'].str.contains(isin, na=False)) &
                (transactions_df['type'] == 'BUY')
            ]

            if not buy_txns.empty:
                total_cost = buy_txns['amount'].abs().sum()
                # Extracting quantity can be fragile, ensure it's robust
                buy_txns['quantity'] = buy_txns['description'].str.extract(r'\((\d+\.?\d*)\s*units\)', expand=False).astype(float)
                total_quantity = buy_txns['quantity'].sum()
                
                if total_quantity > 0:
                    avg_buy_price = total_cost / total_quantity
                    live_price = live_prices[ticker]
                    gain_loss_percent = ((live_price - avg_buy_price) / avg_buy_price) * 100
                    
                    stock_analysis.append(
                        f"- **{stock_name}:** Currently trading at INR {live_price:,.2f}. "
                        f"Your average buy price was approx. INR {avg_buy_price:,.2f}, "
                        f"representing a real-time gain/loss of **{gain_loss_percent:+.2f}%**."
                    )
                else:
                    stock_analysis.append(f"- **{stock_name}:** Holdings found, but could not calculate quantity from transactions.")
            else:
                stock_analysis.append(f"- **{stock_name}:** No purchase history found in transactions to calculate returns.")

        return "\n".join(stock_analysis) if stock_analysis else "No stock holdings with trackable transaction history found."

    def _optimize_portfolio(self, holdings_df):
        """
        Performs portfolio optimization using PyPortfolioOpt with diversification constraints.
        """
        print("Investment Agent: Running portfolio optimization...")
        optimizable_assets = holdings_df[holdings_df['category'].isin(['STOCKS_IND', 'MUTUAL_FUNDS'])]

        # Map asset names to tickers, filtering out those not in the map
        assets_with_tickers = optimizable_assets.copy()
        assets_with_tickers['ticker'] = assets_with_tickers['asset_name'].map(self.ticker_map)
        assets_with_tickers.dropna(subset=['ticker'], inplace=True)
        tickers = assets_with_tickers['ticker'].tolist()

        if len(tickers) < 2:
            return {"status": "skipped", "message": "Portfolio optimization requires at least two assets with trackable tickers."}

        try:
            prices = yf.download(tickers, period="2y", auto_adjust=True)['Close']
            # Drop columns (tickers) where all values are NaN (failed download)
            prices.dropna(axis=1, how='all', inplace=True)
            prices.ffill(inplace=True) # Forward-fill missing values
            
            if prices.shape[1] < 2:
                 return {"status": "skipped", "message": "Could not fetch sufficient historical data for at least two assets to perform optimization."}
                 
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)
            
            ef = EfficientFrontier(mu, S)
            
            # --- NEW: Add Diversification Constraint ---
            # No single asset can have more than 25% of the portfolio weight
            ef.add_constraint(lambda w: w <= 0.25)
            
            ef.max_sharpe() # Optimize for the best risk-adjusted return
            cleaned_weights = ef.clean_weights()
            opt_return, opt_vol, opt_sharpe = ef.portfolio_performance(verbose=False)

            readable_weights = {
                assets_with_tickers[assets_with_tickers['ticker'] == ticker]['asset_name'].iloc[0]: f"{weight:.2%}"
                for ticker, weight in cleaned_weights.items() if weight > 0
            }

            return {
                "status": "success",
                "optimized_recommendation": {
                    "suggested_allocation": readable_weights,
                    "expected_annual_return": f"{opt_return:.2%}",
                    "annual_volatility": f"{opt_vol:.2%}",
                    "sharpe_ratio": f"{opt_sharpe:.2f}"
                }
            }
        except Exception as e:
            print(f"❌ Investment Agent Error (Optimization): {e}")
            return {"status": "error", "message": f"An error occurred during portfolio optimization: {e}"}

    def _generate_optimization_summary(self, optimization_data, persona):
        """
        Generates a user-friendly summary of the optimization results using an LLM.
        """
        print("Investment Agent: Generating user-friendly summary...")
        
        # Handle cases where optimization did not succeed
        if optimization_data.get("status") != "success":
            return {
                "title": "Portfolio Check-Up",
                "summary": "I wasn't able to run an automated optimization on your portfolio at this time. This is usually due to missing historical data for some of your specific funds. This is a technical issue and doesn't reflect on the quality of your investments. I can still provide a general analysis of your holdings.",
                "plan_of_action": ["Review your individual fund performances and continue your regular investment plan."]
            }

        prompt = f"""
        **TASK: Financial Advisor Summary**

        You are Aadi, an expert financial advisor. Your client is a '{persona.get('title', 'client')}' and you need to explain their portfolio optimization results in a simple, clear, and encouraging way.

        **Analysis Data (JSON):**
        {json.dumps(optimization_data, indent=2)}

        **CRITICAL INSTRUCTIONS:**
        1.  **Simplify, Don't Dumb Down:** Explain the concepts simply. For example:
            - 'Annual Volatility' is 'risk' or 'how much the portfolio's value tends to swing up and down.'
            - 'Expected Annual Return' is 'the potential growth you might expect over a year.'
        2.  **Create a Narrative:** Don't just list the numbers. Explain what they mean for the client.
        3.  **Provide a Clear Plan:** Create a bulleted "Plan of Action". The goal is gradual improvement, not a drastic overhaul. Acknowledge that the 'optimized' portfolio is a mathematical suggestion and should be approached thoughtfully.
        4.  **Format the Output:** Return ONLY a valid JSON object with three keys: "title", "summary", and "plan_of_action" (which should be a list of strings).

        **EXAMPLE TONE:** "Your portfolio is on a solid track. My analysis suggests a few tweaks could potentially increase your returns without taking on much more risk. Let's look at the details..."

        **BEGIN YOUR EXPERT SUMMARY (JSON ONLY).**
        """

        try:
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            summary_json = json.loads(cleaned_response)
            return summary_json
        except Exception as e:
            print(f"❌ Investment Agent Error (Summary Generation): {e}")
            return {"title": "Analysis Summary Unavailable", "summary": "Could not generate a summary for the optimization results.", "plan_of_action": []}

    def run(self, structured_data, query, persona):
        """Executes a deep investment analysis and returns a structured dictionary."""
        print(f"Investment Agent: Running deep analysis for query: '{query}'")
        holdings_df = structured_data["holdings"]
        transactions_df = structured_data["transactions"]
        
        all_tickers = [t for t in self.ticker_map.values() if t]
        live_prices = self._get_live_prices(list(set(all_tickers)))
        
        asset_allocation = holdings_df.groupby('asset_class')['current_value'].sum()
        total_portfolio_value = holdings_df['current_value'].sum()
        stock_return_analysis = self._calculate_stock_returns(holdings_df, transactions_df, live_prices)
        
        # Run Portfolio Optimization
        portfolio_optimization_result = self._optimize_portfolio(holdings_df)
        
        # --- NEW: Generate a user-friendly summary ---
        optimization_summary = self._generate_optimization_summary(portfolio_optimization_result, persona)

        return {
            "total_value": total_portfolio_value,
            "asset_allocation": asset_allocation.to_dict(),
            "stock_analysis": stock_return_analysis,
            "mf_xirr": holdings_df[holdings_df['xirr'].notna()][['asset_name', 'xirr']].to_dict('records'),
            "portfolio_optimization_raw": portfolio_optimization_result, # Keep raw data for debugging
            "portfolio_optimization_summary": optimization_summary # The new user-friendly output
        }

# The TaxAgent and GoalSimulationAgent classes remain unchanged...

class TaxAgent:
    """
    A specialist agent that uses a RAG pipeline to answer tax-related questions.
    """
    def __init__(self):
        print("🧾 Tax Agent initialized.")
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        try:
            print("Tax Agent: Loading knowledge base...")
            self.retriever = SentenceTransformer(EMBEDDING_MODEL)
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_DATA_PATH, 'r') as f:
                self.chunks_with_metadata = json.load(f)
            print("✅ Knowledge base loaded successfully.")
        except Exception as e:
            print(f"❌ Tax Agent Error: Could not load knowledge base. Have you run 'build_knowledge_base.py'?")
            self.retriever = None

    def _search_knowledge_base(self, query, k=3):
        """Searches the vector database for the most relevant context."""
        if not self.retriever: return []
        
        query_embedding = self.retriever.encode([query], convert_to_numpy=True)
        _, ids = self.index.search(query_embedding, k)
        
        return [self.chunks_with_metadata[chunk_id]['content'] for chunk_id in ids[0] if chunk_id != -1]

    def run(self, structured_data, query):
        """Executes the RAG pipeline to answer a tax question with authority."""
        print(f"Tax Agent: Running RAG pipeline for query: '{query}'")
        if not self.retriever:
            return "My tax knowledge base is currently unavailable. I cannot answer tax-related questions at this time."

        retrieved_context = self._search_knowledge_base(query)
        if not retrieved_context:
            print("Tax Agent: No specific context found. Using general knowledge.")
            context_str = "No specific context was found in the knowledge base."
        else:
            context_str = "\n\n---\n\n".join(retrieved_context)

        prompt = f"""
        **TASK: Elite Tax Consultant**

        You are Aadi, an expert Indian tax consultant with 20 years of experience. Your tone is confident, authoritative, and direct. You NEVER mention that you are an AI or that you are referencing a "text" or "document". You provide answers as if they are from your own expert knowledge.

        **Source of Truth (Internal Knowledge):**
        ---
        {context_str}
        ---

        **User's Financial Data:**
        - Holdings: {structured_data['holdings'][['asset_name', 'category', 'invested_value']].to_string()}
        - Summary: {json.dumps(structured_data['summary'], indent=2)}

        **User's Question:**
        "{query}"

        **CRITICAL INSTRUCTIONS:**
        1.  **Act as an Expert:** Synthesize an answer using the "Source of Truth" as your own knowledge. Do not say "the text says" or "according to the document".
        2.  **Perform Calculations:** If the user asks for a tax calculation (e.g., "how much tax will I pay?"), you MUST perform the step-by-step calculation based on the tax slabs in your knowledge. Show your work clearly.
        3.  **Be Decisive:** Provide a direct, conclusive answer. If the knowledge base is insufficient, state your conclusion based on general tax principles but note that specific conditions may apply. Do not be evasive.
        
        **BEGIN YOUR EXPERT RESPONSE.**
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Tax Agent Error: {e}")
            return "I encountered an error while analyzing your tax question. Please try again."

class GoalSimulationAgent:
    """
    A specialist agent for running financial simulations and forecasting goal achievement.
    """
    def __init__(self):
        print("🎯 Goal Simulation Agent initialized.")
        self.return_assumptions = {
            "EQUITY": 0.12, "DEBT": 0.07, "CASH": 0.03,
            "COMMODITY": 0.05, "HYBRID": 0.10, "RETIREMENT": 0.08
        }
        self.inflation_rate = 0.06
        self.n_simulations = 10000

    def _parse_goal_from_query(self, query):
        """Intelligently extracts the target amount and timeframe from the user's query."""
        target_goal, timeframe_years = None, None
        
        # Look for amounts (crore, lakh)
        crore_match = re.search(r'([\d\.]+)\s*cr', query, re.IGNORECASE)
        lakh_match = re.search(r'([\d\.]+)\s*lakh', query, re.IGNORECASE)
        
        if crore_match:
            target_goal = float(crore_match.group(1)) * 1_00_00_000
        elif lakh_match:
            target_goal = float(lakh_match.group(1)) * 1_00_000
        
        # Look for timeframes (years)
        year_match = re.search(r'(\d+)\s*year', query, re.IGNORECASE)
        if year_match:
            timeframe_years = int(year_match.group(1))
            
        # Fallback for queries like "reach 10000000 in 10"
        if not target_goal or not timeframe_years:
            numbers = [float(num.replace(',', '')) for num in re.findall(r'[\d,]+', query)]
            if len(numbers) >= 2:
                if not target_goal: target_goal = max(numbers)
                if not timeframe_years: timeframe_years = min(num for num in numbers if num < 100) # Assume years < 100

        return target_goal, timeframe_years

    def _calculate_inputs(self, structured_data):
        """Calculates the necessary inputs for the simulation from user data."""
        summary = structured_data["summary"]
        holdings_df = structured_data["holdings"]
        transactions_df = structured_data["transactions"]
        
        current_net_worth = summary.get("Net Worth (INR)", 0)

        asset_allocation = holdings_df.groupby('asset_class')['current_value'].sum()
        total_invested = asset_allocation.sum()
        weighted_return = 0
        if total_invested > 0:
            for asset_class, value in asset_allocation.items():
                weighted_return += (value / total_invested) * self.return_assumptions.get(asset_class, 0.0)

        recent_txns = transactions_df[transactions_df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
        monthly_investments = recent_txns[
            (recent_txns['type'] == 'BUY') & 
            (recent_txns['source'].isin(['MUTUAL_FUND', 'STOCK']))
        ]['amount'].abs().sum() / 3
        
        return current_net_worth, weighted_return, monthly_investments

    def _run_monte_carlo(self, pv, expected_return, monthly_investment, years, target):
        """Runs the Monte Carlo simulation."""
        annual_investment = monthly_investment * 12
        volatility = 0.15 
        
        final_values = []
        for _ in range(self.n_simulations):
            future_value = pv
            for _ in range(int(years)):
                random_return = np.random.normal(expected_return, volatility)
                future_value = (future_value + annual_investment) * (1 + random_return)
            final_values.append(future_value)
        
        successful_simulations = [val for val in final_values if val >= target]
        probability = len(successful_simulations) / self.n_simulations
        
        return probability, np.mean(final_values), np.percentile(final_values, 5), np.percentile(final_values, 95)

    def _calculate_path_to_100(self, pv, expected_return, monthly_investment, years, target):
        """Calculates the required adjustments to reach the goal with high certainty."""
        annual_investment = monthly_investment * 12
        target_fv = target
        
        def pmt_equation(pmt):
            fv = pv
            for _ in range(int(years)):
                conservative_return = expected_return - 1.645 * 0.15 
                fv = (fv + pmt * 12) * (1 + conservative_return)
            return fv - target_fv
        
        required_pmt = fsolve(pmt_equation, monthly_investment)[0]
        
        def rate_equation(rate):
            fv = pv
            for _ in range(int(years)):
                fv = (fv + annual_investment) * (1 + rate)
            return fv - target_fv
            
        required_rate = fsolve(rate_equation, expected_return)[0]

        return {
            "additional_monthly_investment_needed": f"{max(0, required_pmt - monthly_investment):,.0f}",
            "required_annual_return_percent": f"{required_rate * 100:.2f}%"
        }

    def run(self, structured_data, query):
        """Executes the goal simulation and provides actionable recommendations."""
        print(f"Goal Simulation Agent: Running simulation for query: '{query}'")
        
        target_goal, timeframe_years = self._parse_goal_from_query(query)
        if not target_goal or not timeframe_years:
            return {"error": "Could not determine the financial goal and timeframe from your query. Please be more specific, for example: 'Can I reach 1 Crore in 10 years?'"}

        pv, expected_return, monthly_investment = self._calculate_inputs(structured_data)
        
        probability, avg_fv, worst_fv, best_fv = self._run_monte_carlo(pv, expected_return, monthly_investment, timeframe_years, target_goal)
        
        path_to_100 = None
        if probability < 0.95:
            path_to_100 = self._calculate_path_to_100(pv, expected_return, monthly_investment, timeframe_years, target_goal)

        return {
            "goal_type": "Net Worth Target", "target_goal": f"INR {target_goal:,.0f}",
            "timeframe_years": int(timeframe_years), "inputs": {
                "starting_net_worth": f"INR {pv:,.0f}", "avg_monthly_investment": f"INR {monthly_investment:,.0f}",
                "estimated_annual_return": f"{expected_return:.2%}" },
            "simulation_results": { "n_simulations_run": self.n_simulations,
                "probability_of_success": f"{probability * 100:.2f}%",
                "average_projected_net_worth": f"INR {avg_fv:,.0f} (The average outcome across all simulations)",
                "worst_case_net_worth": f"INR {worst_fv:,.0f} (A 5% chance of ending up with this or less)",
                "best_case_net_worth": f"INR {best_fv:,.0f} (A 5% chance of ending up with this or more)"},
            "path_to_100_percent": path_to_100, "summary": (
                f"Based on {self.n_simulations:,} simulations, you have a {probability*100:.2f}% chance of reaching your goal. "
                f"The average projected outcome is a net worth of INR {avg_fv:,.0f}. "
                + ("However, there is a chance of falling short in poor market conditions." if probability < 0.95 else "You are well on track to achieve this goal."))}

# --- Standalone Testing Block ---
def get_financial_data_for_test(session_id):
    """Helper function to fetch data for testing this module directly."""
    data_url = f"http://127.0.0.1:8080/getmydata?sessionId={session_id}"
    print(f"Connecting to MCP server for session: {session_id}...")
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Connection Error: {e}"); return None

if __name__ == "__main__":
    print("--- Running Agents in Standalone Test Mode ---")
    TEST_SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"
    
    raw_data = get_financial_data_for_test(TEST_SESSION_ID)
    if raw_data:
        structured_data = transform_financial_data(raw_data)
        if structured_data:
            # --- Test Persona Agent First ---
            persona_agent = PersonaAgent()
            persona = persona_agent.run(structured_data)
            
            # --- Test Investment Agent with NEW Enhancements ---
            investment_agent = InvestmentAgent()
            inv_query = "How is my portfolio performing and can it be improved?"
            # Pass the persona to the investment agent
            investment_analysis = investment_agent.run(structured_data, query=inv_query, persona=persona)
            
            print("\n" + "="*50)
            print("--- ✅ INVESTMENT AGENT (with Enhancements) TEST RESULT ---")
            print(f"QUERY: {inv_query}\n")
            # Pretty print the detailed dictionary output
            print(json.dumps(investment_analysis, indent=2))
            print("="*50)