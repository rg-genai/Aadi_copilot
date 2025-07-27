import os
import json
import requests
import asyncio
import traceback
import pandas as pd
import numpy as np
import datetime
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# --- LOCAL MODULES ---
from data_transformer import transform_financial_data
from agents2 import PersonaAgent, InvestmentAgent, TaxAgent, GoalSimulationAgent

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# --- API SETUP ---
app = FastAPI(
    title="Project Aadi - Financial Co-Pilot API",
    description="The backend server for the Project Aadi multi-agent system with an intelligent orchestrator.",
    version="2.0.0"
)

# --- CORS MIDDLEWARE ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AGENT INITIALIZATION ---
try:
    persona_agent = PersonaAgent()
    investment_agent = InvestmentAgent()
    tax_agent = TaxAgent()
    goal_simulation_agent = GoalSimulationAgent()
    orchestrator_model = genai.GenerativeModel('gemini-2.5-pro')
except Exception as e:
    print(f"FATAL: Could not initialize agents. Error: {e}")
    orchestrator_model = None


# --- DATA MODELS ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
    status: str

# --- HELPER FUNCTIONS ---
def get_financial_data(session_id: str):
    """Fetches financial data from the local Go server."""
    data_url = f"http://127.0.0.1:8080/getmydata?sessionId={session_id}"
    print(f"Orchestrator: Connecting to MCP server for session: {session_id}...")
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        print("✅ Orchestrator: Data fetched successfully from MCP server.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Orchestrator (MCP Error): {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to the financial data provider: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Orchestrator (JSON Error): {e}")
        raise HTTPException(status_code=500, detail="Invalid data format from financial provider.")

# --- ORCHESTRATOR LOGIC ---

async def run_planner_agent(query: str) -> list:
    """
    Step 1: The Planner. Deconstructs the user query into a list of required agent tasks.
    """
    print("Orchestrator (Step 1 - Planner): Determining required agents...")
    if not orchestrator_model:
        raise HTTPException(status_code=500, detail="Orchestrator AI model is not available.")

    prompt = f"""
    You are a master financial analyst and the orchestrator of a team of specialist AI agents.
    Based on the user's query, you must decide which specialist(s) are needed to form a complete answer.

    **Available Agents (Tools):**
    - "investment_analysis": Use for analyzing the user's portfolio, performance, diversification, asset allocation, XIRR, and suggesting optimizations. Call this for any query related to "mistakes" or "next investments".
    - "tax_analysis": Use for all questions related to tax efficiency, deductions, tax regimes (old vs. new), and specific tax laws (80C, capital gains).
    - "goal_simulation": Use for running simulations, forecasting net worth, and planning for long-term goals like retirement, home purchase, car purchase, or career breaks.
    - "loan_analysis": Use for questions about prepaying loans, comparing EMIs, and general debt management.

    **User Query:**
    "{query}"

    **Your Task:**
    Return ONLY a valid JSON object with a single key, "required_agents", which is a list of the agent names needed to answer the query.
    If the query is a simple greeting or cannot be answered by the tools, return an empty list.

    Example:
    Query: "How can I optimize my portfolio to reach my retirement goal of 10 Cr faster?"
    Output: {{"required_agents": ["investment_analysis", "goal_simulation"]}}

    Query: "Are my investments tax-efficient?"
    Output: {{"required_agents": ["investment_analysis", "tax_analysis"]}}
    """
    try:
        response = await orchestrator_model.generate_content_async(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        plan = json.loads(cleaned_text)
        required_agents = plan.get("required_agents", [])
        print(f"✅ Planner determined: {required_agents}")
        return required_agents
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Planner Error: Could not parse plan. Error: {e}. Defaulting to general chat.")
        return []

async def run_executor(required_agents: list, structured_data: dict, query: str, persona: dict) -> dict:
    """
    Step 2: The Executor. Runs the required agents and collects their analyses.
    """
    print(f"Orchestrator (Step 2 - Executor): Running agents: {required_agents}")
    agent_outputs = {}

    agent_map = {
        "investment_analysis": investment_agent,
        "tax_analysis": tax_agent,
        "goal_simulation": goal_simulation_agent,
    }

    tasks_to_run = []
    for agent_name in required_agents:
        if agent_name in agent_map:
            agent_instance = agent_map[agent_name]
            if agent_name == 'investment_analysis':
                 task = asyncio.to_thread(agent_instance.run, structured_data, query, persona)
            else:
                 task = asyncio.to_thread(agent_instance.run, structured_data, query)
            tasks_to_run.append((agent_name, task))

    results = await asyncio.gather(*(task for _, task in tasks_to_run))

    for i, (agent_name, _) in enumerate(tasks_to_run):
         agent_outputs[f"{agent_name}_data"] = results[i]

    print(f"✅ Executor finished. Collected data from {len(agent_outputs)} agents.")
    return agent_outputs

async def run_synthesis_agent(query: str, persona: dict, agent_outputs: dict, structured_data: dict) -> str:
    """
    Step 3: The Synthesizer. Creates a single, coherent response from all collected data.
    """
    print("Orchestrator (Step 3 - Synthesizer): Generating final response...")
    if not orchestrator_model:
        raise HTTPException(status_code=500, detail="Orchestrator AI model is not available.")

    if not agent_outputs:
        agent_outputs["user_summary_data"] = structured_data["summary"]

    prompt = f"""
    You are Aadi, an elite AI financial advisor. You have already consulted your specialist agents and have gathered the following raw data to answer the user's query.
    Your task is to synthesize all this information into a single, comprehensive, and easy-to-understand response.

    **User's Financial Persona:**
    - Title: {persona.get('title', 'N/A')}
    - Summary: {persona.get('summary', 'N/A')}

    **Original User Query:**
    "{query}"

    **Collected Raw Data from Your Specialist Analysis (JSON format):**
    ---
    {json.dumps(agent_outputs, indent=2, default=str)}
    ---

    **CRITICAL INSTRUCTIONS:**
    1.  **Synthesize, Don't List:** Do NOT just list the data. Weave the insights from the different JSON blocks into a single, flowing narrative. If the data contains an error message, explain it to the user gracefully.
    2.  **Act as One:** Present all findings as your own analysis. Never mention "my agents" or "the data shows". Say "My analysis shows..." or "Based on your portfolio...".
    3.  **Address the Persona:** Tailor your tone and recommendations to the user's persona (e.g., more aggressive for a 'Wealth Builder', more cautious for a 'Conservative Saver').
    4.  **Handle General Queries:** If the 'Collected Raw Data' only contains 'user_summary_data', it means this is a general query. Answer the user's question based on their financial summary and persona.
    5.  **Structure is Key:** Use Markdown headings (e.g., `### Investment Analysis`) to structure the different parts of your answer clearly.
    6.  **Actionable Conclusion:** End with a concise, bulleted summary of key takeaways or a "Plan of Action".

    **BEGIN YOUR COMPREHENSIVE & SYNTHESIZED RESPONSE.**
    """
    try:
        response = await orchestrator_model.generate_content_async(prompt)
        print("✅ Synthesizer finished.")
        return response.text
    except Exception as e:
        print(f"❌ Synthesizer Error: {e}")
        return "I encountered an error while trying to formulate the final response. Please try rephrasing your question."

# --- API ENDPOINTS ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """Health check endpoint to confirm the server is running."""
    return {"status": "Aadi Backend API v2.0 is running"}

@app.post("/chat", response_model=ChatResponse, tags=["Core Logic"])
async def chat_with_aadi(request: ChatRequest):
    """
    Main chat endpoint that uses the Plan-Execute-Synthesize orchestration framework.
    """
    print(f"\n--- New Request for Session '{request.session_id}' ---")
    print(f"Query: {request.query}")

    try:
        raw_data = get_financial_data(request.session_id)
        structured_data = transform_financial_data(raw_data)
        if not structured_data:
            raise HTTPException(status_code=500, detail="Failed to process financial data.")
        
        persona = persona_agent.run(structured_data)
        required_agents = await run_planner_agent(request.query)

        if not required_agents:
            print("Orchestrator: No specific agent required. Routing to General Chat.")
            final_response = await run_synthesis_agent(request.query, persona, {}, structured_data)
        else:
            agent_outputs = await run_executor(required_agents, structured_data, request.query, persona)
            final_response = await run_synthesis_agent(request.query, persona, agent_outputs, structured_data)

        return ChatResponse(response=final_response, status="success")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in /chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/dashboard_data", tags=["Dashboard"])
async def get_dashboard_data(session_id: str) -> Dict[str, Any]:
    """
    Fetches, transforms, and cleans financial data for frontend dashboard consumption.
    This version correctly handles Timestamp objects for JSON serialization.
    """
    print(f"\n--- New Dashboard Data Request for Session '{session_id}' ---")
    try:
        raw_data = get_financial_data(session_id)
        structured_data = transform_financial_data(raw_data)

        if not structured_data:
            raise HTTPException(status_code=500, detail="Failed to process financial data.")

        response_payload = {}
        for key, value in structured_data.items():
            if isinstance(value, pd.DataFrame):
                df = value.replace({np.nan: None})
                
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                    df[col] = df[col].apply(lambda ts: ts.isoformat() if pd.notna(ts) else None)

                response_payload[key] = df.to_dict(orient='records')
            else:
                response_payload[key] = value
        
        print("--- Final JSON Payload Sent to Frontend ---")
        print(json.dumps(response_payload, indent=2))
        print("-----------------------------------------")

        return JSONResponse(content=response_payload)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in /dashboard_data endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while fetching dashboard data: {e}")
    
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Voice client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo from Aadi: {data}")
    except Exception as e:
        print(f"❌ Voice client disconnected: {e}")
    finally:
        print("Voice connection closed.")

# To run this server from your terminal:
# uvicorn main:app --reload