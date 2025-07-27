import json
import pandas as pd
import requests
from datetime import datetime

# --- DATA TRANSFORMATION FUNCTIONS ---

def _parse_all_transactions(raw_data):
    """Parses and combines all transaction types (Bank, MF, Stocks) into one DataFrame."""
    all_txns = []
    
    # 1. Bank Transactions
    bank_data = raw_data.get("fetch_bank_transactions", {}).get("bankTransactions", [])
    for bank in bank_data:
        for txn_list in bank.get("txns", []):
            try:
                amount = float(txn_list[0])
                # Keep amount as is; we'll handle signs later
                txn_type = "CREDIT" if txn_list[3] == 1 else "DEBIT"
                all_txns.append({
                    "date": pd.to_datetime(txn_list[2]), "description": txn_list[1],
                    "amount": amount, "type": txn_type, "source": "BANK"
                })
            except (ValueError, IndexError):
                continue

    # 2. Mutual Fund Transactions
    mf_data = raw_data.get("fetch_mf_transactions", {}).get("mfTransactions", [])
    for fund in mf_data:
        for txn_list in fund.get("txns", []):
            try:
                amount = float(txn_list[4])
                txn_type = "BUY" if txn_list[0] == 1 else "SELL"
                all_txns.append({
                    "date": pd.to_datetime(txn_list[1]), "description": f"{txn_type}: {fund.get('schemeName')}",
                    "amount": amount, "type": txn_type, "source": "MUTUAL_FUND"
                })
            except (ValueError, IndexError):
                continue

    # 3. Stock Transactions
    stock_data = raw_data.get("fetch_stock_transactions", {}).get("stockTransactions", [])
    for stock in stock_data:
        for txn_list in stock.get("txns", []):
            try:
                txn_type_map = {1: "BUY", 2: "SELL", 3: "BONUS", 4: "SPLIT"}
                txn_type = txn_type_map.get(txn_list[0], "UNKNOWN")
                quantity = txn_list[2]
                price = txn_list[3] if len(txn_list) > 3 else 0
                amount = quantity * price if txn_type in ["BUY", "SELL"] else 0
                all_txns.append({
                    "date": pd.to_datetime(txn_list[1]), "description": f"{txn_type}: {stock.get('isin')} ({quantity} units)",
                    "amount": amount, "type": txn_type, "source": "STOCK"
                })
            except (ValueError, IndexError):
                continue
            
    if not all_txns:
        return pd.DataFrame()
        
    return pd.DataFrame(all_txns).sort_values(by="date", ascending=False)

def _create_holdings_df(raw_data):
    """Creates a unified DataFrame of all asset holdings with detailed parsing."""
    all_holdings = []
    net_worth_data = raw_data.get("fetch_net_worth", {})

    mf_analytics = net_worth_data.get("mfSchemeAnalytics", {}).get("schemeAnalytics", [])
    for mf in mf_analytics:
        details = mf.get("schemeDetail", {})
        analytics = mf.get("enrichedAnalytics", {}).get("analytics", {}).get("schemeDetails", {})
        all_holdings.append({
            "asset_class": details.get("assetClass", "EQUITY").upper(), "category": details.get("categoryName", "UNCATEGORIZED").upper(),
            "asset_name": details.get("nameData", {}).get("longName", "Unknown MF"),
            "current_value": float(analytics.get("currentValue", {}).get("units", 0)),
            "invested_value": float(analytics.get("investedValue", {}).get("units", 0)), "xirr": analytics.get("XIRR", 0.0)
        })

    account_details_map = net_worth_data.get("accountDetailsBulkResponse", {}).get("accountDetailsMap", {})
    for account in account_details_map.values():
        if 'equitySummary' in account and account.get("accountDetails", {}).get("accInstrumentType") == "ACC_INSTRUMENT_TYPE_EQUITIES":
            for stock in account['equitySummary'].get('holdingsInfo', []):
                units = float(stock.get('units', 0)); price = float(stock.get('lastTradedPrice', {}).get('units', 0))
                all_holdings.append({"asset_class": "EQUITY", "category": "STOCKS_IND", "asset_name": stock.get('issuerName'), "current_value": units * price, "invested_value": None, "xirr": None})
        if 'equitySummary' in account and account.get("accountDetails", {}).get("accInstrumentType") == "ACC_INSTRUMENT_TYPE_US_SECURITIES":
             for stock in account['equitySummary'].get('holdingsInfo', []):
                USD_INR_RATE = 83.5; units = float(stock.get('units', 0)); price_usd = float(stock.get('lastTradedPrice', {}).get('units', 0))
                all_holdings.append({"asset_class": "EQUITY", "category": "STOCKS_US", "asset_name": stock.get('issuerName'), "current_value": units * price_usd * USD_INR_RATE, "invested_value": None, "xirr": None})
        if 'etfSummary' in account:
            for etf in account['etfSummary'].get('holdingsInfo', []):
                units = float(etf.get('units', 0)); nav = float(etf.get('nav', {}).get('units', 0))
                all_holdings.append({"asset_class": "EQUITY", "category": "ETF", "asset_name": etf.get('isinDescription'), "current_value": units * nav, "invested_value": None, "xirr": None})
        if 'sgbSummary' in account:
             all_holdings.append({"asset_class": "COMMODITY", "category": "GOLD", "asset_name": "Sovereign Gold Bonds", "current_value": float(account['sgbSummary'].get('currentValue', {}).get('units', 0)), "invested_value": None, "xirr": None})

    asset_values = net_worth_data.get("netWorthResponse", {}).get("assetValues", [])
    for asset in asset_values:
        if "ASSET_TYPE_SAVINGS_ACCOUNTS" in asset.get("netWorthAttribute", ""):
            all_holdings.append({"asset_class": "CASH", "category": "SAVINGS", "asset_name": "Savings Account Balance", "current_value": float(asset.get("value", {}).get("units", 0)), "invested_value": None, "xirr": None})
    
    return pd.DataFrame(all_holdings)

def _create_liabilities_df(raw_data):
    """Creates a DataFrame for all liabilities, now including interest rates."""
    all_liabilities = []
    credit_report = raw_data.get("fetch_credit_report", {}).get("creditReports", [{}])[0]
    accounts = credit_report.get("creditReportData", {}).get("creditAccount", {}).get("creditAccountDetails", [])
    for acc in accounts:
        if acc.get("accountStatus") != "11": continue
        all_liabilities.append({
            "lender": acc.get("subscriberName"), 
            "type": "Credit Card", 
            "outstanding": float(acc.get("currentBalance", 0)), 
            "limit": float(acc.get("creditLimitAmount", 0)),
            "interest_rate_pa": float(acc.get("rateOfInterest", 0.0))
        })
    return pd.DataFrame(all_liabilities)

def _create_epf_df(raw_data):
    """Creates a DataFrame with detailed EPF history from each employer."""
    all_epf_records = []
    epf_data = raw_data.get("fetch_epf_details", {}).get("uanAccounts", [{}])[0]
    for record in epf_data.get("rawDetails", {}).get("est_details", []):
        all_epf_records.append({
            "employer_name": record.get("est_name"), 
            "balance": float(record.get("pf_balance", {}).get("net_balance", 0)), 
            "is_active": record.get("doe_epf") == "NOT AVAILABLE"
        })
    return pd.DataFrame(all_epf_records)

def _create_profile_summary(raw_data, liabilities_df, epf_df):
    """Creates a dictionary of key, single-value metrics, now including EPF totals and Total Assets."""
    net_worth_response = raw_data.get("fetch_net_worth", {}).get("netWorthResponse", {})
    credit_report = raw_data.get("fetch_credit_report", {}).get("creditReports", [{}])[0]
    epf_data = raw_data.get("fetch_epf_details", {}).get("uanAccounts", [{}])[0]
    
    net_worth = float(net_worth_response.get("totalNetWorthValue", {}).get("units", 0))
    credit_score = int(credit_report.get("creditReportData", {}).get("score", {}).get("bureauScore", 0))
    total_liabilities = liabilities_df['outstanding'].sum() if not liabilities_df.empty else 0
    
    total_assets = net_worth + total_liabilities

    overall_pf_balance = epf_data.get("rawDetails", {}).get("overall_pf_balance", {})
    total_epf_balance = float(overall_pf_balance.get("current_pf_balance", epf_df['balance'].sum() if not epf_df.empty else 0))
    total_pension_balance = float(overall_pf_balance.get("pension_balance", 0))

    return {
        "Net Worth (INR)": net_worth, 
        "Credit Score": credit_score, 
        "Total Assets (INR)": total_assets,
        "Total Liabilities (INR)": total_liabilities,
        "Total EPF Balance (INR)": total_epf_balance,
        "Total Pension Balance (INR)": total_pension_balance
    }

def _prepare_cashflow_data(transactions_df):
    """
    Finds the top 3 months with the most cashflow activity and returns
    a sorted summary ready for charting.
    """
    if transactions_df.empty or 'date' not in transactions_df.columns:
        return []

    df = transactions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df['month'] = df['date'].dt.to_period('M')

    # Correctly calculate credits and debits based on transaction type
    df['credit'] = df.apply(
        lambda row: row['amount'] if row['type'] == 'CREDIT' else (abs(row['amount']) if row['type'] == 'SELL' else 0), 
        axis=1
    )
    df['debit'] = df.apply(
        lambda row: row['amount'] if row['type'] == 'DEBIT' else (abs(row['amount']) if row['type'] == 'BUY' else 0), 
        axis=1
    )

    monthly_summary = df.groupby('month').agg(
        credit=('credit', 'sum'),
        debit=('debit', 'sum')
    ).reset_index()

    monthly_summary['total_activity'] = monthly_summary['credit'] + abs(monthly_summary['debit'])
    
    top_3_months = monthly_summary.sort_values(by='total_activity', ascending=False).head(3)

    final_summary = top_3_months.sort_values(by='month')

    output = []
    for _, row in final_summary.iterrows():
        output.append({
            "month": str(row['month']),
            "credit": float(row['credit']),
            "debit": abs(float(row['debit'])) 
        })
        
    return output


# --- MAIN TRANSFORMATION FUNCTION ---

def transform_financial_data(raw_data):
    """
    Main function to transform raw JSON into a dictionary of structured data.
    """
    if not raw_data: 
        print("❌ No raw data provided to transformer.")
        return None
    
    transactions_df = _parse_all_transactions(raw_data)
    epf_df = _create_epf_df(raw_data)
    liabilities_df = _create_liabilities_df(raw_data)
    holdings_df = _create_holdings_df(raw_data)
    
    if not epf_df.empty and 'balance' in epf_df.columns:
        epf_total_df = pd.DataFrame([{
            "asset_class": "DEBT", 
            "category": "RETIREMENT", 
            "asset_name": "EPF (Total)", 
            "current_value": epf_df['balance'].sum()
        }])
        epf_total_df = epf_total_df.reindex(columns=holdings_df.columns)
        complete_holdings_df = pd.concat([holdings_df, epf_total_df], ignore_index=True)
    else:
        complete_holdings_df = holdings_df

    # --- DEBUGGING PRINT STATEMENT ADDED HERE ---
    cashflow_summary_data = _prepare_cashflow_data(transactions_df)
    print("\n--- DEBUG: CASHFLOW JSON SENT TO FRONTEND ---")
    print(json.dumps(cashflow_summary_data, indent=2))
    print("-------------------------------------------\n")

    return {
        "transactions": transactions_df,
        "cashflow_summary": cashflow_summary_data,
        "holdings": complete_holdings_df,
        "liabilities": liabilities_df,
        "epf_details": epf_df,
        "summary": _create_profile_summary(raw_data, liabilities_df, epf_df)
    }

# --- Standalone Testing Block ---
if __name__ == "__main__":
    print("Running Data Transformer in standalone test mode...")
    
    TEST_SESSION_ID = "123e4567-e89b-12d3-a456-426614174000" 
    
    def get_financial_data_for_test(session_id):
        data_url = f"http://127.0.0.1:8080/getmydata?sessionId={session_id}"
        print(f"Connecting to MCP server for session: {session_id}...")
        try:
            response = requests.get(data_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Connection Error: {e}"); return None

    raw_data_for_test = get_financial_data_for_test(TEST_SESSION_ID)
    
    if raw_data_for_test:
        structured_data = transform_financial_data(raw_data_for_test)
        
        if structured_data:
            # The new print statement will automatically run here
            pass
        else:
            print("Data transformation failed.")
    else:
        print("Could not fetch data from the server. Halting test.")
