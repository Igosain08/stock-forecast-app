import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_company_fundamentals(ticker_symbol):
    """
    Retrieve fundamental financial data for a company
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Basic company information
        company_info = {
            'name': info.get('longName', ticker_symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A')
        }
        
        # Valuation metrics
        valuation = {
            'market_cap': info.get('marketCap', 'N/A'),
            'enterprise_value': info.get('enterpriseValue', 'N/A'),
            'trailing_pe': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue', 'N/A'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda', 'N/A'),
        }
        
        # Financial metrics
        financials = {
            'profit_margins': info.get('profitMargins', 'N/A'),
            'operating_margins': info.get('operatingMargins', 'N/A'),
            'roa': info.get('returnOnAssets', 'N/A'),
            'roe': info.get('returnOnEquity', 'N/A'),
            'revenue': info.get('totalRevenue', 'N/A'),
            'revenue_per_share': info.get('revenuePerShare', 'N/A'),
            'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth', 'N/A'),
            'gross_margins': info.get('grossMargins', 'N/A'),
            'ebitda_margins': info.get('ebitdaMargins', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'current_ratio': info.get('currentRatio', 'N/A'),
            'quick_ratio': info.get('quickRatio', 'N/A'),
        }
        
        # Dividend information
        dividends = {
            'dividend_rate': info.get('dividendRate', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'payout_ratio': info.get('payoutRatio', 'N/A'),
            'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', 'N/A'),
        }
        
        # Get quarterly financial data
        quarterly_financials = get_quarterly_financials(ticker)
        
        return {
            'company_info': company_info,
            'valuation': valuation,
            'financials': financials,
            'dividends': dividends,
            'quarterly_financials': quarterly_financials
        }
    
    except Exception as e:
        return {'error': str(e)}

def get_quarterly_financials(ticker):
    """
    Retrieve quarterly financial statements
    """
    try:
        # Get quarterly income statement, balance sheet, and cash flow
        quarterly_income = ticker.quarterly_income_stmt
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        # Process the data
        income_data = process_financial_statement(quarterly_income)
        balance_data = process_financial_statement(quarterly_balance)
        cashflow_data = process_financial_statement(quarterly_cashflow)
        
        # Calculate key metrics
        key_metrics = calculate_financial_metrics(income_data, balance_data, cashflow_data)
        
        return {
            'income_statement': income_data,
            'balance_sheet': balance_data,
            'cash_flow': cashflow_data,
            'key_metrics': key_metrics
        }
    
    except Exception as e:
        return {'error': f"Error retrieving quarterly financials: {str(e)}"}

def process_financial_statement(statement):
    """
    Process a financial statement into a more usable format
    """
    if statement is None or statement.empty:
        return {}
    
    # Convert to dictionary format
    result = {}
    for column in statement.columns:
        date_key = column.strftime('%Y-%m-%d')
        result[date_key] = {}
        
        for index, value in statement[column].items():
            result[date_key][index] = None if pd.isna(value) else value
    
    return result

def calculate_financial_metrics(income_data, balance_data, cashflow_data):
    """
    Calculate additional financial metrics from the statements
    """
    metrics = {}
    
    # For each quarter, calculate metrics
    for date in income_data.keys():
        if date in balance_data and date in cashflow_data:
            income = income_data[date]
            balance = balance_data[date]
            cashflow = cashflow_data[date]
            
            # Initialize the quarter's metrics
            metrics[date] = {}
            
            # Calculate metrics if data is available
            # Revenue Growth (quarter-over-quarter)
            if 'Total Revenue' in income:
                metrics[date]['Revenue'] = income.get('Total Revenue')
            
            # Gross Profit Margin
            if 'Total Revenue' in income and 'Gross Profit' in income:
                if income.get('Total Revenue') and income.get('Gross Profit'):
                    metrics[date]['Gross Margin'] = income.get('Gross Profit') / income.get('Total Revenue')
            
            # Net Profit Margin
            if 'Total Revenue' in income and 'Net Income' in income:
                if income.get('Total Revenue') and income.get('Net Income'):
                    metrics[date]['Net Margin'] = income.get('Net Income') / income.get('Total Revenue')
            
            # EPS (Earnings Per Share)
            if 'Net Income' in income and 'Basic EPS' in income:
                metrics[date]['EPS'] = income.get('Basic EPS')
            
            # Add more metrics as needed...
    
    return metrics

def get_analyst_recommendations(ticker_symbol):
    """
    Get analyst recommendations for a stock
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        recommendations = ticker.recommendations
        
        if recommendations is not None and not recommendations.empty:
            # Convert to a more accessible format
            result = []
            for date, row in recommendations.iterrows():
                result.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'firm': row.get('Firm', 'N/A'),
                    'to_grade': row.get('To Grade', 'N/A'),
                    'from_grade': row.get('From Grade', 'N/A'),
                    'action': row.get('Action', 'N/A')
                })
            
            return result
        else:
            return []
    
    except Exception as e:
        return {'error': str(e)}

def get_earnings_trend(ticker_symbol):
    """
    Get earnings estimates and surprises
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        earnings = ticker.earnings
        earnings_dates = ticker.earnings_dates
        
        result = {
            'yearly_earnings': {},
            'quarterly_surprises': []
        }
        
        # Annual earnings
        if earnings is not None and not earnings.empty:
            for year, row in earnings.iterrows():
                result['yearly_earnings'][str(year)] = {
                    'revenue': row.get('Revenue', None),
                    'earnings': row.get('Earnings', None)
                }
        
        # Quarterly earnings surprises
        if earnings_dates is not None and not earnings_dates.empty:
            for date, row in earnings_dates.iterrows():
                result['quarterly_surprises'].append({
                    'date': date.strftime('%Y-%m-%d'),
                    'epsEstimate': row.get('EPS Estimate', None),
                    'epsActual': row.get('EPS Actual', None),
                    'epsSurprise': row.get('Surprise(%)', None)
                })
        
        return result
    
    except Exception as e:
        return {'error': str(e)}
