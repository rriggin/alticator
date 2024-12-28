import pandas as pd
import yfinance as yf
from datetime import datetime, date

def get_ytd_returns(tickers):
    """
    Get YTD returns for a list of tickers
    """
    start_date = date(date.today().year, 1, 1)  # January 1st of current year
    end_date = date.today()
    
    returns_dict = {}
    for ticker in tickers:
        # Skip non-ticker inputs (e.g., asset types, comments)
        if not ticker.isalnum():
            returns_dict[ticker] = None
            continue
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                ytd_return = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]
                returns_dict[ticker] = ytd_return
            else:
                returns_dict[ticker] = None
        except Exception as e:
            returns_dict[ticker] = None
            print(f"Could not fetch return data for {ticker}")
    
    return returns_dict

def analyze_portfolio(tickers, weights):
    """
    Analyze portfolio given tickers and weights
    """
    if abs(sum(weights) - 1.0) > 0.0001:
        raise ValueError("Weights must sum to 1")
    
    if len(tickers) != len(weights):
        raise ValueError("Number of tickers must match number of weights")
    
    # Get YTD returns
    returns = get_ytd_returns(tickers)
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'ytd_return': [returns[ticker] for ticker in tickers]
    })
    
    # Calculate portfolio return
    # Only include assets with return data in the calculation
    valid_returns = portfolio_df[portfolio_df['ytd_return'].notna()]
    portfolio_return = (valid_returns['weight'] * valid_returns['ytd_return']).sum()
    
    return {
        'portfolio_df': portfolio_df,
        'portfolio_return': portfolio_return
    }

def read_portfolio_csv(file):
    """
    Read portfolio from CSV file
    Expected format:
    ticker,weight
    AAPL,0.4
    MSFT,0.3
    """
    try:
        df = pd.read_csv(file)
        required_columns = ['ticker', 'weight']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV must contain columns: ticker, weight")
        
        # Convert weight column to float, handling percentage strings if present
        df['weight'] = df['weight'].apply(lambda x: float(str(x).strip('%')) / 100 if isinstance(x, str) and '%' in str(x) else float(x))
        
        # Verify weights sum to 1
        if not abs(df['weight'].sum() - 1.0) < 0.0001:
            raise ValueError("Portfolio weights must sum to 1")
        
        return df['ticker'].tolist(), df['weight'].tolist()
            
    except Exception as e:
        if "could not convert string to float" in str(e):
            raise Exception("Oops! Make sure your weights are numbers (like 0.4 or 40%)")
        else:
            raise Exception(f"Error reading portfolio file: {str(e)}") 

def simulate_portfolio_change(portfolio_df, new_asset, new_weight, rebalance_method='proportional'):
    """
    Simulate adding a new asset to the portfolio
    
    Args:
        portfolio_df: Current portfolio DataFrame
        new_asset: Ticker or name of new asset to add
        new_weight: Target weight for new asset (as decimal)
        rebalance_method: How to rebalance existing weights ('proportional' or 'largest')
    """
    # Create a copy of the portfolio
    new_portfolio = portfolio_df.copy()
    
    # Calculate how much to reduce existing weights
    total_reduction = new_weight
    
    if rebalance_method == 'proportional':
        # Reduce each position proportionally
        scale_factor = 1 - new_weight
        new_portfolio['weight'] = new_portfolio['weight'] * scale_factor
    else:  # 'largest' method
        # Reduce largest positions first
        sorted_positions = new_portfolio.sort_values('weight', ascending=False)
        remaining_reduction = total_reduction
        
        for idx, row in sorted_positions.iterrows():
            if remaining_reduction <= 0:
                break
            
            reduction = min(row['weight'], remaining_reduction)
            new_portfolio.loc[idx, 'weight'] -= reduction
            remaining_reduction -= reduction
    
    # Add new asset
    new_asset_return = None
    if new_asset.isalnum():  # Only fetch return if it looks like a ticker
        try:
            returns_dict = get_ytd_returns([new_asset])
            new_asset_return = returns_dict[new_asset]
        except:
            new_asset_return = None
    
    new_row = pd.DataFrame({
        'ticker': [new_asset],
        'weight': [new_weight],
        'ytd_return': [new_asset_return]
    })
    
    new_portfolio = pd.concat([new_portfolio, new_row], ignore_index=True)
    
    # Calculate new portfolio return
    portfolio_return = (new_portfolio['weight'] * new_portfolio['ytd_return']).sum()
    
    return {
        'portfolio_df': new_portfolio,
        'portfolio_return': portfolio_return
    } 