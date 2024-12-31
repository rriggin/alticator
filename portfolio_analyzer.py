import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date

def get_ytd_returns(tickers):
    """
    Get YTD returns for a list of tickers
    """
    start_date = date(date.today().year, 1, 1)
    end_date = date.today()
    
    returns_dict = {}
    for ticker in tickers:
        # Skip non-ticker inputs (e.g., asset types, comments)
        if not str(ticker).replace(' ', '').isalnum():
            returns_dict[ticker] = None
            continue
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            print(f"\nDebug - {ticker} history:")
            print(f"Start date: {start_date}")
            print(f"End date: {end_date}")
            print(f"Data points: {len(hist)}")
            if len(hist) > 0:
                print(f"First close: {hist['Close'].iloc[0]}")
                print(f"Last close: {hist['Close'].iloc[-1]}")
            
            if not hist.empty:
                ytd_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                returns_dict[ticker] = ytd_return
            else:
                print(f"No data found for {ticker}")
                returns_dict[ticker] = None
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
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
    
    # Get historical data
    historical_data = get_historical_returns(tickers)
    
    # Calculate portfolio historical returns
    portfolio_hist = pd.DataFrame()
    for ticker, weight in zip(tickers, weights):
        if ticker in historical_data.columns:
            if portfolio_hist.empty:
                portfolio_hist = historical_data[ticker] * weight
            else:
                portfolio_hist += historical_data[ticker] * weight
    
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
    
    # Convert portfolio history to dictionary with string dates
    if isinstance(portfolio_hist, pd.Series):
        portfolio_hist_dict = {
            str(k): float(v) if not pd.isna(v) else 0.0 
            for k, v in portfolio_hist.items()
        }
    else:
        portfolio_hist_dict = {}
    
    return {
        'portfolio_df': portfolio_df,
        'portfolio_return': portfolio_return,
        'historical_data': historical_data.to_dict('records') if not historical_data.empty else {},
        'portfolio_hist': portfolio_hist_dict
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
        # Clean up any NaN values from the CSV
        df = df.dropna(subset=['ticker'])
        df = df[df['ticker'] != 'nan']
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

def simulate_portfolio_change(portfolio_df, new_asset, new_weight, rebalance_method='proportional', manual_return=None):
    """
    Simulate adding a new asset to the portfolio
    
    Args:
        portfolio_df: Current portfolio DataFrame
        new_asset: Ticker or name of new asset to add
        new_weight: Target weight for new asset (as decimal)
        rebalance_method: How to rebalance existing weights ('proportional' or 'largest')
        manual_return: Optional manual return rate for non-ticker assets (as decimal)
    """
    # Clean input DataFrame
    portfolio_df = portfolio_df[portfolio_df['ticker'].notna()]
    portfolio_df = portfolio_df[portfolio_df['ticker'] != 'nan'].copy()
    
    # Get all tickers including the new one
    all_tickers = portfolio_df['ticker'].tolist() + [new_asset]
    
    # Handle manual return first if provided
    if manual_return is not None:
        # Create synthetic data structure
        historical_data = pd.DataFrame()
        
        # Get data for existing portfolio tickers
        for ticker in portfolio_df['ticker']:
            if ' ' not in ticker:  # Only get market data for actual tickers
                ticker_data = get_historical_returns([ticker])
                if not ticker_data.empty:
                    historical_data[ticker] = ticker_data[ticker]
        
        # Add synthetic data for the new asset
        if historical_data.empty:
            # If no market data, create basic date range
            dates = pd.date_range(start=date(date.today().year, 1, 1), end=date.today(), freq='B')
            historical_data.index = dates
        
        # Create synthetic returns for the new asset
        base_dates = historical_data.index
        num_days = len(base_dates)
        # Use existing dates but create new values
        dates = base_dates
        values = [i/num_days * manual_return for i in range(num_days)]
        returns = pd.Series(data=values, index=dates)
        historical_data[new_asset] = returns
        # Ensure we have a DataFrame
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
    else:
        # Get historical data for all tickers at once
        historical_data = get_historical_returns(all_tickers)
    
    # Create a copy of the portfolio
    new_portfolio = portfolio_df.copy()
    
    # Calculate how much to reduce existing weights
    total_reduction = new_weight
    
    if rebalance_method == 'proportional':
        # Reduce each position proportionally
        scale_factor = 1 - new_weight
        new_portfolio['weight'] = new_portfolio['weight'] * scale_factor
    elif rebalance_method == 'largest':  # 'largest' method
        # Reduce largest positions first
        sorted_positions = new_portfolio.sort_values('weight', ascending=False)
        remaining_reduction = total_reduction
        
        for idx, row in sorted_positions.iterrows():
            if remaining_reduction <= 0:
                break
            
            reduction = min(row['weight'], remaining_reduction)
            new_portfolio.loc[idx, 'weight'] -= reduction
            remaining_reduction -= reduction
    else:  # 'worst' method
        # Sort by YTD return, handling None values
        new_portfolio['ytd_return'] = pd.to_numeric(new_portfolio['ytd_return'], errors='coerce')
        sorted_positions = new_portfolio.sort_values('ytd_return', ascending=True)
        remaining_reduction = total_reduction
        
        for idx, row in sorted_positions.iterrows():
            if remaining_reduction <= 0:
                break
            
            reduction = min(row['weight'], remaining_reduction)
            new_portfolio.loc[idx, 'weight'] -= reduction
            remaining_reduction -= reduction
    
    # Add new asset
    new_asset_return = None
    if manual_return is not None:
        # Use manual return if provided
        new_asset_return = manual_return
    elif str(new_asset).replace(' ', '').isalnum():
        try:
            # Get YTD return using the same method as the original portfolio
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
    
    # Clean up any NaN rows
    new_portfolio = new_portfolio.dropna(subset=['ticker'])
    new_portfolio = new_portfolio[new_portfolio['ticker'] != 'nan']
    
    # Calculate new portfolio return
    portfolio_return = (new_portfolio['weight'] * new_portfolio['ytd_return']).sum()
    
    # Get historical data for the new portfolio
    tickers = new_portfolio['ticker'].tolist()
    weights = new_portfolio['weight'].tolist()
    print("\nDebug info:")
    print(f"Historical data type: {type(historical_data)}")
    print(f"Historical data columns: {historical_data.columns if isinstance(historical_data, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"Historical data index type: {type(historical_data.index[0]) if not historical_data.empty else 'Empty'}")
    portfolio_hist = calculate_portfolio_historical_returns(historical_data, tickers, weights)
    print(f"Portfolio hist type: {type(portfolio_hist)}")
    print(f"Portfolio hist index type: {type(portfolio_hist.index[0]) if isinstance(portfolio_hist, pd.Series) else 'Not a Series'}")
    print(f"Portfolio hist first few items: {list(portfolio_hist.items())[:3] if isinstance(portfolio_hist, pd.Series) else 'Not a Series'}")

    # Convert portfolio history to dictionary with string dates
    if isinstance(portfolio_hist, pd.Series):
        portfolio_hist_dict = {
            str(k): float(v) if not pd.isna(v) else 0.0 
            for k, v in portfolio_hist.items()
        }
    else:
        portfolio_hist_dict = {}
    
    return {
        'portfolio_df': new_portfolio,
        'portfolio_return': portfolio_return,
        'historical_data': historical_data.to_dict() if isinstance(historical_data, pd.DataFrame) else {},
        'portfolio_hist': portfolio_hist_dict
    }

def get_historical_returns(tickers, interval='1d', period='ytd'):
    """
    Get historical returns for portfolio tickers
    """
    historical_data = pd.DataFrame()
    
    for ticker in tickers:
        # Skip yfinance lookup for non-ticker assets (containing spaces)
        if ' ' in ticker:
            continue
            
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(interval=interval, period=period)
            if not hist.empty:
                # Calculate returns instead of using prices
                returns = hist['Close'].pct_change()
                # Fill first NaN with 0
                returns.iloc[0] = 0
                # Calculate cumulative returns
                cumulative_returns = (1 + returns).cumprod() - 1
                # Convert index to timezone-naive dates
                cumulative_returns.index = cumulative_returns.index.tz_localize(None).strftime('%Y-%m-%d')
                historical_data = pd.concat([historical_data, pd.DataFrame({ticker: cumulative_returns})], axis=1)
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            
    return historical_data

def calculate_portfolio_historical_returns(historical_data, tickers, weights):
    """
    Calculate historical returns for the entire portfolio
    
    Args:
        historical_data: DataFrame with historical prices for each ticker
        tickers: List of ticker symbols
        weights: List of portfolio weights
    """
    portfolio_hist = None
    
    for ticker, weight in zip(tickers, weights):
        if ticker in historical_data.columns:
            if portfolio_hist is None:
                portfolio_hist = historical_data[ticker] * weight
            else:
                portfolio_hist += historical_data[ticker] * weight
    
    # Return empty dict if no data
    if portfolio_hist is None:
        return pd.Series()
    
    return portfolio_hist 