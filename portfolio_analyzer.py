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
    
    return {
        'portfolio_df': portfolio_df,
        'portfolio_return': portfolio_return,
        'historical_data': historical_data.to_dict('records') if not historical_data.empty else {},
        'portfolio_hist': {str(k): v for k, v in portfolio_hist.items()} if not portfolio_hist.empty else {}
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

def simulate_portfolio_change(portfolio_df, new_asset, new_weight, rebalance_method='proportional'):
    """
    Simulate adding a new asset to the portfolio
    
    Args:
        portfolio_df: Current portfolio DataFrame
        new_asset: Ticker or name of new asset to add
        new_weight: Target weight for new asset (as decimal)
        rebalance_method: How to rebalance existing weights ('proportional' or 'largest')
    """
    # Clean input DataFrame
    portfolio_df = portfolio_df[portfolio_df['ticker'].notna()]
    portfolio_df = portfolio_df[portfolio_df['ticker'] != 'nan'].copy()
    
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
    print("\n=== Debug Info ===")
    print(f"New asset type: {type(new_asset)}, value: {new_asset}")
    print(f"New weight type: {type(new_weight)}, value: {new_weight}")
    
    if str(new_asset).replace(' ', '').isalnum():
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
    print("\nNew row DataFrame:")
    print(new_row.dtypes)
    print(new_row)
    
    new_portfolio = pd.concat([new_portfolio, new_row], ignore_index=True)
    print("\nFull portfolio after concat:")
    print(new_portfolio.dtypes)
    print(new_portfolio)
    
    # Clean up any NaN rows before calculating return
    new_portfolio = new_portfolio.dropna(subset=['ticker'])
    new_portfolio = new_portfolio[new_portfolio['ticker'] != 'nan']
    
    # Calculate new portfolio return
    portfolio_return = (new_portfolio['weight'] * new_portfolio['ytd_return']).sum()
    
    # Clean up the DataFrame more thoroughly
    new_portfolio = new_portfolio[
        (new_portfolio['ticker'].notna()) &  # Remove NaN tickers
        (new_portfolio['weight'].notna()) &  # Remove NaN weights
        (new_portfolio['ticker'] != 'nan') &  # Remove 'nan' strings
        (new_portfolio['ticker'] != '') &     # Remove empty strings
        (new_portfolio['ticker'].str.strip() != '')  # Remove whitespace-only strings
    ].copy()
    
    return {
        'portfolio_df': new_portfolio,
        'portfolio_return': portfolio_return
    } 

def get_historical_returns(tickers, interval='1d', period='ytd'):
    """
    Get historical returns for portfolio tickers
    """
    historical_data = {}
    
    for ticker in tickers:
        if not str(ticker).replace(' ', '').isalnum():
            continue
            
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(interval=interval, period=period)
            if not hist.empty:
                historical_data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            
    return pd.DataFrame(historical_data)

def calculate_portfolio_historical_returns(historical_data, tickers, weights):
    """
    Calculate historical returns for the entire portfolio
    
    Args:
        historical_data: DataFrame with historical prices for each ticker
        tickers: List of ticker symbols
        weights: List of portfolio weights
    """
    portfolio_hist = pd.DataFrame()
    
    for ticker, weight in zip(tickers, weights):
        if ticker in historical_data.columns:
            if portfolio_hist.empty:
                portfolio_hist = historical_data[ticker] * weight
            else:
                portfolio_hist += historical_data[ticker] * weight
    
    return portfolio_hist 