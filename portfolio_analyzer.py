import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date

def get_returns(tickers, start_date=None, end_date=None):
    """
    Get returns for a list of tickers over a specified date range
    """
    if start_date is None or end_date is None:
        start_date = date(date.today().year, 1, 1)
        end_date = date.today()
    else:
        # If end date is in the future, use today's date
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # If dates are in the future, adjust them
        if end_date > date.today():
            print(f"Warning: End date {end_date} is in the future, using today's date instead")
            end_date = date.today()
        if start_date > date.today():
            print(f"Warning: Start date {start_date} is in the future, using 1 year ago")
            start_date = date.today().replace(year=date.today().year - 1)
    
    print(f"\nFetching returns from {start_date} to {end_date}")
    returns_dict = {}
    for ticker in tickers:
        # Skip non-ticker inputs (e.g., asset types, comments)
        if not str(ticker).replace(' ', '').isalnum():
            returns_dict[ticker] = None
            continue
        
        try:
            stock = yf.Ticker(ticker)
            print(f"\nFetching data for {ticker}")
            # Get data up to today if end date is in the future
            hist = stock.history(start=start_date, end=min(end_date, date.today()), interval="1d", prepost=False)
            print(f"Raw history data:")
            print(hist.head())
            print(f"\nDebug - {ticker} history:")
            print(f"Start date: {start_date}")
            print(f"End date: {end_date}")
            print(f"Data points: {len(hist)}")
            if len(hist) > 0:
                print(f"First close: {hist['Close'].iloc[0]}")
                print(f"Last close: {hist['Close'].iloc[-1]}")
                print(f"First date: {hist.index[0]}")
                print(f"Last date: {hist.index[-1]}")
            
            if not hist.empty:
                ytd_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                print(f"Calculated YTD return: {ytd_return * 100:.2f}%")
                returns_dict[ticker] = ytd_return
            else:
                print(f"No data found for {ticker}")
                returns_dict[ticker] = None
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            returns_dict[ticker] = None
            print(f"Could not fetch return data for {ticker}")
    
    return returns_dict

def analyze_portfolio(tickers, weights, start_date=None, end_date=None):
    """
    Analyze portfolio given tickers and weights
    """
    # Use provided dates or default to current year
    if start_date is None or end_date is None:
        start_date = date(date.today().year, 1, 1)
        end_date = date.today()
    else:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    if abs(sum(weights) - 1.0) > 0.0001:
        raise ValueError("Weights must sum to 1")
    
    if len(tickers) != len(weights):
        raise ValueError("Number of tickers must match number of weights")
    
    # Get returns
    returns = get_returns(tickers, start_date=start_date, end_date=end_date)
    print("\nDebug - Returns:")
    for ticker, ret in returns.items():
        if ret is not None:
            print(f"{ticker}: {ret * 100:.2f}%")
        else:
            print(f"{ticker}: None")
    
    # Get historical data
    historical_data = get_historical_returns(tickers, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    
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
    
    # Calculate risk metrics for each asset
    for ticker in tickers:
        if ticker in historical_data.columns:
            # Calculate daily returns from prices
            returns = historical_data[ticker].pct_change()
            # Clean up returns data
            returns = returns.fillna(0)
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(returns)
            print(f"\nDebug - {ticker} risk metrics:")
            print(f"Volatility: {risk_metrics['volatility']}")
            print(f"Sharpe: {risk_metrics['sharpe_ratio']}")
            portfolio_df.loc[portfolio_df['ticker'] == ticker, 'volatility'] = risk_metrics['volatility']
            portfolio_df.loc[portfolio_df['ticker'] == ticker, 'sharpe_ratio'] = risk_metrics['sharpe_ratio']
    
    # Calculate returns for all assets
    returns_df = historical_data.pct_change().fillna(0)
    weights_series = pd.Series(weights, index=tickers)
    valid_assets = weights_series[historical_data.columns]
    
    # Calculate portfolio metrics with covariance
    valid_returns = returns_df[valid_assets.index].copy()
    
    # Initialize valid_weights
    valid_weights = valid_assets.values
    
    # Handle missing data for BTC
    if 'BTC' in valid_assets.index:
        # Get only the period where we have BTC data
        btc_data = valid_returns['BTC']
        btc_start = btc_data[btc_data != 0].index[0]  # Get first valid BTC date
        valid_dates = btc_data[btc_data != 0].index
        valid_returns = valid_returns.loc[valid_dates]
        
        # Create weights series for normalization
        weights_series = pd.Series(valid_weights, index=valid_assets.index)
        valid_weights = weights_series[valid_returns.columns].values
        # Normalize weights
        valid_weights = valid_weights / valid_weights.sum()
            
        print(f"\nDebug - BTC Data Handling:")
        print(f"Using {len(valid_dates)} data points where BTC data exists")
        print(f"BTC start date: {btc_start}")
        print(f"Normalized weights: {valid_weights}")
    
    print(f"\nDebug - Portfolio Returns:")
    print(f"Valid tickers: {valid_assets.index}")
    print(f"Valid weights: {valid_weights}")
    print(f"Returns shape: {valid_returns.shape}")
    
    # Calculate portfolio metrics with covariance
    portfolio_risk_metrics = calculate_portfolio_risk_metrics(
        valid_returns,
        valid_weights
    )
    
    # Calculate individual asset risk metrics
    for ticker in valid_assets.index:
        # Calculate returns from prices
        prices = historical_data[ticker]
        if ticker == 'BTC':
            # Get only non-zero prices for BTC
            prices = prices[prices != 0]
            returns = prices.pct_change()
            print(f"\nDebug - BTC Data:")
            print(f"Valid prices: {len(prices)}")
            print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        else:
            returns = prices.pct_change().fillna(0)
        
        risk_metrics = calculate_risk_metrics(returns)
        print(f"\nDebug - {ticker} metrics:")
        print(f"  Returns length: {len(returns)}")
        print(f"  Mean return: {returns.mean():.6f}")
        print(f"  Volatility: {risk_metrics['volatility']:.4f}")
        print(f"  Sharpe: {risk_metrics['sharpe_ratio']:.4f}")
        portfolio_df.loc[portfolio_df['ticker'] == ticker, 'volatility'] = risk_metrics['volatility']
        portfolio_df.loc[portfolio_df['ticker'] == ticker, 'sharpe_ratio'] = risk_metrics['sharpe_ratio']
    
    print(f"\nDebug - Portfolio Data:")
    print(f"Historical Data: {historical_data.head().to_dict()}")
    print(f"Portfolio History: {portfolio_hist.head().to_dict() if isinstance(portfolio_hist, pd.Series) else {}}")
    
    print("\nDebug - Portfolio Risk Calculation:")
    print(f"Valid tickers: {valid_assets.index}")
    print(f"Valid weights: {valid_weights}")
    print(f"Portfolio metrics: {portfolio_risk_metrics}")
    
    return {
        'portfolio_df': portfolio_df,
        'portfolio_return': portfolio_return,
        'historical_data': historical_data.to_dict('records'),
        'portfolio_hist': portfolio_hist_dict,
        'portfolio_risk': float(portfolio_risk_metrics['volatility']) if portfolio_risk_metrics['volatility'] is not None else 0,
        'portfolio_sharpe': float(portfolio_risk_metrics['sharpe_ratio']) if portfolio_risk_metrics['sharpe_ratio'] is not None else 0,
        'portfolio_sortino': float(portfolio_risk_metrics['sortino_ratio']) if portfolio_risk_metrics['sortino_ratio'] is not None else 0
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

def simulate_portfolio_change(portfolio_df, new_asset, new_weight, rebalance_method='proportional', 
                          manual_return=None, start_date=None, end_date=None):
    """
    Simulate adding a new asset to the portfolio
    
    Args:
        portfolio_df: Current portfolio DataFrame
        new_asset: Ticker or name of new asset to add
        new_weight: Target weight for new asset (as decimal)
        rebalance_method: How to rebalance existing weights ('proportional' or 'largest')
        manual_return: Optional manual return rate for non-ticker assets (as decimal)
        start_date: Start date for historical data
        end_date: End date for historical data
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
                ticker_data = get_historical_returns([ticker], start_date=start_date, end_date=end_date)
                if not ticker_data.empty:
                    historical_data[ticker] = ticker_data[ticker]
        
        # Add synthetic data for the new asset
        if historical_data.empty:
            # If no market data, create basic date range
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
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
            # Get return using the same method as the original portfolio
            returns_dict = get_returns([new_asset], start_date=start_date, end_date=end_date)
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
    
    # Always get fresh historical data for risk calculations
    historical_data = get_historical_returns(
        [t for t in tickers if ' ' not in t],  # Only get data for valid tickers
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate portfolio historical returns
    portfolio_hist = calculate_portfolio_historical_returns(historical_data, tickers, weights)
    
    # Clean and validate returns data
    returns_df = historical_data.pct_change()
    returns_df = returns_df.fillna(0)
    
    # Get valid assets (those with data)
    valid_tickers = [t for t in tickers if t in historical_data.columns]
    valid_weights = [w for t, w in zip(tickers, weights) if t in historical_data.columns]
    
    # Normalize valid weights to sum to 1
    if valid_weights:
        weight_sum = sum(valid_weights)
        valid_weights = [w/weight_sum for w in valid_weights]
    
    # Calculate portfolio-level metrics using covariance
    valid_returns = pd.DataFrame()
    valid_weights = []
    
    # Build returns matrix and weights vector
    for ticker in valid_tickers:
        if ticker == 'BTC':
            prices = historical_data[ticker][historical_data[ticker] != 0]
        else:
            prices = historical_data[ticker]
        returns = prices.pct_change().fillna(0)
        valid_returns[ticker] = returns
        valid_weights.append(new_portfolio.loc[new_portfolio['ticker'] == ticker, 'weight'].iloc[0])
    
    # Normalize weights
    valid_weights = np.array(valid_weights) / sum(valid_weights)
    
    print("\nDebug - Portfolio Calculation:")
    print(f"Returns shape: {valid_returns.shape}")
    print(f"Weights: {valid_weights}")
    
    # Calculate portfolio metrics with covariance
    portfolio_risk_metrics = calculate_portfolio_risk_metrics(
        valid_returns,
        valid_weights
    )
    
    # Calculate individual asset risk metrics
    for ticker in valid_tickers:
        # Calculate returns from prices
        prices = historical_data[ticker]
        if ticker == 'BTC':
            # Get only non-zero prices for BTC
            prices = prices[prices != 0]
            returns = prices.pct_change()
            print(f"\nDebug - BTC Data:")
            print(f"Valid prices: {len(prices)}")
            print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        else:
            returns = prices.pct_change().fillna(0)
        
        risk_metrics = calculate_risk_metrics(returns)
        new_portfolio.loc[new_portfolio['ticker'] == ticker, 'volatility'] = risk_metrics['volatility']
        new_portfolio.loc[new_portfolio['ticker'] == ticker, 'sharpe_ratio'] = risk_metrics['sharpe_ratio']
        new_portfolio.loc[new_portfolio['ticker'] == ticker, 'sortino_ratio'] = risk_metrics['sortino_ratio']
    
    # Convert portfolio history to dictionary with string dates
    if isinstance(portfolio_hist, pd.Series):
        portfolio_hist_dict = {
            str(k): float(v) if not pd.isna(v) else 0.0 
            for k, v in portfolio_hist.items()
        }
    else:
        portfolio_hist_dict = {}
    
    print("\nDebug - Portfolio Risk Calculation:")
    print(f"Valid tickers: {valid_tickers}")
    print(f"Valid weights: {valid_weights}")
    print(f"Portfolio metrics: {portfolio_risk_metrics}")
    
    return {
        'portfolio_df': new_portfolio,
        'portfolio_return': portfolio_return,
        'historical_data': historical_data.to_dict() if isinstance(historical_data, pd.DataFrame) else {},
        'portfolio_hist': portfolio_hist_dict,
        'portfolio_risk': float(portfolio_risk_metrics['volatility']) if portfolio_risk_metrics['volatility'] is not None else 0,
        'portfolio_sharpe': float(portfolio_risk_metrics['sharpe_ratio']) if portfolio_risk_metrics['sharpe_ratio'] is not None else 0,
        'portfolio_sortino': float(portfolio_risk_metrics['sortino_ratio']) if portfolio_risk_metrics['sortino_ratio'] is not None else 0
    }

def get_historical_returns(tickers, interval='1d', start_date=None, end_date=None):
    """
    Get historical returns for portfolio tickers
    """
    # Get data for all tickers first
    ticker_data = {}
    
    # Return empty DataFrame with proper index if no tickers
    if not tickers:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    
    # Use provided dates or default to YTD
    if start_date is None or end_date is None:
        start_date = date(date.today().year, 1, 1)
        end_date = date.today()
    else:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # If end date is in the future, use today's date
    if end_date > date.today():
        print(f"Warning: End date {end_date} is in the future, using today's date instead")
        end_date = date.today()
    
    # If start date is in the future, use 1 year ago from today
    if start_date > date.today():
        print(f"Warning: Start date {start_date} is in the future, using 1 year ago")
        start_date = date.today().replace(year=date.today().year - 1)
    
    print(f"\nFetching historical returns for {tickers}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Interval: {interval}")
    
    # Get data for all tickers first
    ticker_data = {}
    
    for ticker in tickers:
        # Skip yfinance lookup for non-ticker assets (containing spaces)
        if ' ' in ticker:
            print(f"Skipping {ticker} - contains spaces")
            continue
            
        try:
            stock = yf.Ticker(ticker)
            # Get data for our date range
            hist = stock.history(interval=interval, start=start_date, end=end_date, prepost=False)
            if not hist.empty:
                # Store prices directly instead of returns
                prices = hist['Close']
                # Convert index to timezone-naive dates
                prices.index = pd.to_datetime(prices.index).tz_localize(None)
                ticker_data[ticker] = prices
                print(f"Got {len(hist)} data points for {ticker}")
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
    
    # Create DataFrame with common date range
    if ticker_data:
        # Get common date range from actual data
        all_dates = sorted(set().union(*[data.index for data in ticker_data.values()]))
        historical_data = pd.DataFrame(index=all_dates)
        
        # Add each ticker's data
        for ticker, data in ticker_data.items():
            historical_data[ticker] = data
            
        # Forward fill missing values
        historical_data = historical_data.ffill()
        
        # Fill remaining NaN with 0
        historical_data = historical_data.fillna(0)
    
    # Convert index to string dates only if we have data
    if not historical_data.empty and isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = historical_data.index.strftime('%Y-%m-%d')
    
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

def calculate_risk_metrics(returns, risk_free_rate=0.05):
    """Calculate risk metrics for a series of returns"""
    # If input is prices, convert to returns
    if returns.mean() > 1:  # Likely prices rather than returns
        returns = returns.pct_change().fillna(0)
    
    # Print detailed debug info
    print(f"\nDetailed Risk Metrics Calculation:")
    print(f"Daily returns stats:")
    print(f"  Mean: {returns.mean():.6f}")
    print(f"  Std Dev: {returns.std():.6f}")
    print(f"  Min: {returns.min():.6f}")
    print(f"  Max: {returns.max():.6f}")
    
    # Annualize metrics
    periods = 252  # Daily data
    ann_factor = np.sqrt(periods)
    
    # Calculate annualized metrics
    ann_return = (1 + returns.mean()) ** periods - 1
    ann_vol = returns.std() * ann_factor
    
    # Calculate excess return
    excess_return = ann_return - risk_free_rate
    
    # Calculate Sharpe ratio
    sharpe = excess_return / ann_vol if ann_vol != 0 else 0
    
    # Calculate Sortino ratio (using only negative returns)
    downside_returns = returns[returns < 0]
    downside_var = downside_returns.var() * periods  # Annualize variance
    downside_vol = np.sqrt(downside_var) if not downside_returns.empty else 0
    sortino = excess_return / downside_vol if downside_vol != 0 else 0
    
    print(f"\nAnnualized metrics:")
    print(f"  Return: {ann_return:.4%}")
    print(f"  Volatility: {ann_vol:.4%}")
    print(f"  Risk-free rate: {risk_free_rate:.4%}")
    print(f"  Sharpe ratio: {sharpe:.4f}")
    print(f"  Sortino ratio: {sortino:.4f}")
    
    return {
        'volatility': float(ann_vol) if not np.isnan(ann_vol) else 0,
        'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else 0,
        'sortino_ratio': float(sortino) if not np.isnan(sortino) else 0
    }

def calculate_portfolio_risk_metrics(returns_df, weights):
    """Calculate portfolio risk metrics accounting for covariance"""
    # Validate inputs
    if returns_df.empty or len(weights) == 0:
        return {
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
    
    # Clean up any NaN values in returns
    returns_df = returns_df.fillna(0)
    
    # Special handling for BTC - use only the period where we have BTC data
    if 'BTC' in returns_df.columns:
        btc_data = returns_df['BTC']
        # Get only dates where BTC is non-zero
        valid_dates = btc_data[btc_data != 0].index
        returns_df = returns_df.loc[valid_dates]
        print("\nDebug - BTC Data:")
        print(f"Using {len(valid_dates)} data points where BTC data exists")
        print(f"Date range: {valid_dates[0]} to {valid_dates[-1]}")
    
    # Calculate covariance matrix first
    cov_matrix = returns_df.cov() * 252  # Annualize covariance
    
    print("\nDebug - Covariance Details:")
    print(f"Covariance matrix diagonal (individual variances):")
    for i, ticker in enumerate(returns_df.columns):
        print(f"{ticker}: {cov_matrix.iloc[i,i]:.4f}")
    
    if 'BTC' in returns_df.columns:
        print("\nBTC Correlations:")
        btc_corr = returns_df.corr()['BTC'].sort_values(ascending=False)
        print(btc_corr)
    
    print("\nDebug - Portfolio Variance Calculation:")
    print("Individual Contributions:")
    for i, ticker in enumerate(returns_df.columns):
        contribution = weights[i]**2 * cov_matrix.iloc[i,i]
        print(f"{ticker}: {contribution:.6f} (weight: {weights[i]:.4f})")
    
    print("\nCovariance Terms:")
    for i, ticker1 in enumerate(returns_df.columns):
        for j, ticker2 in enumerate(returns_df.columns):
            if i < j:  # Only print upper triangle
                cov_term = 2 * weights[i] * weights[j] * cov_matrix.iloc[i,j]
                print(f"{ticker1}-{ticker2}: {cov_term:.6f}")
    
    # Calculate portfolio variance using matrix multiplication
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var) if not np.isnan(portfolio_var) else 0
    
    # Calculate daily returns for the portfolio
    portfolio_returns = returns_df.mul(weights, axis=1).sum(axis=1)
    ann_return = (1 + portfolio_returns.mean()) ** 252 - 1
    
    print("\nDebug - Data Validation:")
    print(f"Original data points: {len(returns_df)}")
    print(f"Portfolio returns points: {len(portfolio_returns)}")
    
    # Calculate Sharpe ratio
    excess_return = ann_return - 0.05  # Using 5% risk-free rate
    # Calculate Sharpe using annualized metrics
    sharpe = excess_return / portfolio_vol if portfolio_vol != 0 else 0
    
    # Calculate Sortino ratio (using only negative returns)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_var = downside_returns.var() * 252  # Annualize variance
    downside_vol = np.sqrt(downside_var) if not downside_returns.empty else 0
    sortino = excess_return / downside_vol if downside_vol != 0 else 0
    
    print(f"\nDebug - Portfolio Calculations:")
    print(f"Portfolio variance: {portfolio_var:.6f}")
    print(f"Portfolio volatility: {portfolio_vol:.4%}")
    print(f"Annual return: {ann_return:.4%}")
    print(f"Excess return: {excess_return:.4%}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Sortino ratio: {sortino:.4f}")
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Calculate and show diversification benefit
    individual_variance = sum(weights[i]**2 * cov_matrix.iloc[i,i] for i in range(len(weights)))
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    diversification_benefit = individual_variance - portfolio_variance
    
    print("\nDiversification Analysis:")
    print(f"Sum of individual variances: {individual_variance:.6f}")
    print(f"Portfolio variance: {portfolio_variance:.6f}")
    print(f"Diversification benefit: {diversification_benefit:.6f}")
    print(f"Percent reduction in risk: {(diversification_benefit/individual_variance)*100:.2f}%")
    
    return {
        'volatility': float(portfolio_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino)
    } 