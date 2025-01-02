from flask import Flask, render_template, request, flash, session, redirect, url_for, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import pandas as pd
from portfolio_analyzer import analyze_portfolio, read_portfolio_csv, simulate_portfolio_change, get_historical_returns, calculate_portfolio_historical_returns, calculate_risk_metrics
from datetime import date

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'portfolio_file' in request.files and request.files['portfolio_file'].filename:
                # Handle CSV upload
                file = request.files['portfolio_file']
                if not allowed_file(file.filename):
                    flash('Invalid file type. Please upload a CSV file.')
                    return render_template('index.html')
                
                tickers, weights = read_portfolio_csv(file)
            else:
                # Handle manual input
                if not request.form.get('tickers') and not request.files['portfolio_file'].filename:
                    flash("Wait, you have to upload a file first. Or you can type your tickers in manually. 😊")
                    return render_template('index.html')
                
                tickers = request.form['tickers'].upper().split()
                weights = [float(w) for w in request.form['weights'].split()]
            
            # Get date range from form
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            
            # Analyze portfolio
            result = analyze_portfolio(tickers, weights, start_date=start_date, end_date=end_date)
            
            print("Debug - Portfolio Data:")
            print(f"Historical Data: {result['historical_data']}")
            print(f"Portfolio History: {result['portfolio_hist']}")
            
            # Store current portfolio in session
            session['current_portfolio'] = result['portfolio_df'].to_dict('records')
            
            return render_template('index.html', 
                                result=result['portfolio_return'],
                                portfolio=result['portfolio_df'].to_dict('records'),
                                historical_data=result['historical_data'],
                                portfolio_hist=result['portfolio_hist'],
                                portfolio_risk=result['portfolio_risk'],
                                portfolio_sharpe=result['portfolio_sharpe'],
                                portfolio_sortino=result['portfolio_sortino'])
            
        except Exception as e:
            flash(str(e))
            return render_template('index.html', historical_data={}, portfolio_hist={})
    
    return render_template('index.html', historical_data={}, portfolio_hist={})

@app.route('/simulate', methods=['GET', 'POST'])
def simulate():
    """Handle portfolio simulation requests"""
    if request.method == 'GET':
        # Check if there's a portfolio in the session
        if not session.get('current_portfolio'):
            flash("Please upload a portfolio before running simulations. 📊")
        else:
            flash("To run a simulation, please use the simulation form below. 📈")
        return redirect(url_for('index'))

    try:
        new_asset = request.form['new_asset']
        new_weight = float(request.form['new_weight']) / 100
        rebalance_method = request.form['rebalance_method']
        manual_return = None
        if request.form.get('manual_return'):
            manual_return = float(request.form['manual_return']) / 100
        
        # Get date range from form
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        interval = request.form.get('interval', '1d')  # Default to daily if not provided
        
        # If no dates provided, use defaults
        if not start_date or not end_date:
            if date.today().month == 1:
                start_date = date(date.today().year - 1, 1, 1).strftime('%Y-%m-%d')
                end_date = date(date.today().year - 1, 12, 31).strftime('%Y-%m-%d')
            else:
                start_date = date(date.today().year, 1, 1).strftime('%Y-%m-%d')
                end_date = date.today().strftime('%Y-%m-%d')
        
        # Get original portfolio from session
        original_portfolio = session.get('current_portfolio')
        portfolio_df = pd.DataFrame(original_portfolio)
        
        # Get original portfolio historical data
        original_tickers = portfolio_df['ticker'].tolist()
        original_weights = portfolio_df['weight'].tolist()
        original_historical = get_historical_returns(original_tickers, interval=interval, start_date=start_date, end_date=end_date)
        original_hist = calculate_portfolio_historical_returns(original_historical, original_tickers, original_weights)
        
        # Calculate risk metrics for original portfolio
        original_returns = original_hist.pct_change().fillna(0)
        original_risk_metrics = calculate_risk_metrics(original_returns)
        
        original_hist_dict = {
            str(k): float(v) if not pd.isna(v) else 0.0 
            for k, v in original_hist.items()
        }
        
        # Simulate changes
        simulation = simulate_portfolio_change(
            portfolio_df,
            new_asset, 
            new_weight,
            rebalance_method,
            manual_return=manual_return,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get simulation historical data
        sim_tickers = [row['ticker'] for row in simulation['portfolio_df'].to_dict('records')]
        sim_weights = [row['weight'] for row in simulation['portfolio_df'].to_dict('records')]
        sim_historical = get_historical_returns(
            sim_tickers, 
            interval=interval,
            start_date=start_date, 
            end_date=end_date
        )
        print(f"Raw simulation historical data:")
        print(sim_historical.head())
        simulation_hist = calculate_portfolio_historical_returns(sim_historical, sim_tickers, sim_weights)
        sim_hist_dict = {
            str(k): float(v) if not pd.isna(v) else 0.0 
            for k, v in simulation_hist.items()
        }
        
        # Clean up simulation results
        simulation_portfolio = [
            row for row in simulation['portfolio_df'].to_dict('records')
            if row.get('ticker') and str(row['ticker']) != 'nan' and str(row['ticker']).strip() != ''
        ]
        
        # Store simulation in session
        session['simulation_portfolio'] = simulation_portfolio
        session['current_interval'] = interval
        session['simulation_data'] = {
            'tickers': sim_tickers,
            'weights': [float(w) for w in sim_weights]  # Convert to basic floats
        }
        
        return render_template('index.html',
                             portfolio=original_portfolio,
                             result=(portfolio_df['weight'] * portfolio_df['ytd_return']).sum(),
                             simulation_portfolio=simulation_portfolio,
                             simulation_return=simulation['portfolio_return'],
                             historical_data=original_historical.to_dict(),
                             portfolio_hist=original_hist_dict,
                             portfolio_risk=original_risk_metrics['volatility'],
                             portfolio_sharpe=original_risk_metrics['sharpe_ratio'],
                             portfolio_sortino=original_risk_metrics['sortino_ratio'],
                             simulation_risk=simulation['portfolio_risk'],
                             simulation_sharpe=simulation['portfolio_sharpe'],
                             simulation_sortino=simulation['portfolio_sortino'],
                             simulation_data={
                                 'original': original_hist_dict,
                                 'simulation': sim_hist_dict
                             },
                             simulation=True)
                             
    except Exception as e:
        flash(f"Oops! Couldn't simulate that change: {str(e)} 😊")
        return redirect(url_for('index'))

@app.route('/historical-data', methods=['GET', 'POST'])
def historical_data():
    # Move convert_series_to_dict function to the top
    def convert_series_to_dict(series):
        if isinstance(series, pd.Series):
            result = {}
            for date_idx, value in series.items():
                try:
                    date_str = pd.Timestamp(date_idx).strftime('%Y-%m-%d')
                    result[date_str] = float(value)
                except Exception as e:
                    print(f"Error converting value: {e}")
                    print(f"date_idx: {date_idx}, value: {value}")
            return result
        return {}

    # Get interval from either POST or GET, or session
    interval = request.form.get('interval') or request.args.get('interval') or session.get('current_interval', '1d')
    start_date = request.args.get('start') or request.form.get('start_date')
    end_date = request.args.get('end') or request.form.get('end_date')
    
    print(f"\nDebug - Historical Data Request:")
    print(f"Interval: {interval}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    
    # Get original portfolio data
    tickers = [row['ticker'] for row in session.get('current_portfolio', [])]
    weights = [row['weight'] for row in session.get('current_portfolio', [])]
    historical_data = get_historical_returns(tickers, interval=interval, start_date=start_date, end_date=end_date)
    portfolio_hist = calculate_portfolio_historical_returns(historical_data, tickers, weights)
    
    # Get simulation data if it exists
    simulation_hist = {}
    if 'simulation_data' in session and session['simulation_data'].get('tickers'):
        print("\nDebug - Simulation Data:")
        sim_data = session.get('simulation_data', {})
        print(f"Session simulation data: {sim_data}")
        sim_tickers = sim_data.get('tickers', [])
        # Convert weights back to float if needed
        sim_weights = [float(w) if isinstance(w, str) else w for w in sim_data.get('weights', [])]
        
        if sim_tickers and sim_weights:
            print(f"Simulation Tickers: {sim_tickers}")
            print(f"Simulation Weights: {sim_weights}")
            
            sim_historical = get_historical_returns(
                sim_tickers, 
                interval=interval,
                start_date=start_date, 
                end_date=end_date
            )
            simulation_hist = calculate_portfolio_historical_returns(sim_historical, sim_tickers, sim_weights)
            print(f"\nSimulation Portfolio History:")
            print(f"Length: {len(simulation_hist) if isinstance(simulation_hist, pd.Series) else 'Not a Series'}")
    
    # Prepare response
    response_data = {
        'historical_data': historical_data.to_dict('records'),
        'portfolio_hist': convert_series_to_dict(portfolio_hist),
        'simulation_hist': convert_series_to_dict(simulation_hist)
    }
    
    print(f"\nResponse data simulation hist: {response_data['simulation_hist']}")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000) 