from flask import Flask, render_template, request, flash, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from portfolio_analyzer import analyze_portfolio, read_portfolio_csv, simulate_portfolio_change, get_historical_returns, calculate_portfolio_historical_returns

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            
            # Analyze portfolio
            result = analyze_portfolio(tickers, weights)
            
            # Store current portfolio in session
            session['current_portfolio'] = result['portfolio_df'].to_dict('records')
            
            return render_template('index.html', 
                                result=result['portfolio_return'],
                                portfolio=result['portfolio_df'].to_dict('records'),
                                historical_data=result['historical_data'],
                                portfolio_hist=result['portfolio_hist'])
            
        except Exception as e:
            flash(str(e))
            return render_template('index.html', historical_data={}, portfolio_hist={})
    
    return render_template('index.html', historical_data={}, portfolio_hist={})

@app.route('/simulate', methods=['POST'])
def simulate():
    """Handle portfolio simulation requests"""
    try:
        new_asset = request.form['new_asset']
        new_weight = float(request.form['new_weight']) / 100
        rebalance_method = request.form['rebalance_method']
        manual_return = None
        if request.form.get('manual_return'):
            manual_return = float(request.form['manual_return']) / 100
        
        # Get original portfolio from session
        original_portfolio = session.get('current_portfolio')
        portfolio_df = pd.DataFrame(original_portfolio)
        
        # Get original portfolio historical data
        original_tickers = portfolio_df['ticker'].tolist()
        original_weights = portfolio_df['weight'].tolist()
        original_historical = get_historical_returns(original_tickers)
        original_hist = calculate_portfolio_historical_returns(original_historical, original_tickers, original_weights)
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
            manual_return=manual_return
        )
        
        # Clean up simulation results
        simulation_portfolio = [
            row for row in simulation['portfolio_df'].to_dict('records')
            if row.get('ticker') and str(row['ticker']) != 'nan' and str(row['ticker']).strip() != ''
        ]
        
        # Store simulation in session
        session['simulation_portfolio'] = simulation_portfolio
        
        return render_template('index.html',
                             portfolio=original_portfolio,
                             result=(portfolio_df['weight'] * portfolio_df['ytd_return']).sum(),
                             simulation_portfolio=simulation_portfolio,
                             simulation_return=simulation['portfolio_return'],
                             historical_data=original_historical.to_dict(),
                             portfolio_hist=original_hist_dict,
                             simulation_portfolio_hist=simulation['portfolio_hist'],
                             simulation=True)
                             
    except Exception as e:
        flash(f"Oops! Couldn't simulate that change: {str(e)} 😊")
        return redirect(url_for('index'))

@app.route('/historical-data')
def historical_data():
    interval = request.args.get('interval', '1d')
    tickers = [row['ticker'] for row in session.get('current_portfolio', [])]
    weights = [row['weight'] for row in session.get('current_portfolio', [])]
    
    historical_data = get_historical_returns(tickers, interval=interval)
    portfolio_hist = calculate_portfolio_historical_returns(historical_data, tickers, weights)
    
    # Get simulation data if it exists in session
    simulation_hist = {}
    if 'simulation_portfolio' in session:
        sim_tickers = [row['ticker'] for row in session['simulation_portfolio']]
        sim_weights = [row['weight'] for row in session['simulation_portfolio']]
        sim_historical = get_historical_returns(sim_tickers, interval=interval)
        simulation_hist = calculate_portfolio_historical_returns(sim_historical, sim_tickers, sim_weights)
        simulation_hist = {str(k): v for k, v in simulation_hist.items()}
    
    # Convert data for JSON response
    def convert_series_to_dict(series):
        if isinstance(series, pd.Series):
            result = {}
            for date_idx, value in series.items():
                date_str = pd.Timestamp(date_idx).strftime('%Y-%m-%d')
                result[date_str] = float(value)
            return result
        return {}
    
    return jsonify({
        'historical_data': historical_data.to_dict('records'),
        'portfolio_hist': convert_series_to_dict(portfolio_hist),
        'simulation_hist': convert_series_to_dict(simulation_hist)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000) 