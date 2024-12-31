from flask import Flask, render_template, request, flash, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
from portfolio_analyzer import analyze_portfolio, read_portfolio_csv, simulate_portfolio_change

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
                    # Check which form was submitted
                    if 'portfolio_file' in request.files:
                        flash("Wait, you have to upload a file first. Or you can type your tickers in manually. 😊")
                    else:
                        flash("Oops, you have to enter your tickers first. 😊")
                    return render_template('index.html')
                
                tickers = request.form['tickers'].upper().split()
                weights = [float(w) for w in request.form['weights'].split()]
            
            # Analyze portfolio
            result = analyze_portfolio(tickers, weights)
            
            # Store current portfolio in session
            session['current_portfolio'] = result['portfolio_df'].to_dict('records')
            
            return render_template('index.html', 
                                result=result['portfolio_return'],
                                portfolio=result['portfolio_df'].to_dict('records'))
            
        except Exception as e:
            # Make error messages more user-friendly
            error_msg = str(e)
            if '400 Bad Request' in error_msg:
                # Check which form was submitted
                if 'portfolio_file' in request.files:
                    error_msg = "Wait, you have to upload a file first. Or you can type your tickers in manually. 😊"
                else:
                    error_msg = "Oops, you have to enter your tickers first. 😊"
            flash(error_msg)
            return render_template('index.html')
    
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    """Handle portfolio simulation requests"""
    try:
        new_asset = request.form['new_asset']
        new_weight = float(request.form['new_weight']) / 100
        rebalance_method = request.form['rebalance_method']
        
        # Get original portfolio from session
        original_portfolio = session.get('current_portfolio')
        # Convert list of dictionaries back to DataFrame
        portfolio_df = pd.DataFrame(original_portfolio)
        
        # Simulate changes
        simulation = simulate_portfolio_change(
            portfolio_df,
            new_asset, 
            new_weight,
            rebalance_method
        )
        
        # Clean up simulation results
        simulation_portfolio = [
            row for row in simulation['portfolio_df'].to_dict('records')
            if row.get('ticker') and str(row['ticker']) != 'nan' and str(row['ticker']).strip() != ''
        ]
        print("\nFiltered simulation portfolio:")
        for row in simulation_portfolio:
            print(f"Row: {row}")
        
        return render_template('index.html',
                             portfolio=original_portfolio,
                             result=(portfolio_df['weight'] * portfolio_df['ytd_return']).sum(),
                             simulation_portfolio=simulation_portfolio,
                             simulation_return=simulation['portfolio_return'],
                             simulation=True)
                             
    except Exception as e:
        flash(f"Oops! Couldn't simulate that change: {str(e)} 😊")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000) 