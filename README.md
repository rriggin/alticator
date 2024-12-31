# Alticator

A web application that helps users analyze and simulate changes to their investment portfolio allocations. The tool calculates YTD returns for stocks and supports mixed portfolios with alternative assets.

## Features

- Upload portfolio via CSV or manual input
- Calculate YTD returns for stock positions
- Support for alternative assets (e.g., "Art", "Crypto")
- Manual return rates for non-ticker assets
- Portfolio rebalancing simulation with two methods:
  - Proportional reduction
  - Reduce largest positions first
- Side-by-side comparison of original and simulated portfolios
- Interactive performance chart showing both portfolios

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd alticator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser to:
   ```
   http://127.0.0.1:5000
   ```

## ToDos
- add a line graph for the original portfolio
- add a line graph with both the original portfolio and simulation
- add support for inputing a rate for non ticker assets in simulation
- enhance front end design

## Project Structure

```
alticator/
├── app.py
├── portfolio_analyzer.py
├── requirements.txt
├── README.md
├── static/
│   └── images/
│       └── logo.png
└── templates/
    └── index.html
```
