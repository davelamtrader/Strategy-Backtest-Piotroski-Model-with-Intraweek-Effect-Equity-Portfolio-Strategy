#基於Piotroski 模型改良結合週內效應的投資策略
import os
import requests
import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Step 0: Configuration ---
# IMPORTANT: Replace with your actual EODHD API key
EODHD_API_KEY = os.getenv('EODHD_API_KEY', 'YOUR_API_KEY')
# The user-requested backtest period
START_DATE = '2016-01-01'
END_DATE = '2025-07-01'

# --- Step 1: Data Preparation ---

def get_sp500_constituents(api_key):
    """Fetches historical S&P 500 constituents from EODHD."""
    url = f"https://eodhd.com/api/fundamentals/GSPC.INDX?api_token={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Process historical components to get a set of all unique tickers
        historical_components = data.get('HistoricalTickerComponents', [])
        all_tickers = {component['Code'] for component in historical_components}
        
        # Add current components
        current_components = data.get('Components', {})
        for _, component_details in current_components.items():
            all_tickers.add(component_details['Code'])
            
        return list(all_tickers)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching S&P 500 constituents: {e}")
        return []

def fetch_eod_data(ticker, start_date, end_date, api_key):
    """Fetches end-of-day price data for a single ticker."""
    url = f"https://eodhd.com/api/eod/{ticker}.US?from={start_date}&to={end_date}&period=d&api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching EOD data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_fundamental_data(ticker, api_key):
    """Fetches fundamental data for a single ticker."""
    # EODHD provides extensive fundamental data including financial statements 
    url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fundamental data for {ticker}: {e}")
        return None

def load_and_cache_data(constituents, start_date, end_date, api_key):
    """Loads data from cache if available, otherwise fetches and saves it."""
    eod_data_cache = 'eod_data.pkl'
    fundamentals_cache = 'fundamentals.pkl'
    
    if os.path.exists(eod_data_cache) and os.path.exists(fundamentals_cache):
        print("Loading data from cache...")
        all_eod_data = pd.read_pickle(eod_data_cache)
        all_fundamentals = pd.read_pickle(fundamentals_cache)
    else:
        print("Fetching data from EODHD API...")
        all_eod_data = {}
        all_fundamentals = {}
        for i, ticker in enumerate(constituents):
            print(f"Fetching data for {ticker} ({i+1}/{len(constituents)})...")
            eod_df = fetch_eod_data(ticker, start_date, end_date, api_key)
            if not eod_df.empty:
                all_eod_data[ticker] = eod_df
            
            fundamentals = fetch_fundamental_data(ticker, api_key)
            if fundamentals:
                all_fundamentals[ticker] = fundamentals
        
        pd.to_pickle(all_eod_data, eod_data_cache)
        pd.to_pickle(all_fundamentals, fundamentals_cache)
        
    return all_eod_data, all_fundamentals


# --- Step 2: Signal Generation (Improved Piotroski F-Score) ---

def get_fundamental_value(financials, statement, key, date, period='quarterly'):
    """Helper to safely get a financial value for a specific date."""
    try:
        data = financials.get('Financials', {}).get(statement, {}).get(period, {})
        if not data: return None
        
        df = pd.DataFrame(data).transpose()
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the latest report as of the given date (with a lag for reporting)
        # The report suggests using lagged data
        # We assume a 90-day lag for quarterly reports to be safe
        available_data = df[df['date'] <= (date - pd.Timedelta(days=90))]
        if available_data.empty: return None

        latest_report = available_data.sort_values('date', ascending=False).iloc[0]
        return float(latest_report.get(key, 0))
    except (TypeError, KeyError, ValueError):
        return None

def calculate_improved_piotroski(fundamentals, date):
    """
    Calculates the improved Piotroski score based on the Chinese report's criteria. 
    This version is adapted from the report's scoring table.
    """
    if not fundamentals:
        return 0

    score = 0
    
    # --- Profitability ---
    ni = get_fundamental_value(fundamentals, 'Income_Statement', 'netIncome', date)
    total_assets_y0 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalAssets', date)
    total_assets_y1 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalAssets', date - pd.Timedelta(days=365))
    roe_y0 = get_fundamental_value(fundamentals, 'Income_Statement', 'netIncome', date) / get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalAssets', date) if ni and total_assets_y0 else None
    roe_y1 = get_fundamental_value(fundamentals, 'Income_Statement', 'netIncome', date - pd.Timedelta(days=365)) / total_assets_y1 if total_assets_y1 else None
    
    ocf = get_fundamental_value(fundamentals, 'Cash_Flow', 'totalCashFromOperatingActivities', date)

    # 1. Return on Equity (ROE > 0) [Report uses ROE, not ROA]
    if roe_y0 and roe_y0 > 0:
        score += 1
        
    # 2. Growth of ROE (ΔROE > 0)
    if roe_y0 and roe_y1 and roe_y0 > roe_y1:
        score += 1
        
    # 3. Net Cash Flow from Operating Activities (OCF > 0)
    if ocf and ocf > 0:
        score += 1
        
    # 4. OCF > Net Income
    if ocf and ni and ocf > ni:
        score += 1
        
    # --- Solvency ---
    ltd_y0 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'longTermDebt', date)
    ltd_y1 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'longTermDebt', date - pd.Timedelta(days=365))
    
    debt_ratio_y0 = ltd_y0 / total_assets_y0 if ltd_y0 and total_assets_y0 else None
    debt_ratio_y1 = ltd_y1 / total_assets_y1 if ltd_y1 and total_assets_y1 else None

    # 5. Growth of Long-term Debt Ratio < 0
    if debt_ratio_y0 is not None and debt_ratio_y1 is not None and debt_ratio_y0 < debt_ratio_y1:
        score += 1
        
    ca_y0 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalCurrentAssets', date)
    cl_y0 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalCurrentLiabilities', date)
    ca_y1 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalCurrentAssets', date - pd.Timedelta(days=365))
    cl_y1 = get_fundamental_value(fundamentals, 'Balance_Sheet', 'totalCurrentLiabilities', date - pd.Timedelta(days=365))
    
    cr_y0 = ca_y0 / cl_y0 if ca_y0 and cl_y0 else None
    cr_y1 = ca_y1 / cl_y1 if ca_y1 and cl_y1 else None
    
    # 6. Growth of Current Ratio > 0
    if cr_y0 and cr_y1 and cr_y0 > cr_y1:
        score += 1

    # --- Development Ability ---
    # Report replaces 'no new shares issued' with 'Basic EPS > 0' 
    eps = get_fundamental_value(fundamentals, 'Income_Statement', 'epsActual', date)
    
    # 7. Basic Earnings per Share > 0
    if eps and eps > 0:
        score += 1

    # --- Operating Ability ---
    gp_y0 = get_fundamental_value(fundamentals, 'Income_Statement', 'grossProfit', date)
    rev_y0 = get_fundamental_value(fundamentals, 'Income_Statement', 'totalRevenue', date)
    gp_y1 = get_fundamental_value(fundamentals, 'Income_Statement', 'grossProfit', date - pd.Timedelta(days=365))
    rev_y1 = get_fundamental_value(fundamentals, 'Income_Statement', 'totalRevenue', date - pd.Timedelta(days=365))
    
    margin_y0 = gp_y0 / rev_y0 if gp_y0 and rev_y0 else None
    margin_y1 = gp_y1 / rev_y1 if gp_y1 and rev_y1 else None

    # 8. Growth of Gross Profit Margin > 0
    if margin_y0 and margin_y1 and margin_y0 > margin_y1:
        score += 1
        
    asset_turnover_y0 = rev_y0 / total_assets_y0 if rev_y0 and total_assets_y0 else None
    asset_turnover_y1 = rev_y1 / total_assets_y1 if rev_y1 and total_assets_y1 else None

    # 9. Growth of Total Asset Turnover > 0
    if asset_turnover_y0 and asset_turnover_y1 and asset_turnover_y0 > asset_turnover_y1:
        score += 1
        
    return score


# --- Step 3 & 4: Backtest Logic and Execution ---

def run_backtest(eod_data, fundamentals, start_date, end_date):
    """
    Runs the backtest for the combined Piotroski and Day-of-the-Week strategy.
    """
    print("Starting backtest...")
    initial_capital = 100000.0
    cash = initial_capital
    portfolio_value = pd.Series(index=pd.date_range(start_date, end_date, freq='B'), dtype=float)
    positions = {} # {ticker: shares}
    
    rebalance_dates = pd.date_range(start_date, end_date, freq='QS-JAN')
    eligible_stocks = []

    for date in portfolio_value.index:
        current_value = cash
        for ticker, shares in positions.items():
            if ticker in eod_data and date in eod_data[ticker].index:
                current_value += shares * eod_data[ticker].loc[date, 'close']
        portfolio_value[date] = current_value

        # Quarterly Rebalance: Recalculate Piotroski scores
        if date in rebalance_dates:
            print(f"--- Rebalancing for {date.date()} ---")
            new_eligible_stocks = []
            for ticker, fund_data in fundamentals.items():
                if ticker in eod_data and not eod_data[ticker].empty:
                    score = calculate_improved_piotroski(fund_data, date)
                    if score >= 8: # Select stocks with score 8 or 9 
                        new_eligible_stocks.append(ticker)
            eligible_stocks = new_eligible_stocks
            print(f"Found {len(eligible_stocks)} eligible stocks.")

        # Trading Logic: Day-of-the-Week Effect 
        # Buy on Thursday close, sell on Monday close
        weekday = date.weekday()
        
        # Sell on Monday
        if weekday == 0 and positions: # Monday
            print(f"Monday {date.date()}: Selling all positions.")
            for ticker, shares in list(positions.items()):
                if ticker in eod_data and date in eod_data[ticker].index:
                    price = eod_data[ticker].loc[date, 'close']
                    cash += shares * price
                    del positions[ticker]
        
        # Buy on Thursday
        if weekday == 3 and eligible_stocks: # Thursday
            print(f"Thursday {date.date()}: Buying eligible stocks.")
            # Liquidate any remaining positions (should not happen in this logic, but good practice)
            for ticker, shares in list(positions.items()):
                if ticker in eod_data and date in eod_data[ticker].index:
                    price = eod_data[ticker].loc[date, 'close']
                    cash += shares * price
                    del positions[ticker]

            # Buy new portfolio
            capital_per_stock = portfolio_value[date] / len(eligible_stocks)
            for ticker in eligible_stocks:
                if ticker in eod_data and date in eod_data[ticker].index:
                    price = eod_data[ticker].loc[date, 'close']
                    if price > 0:
                        shares_to_buy = capital_per_stock / price
                        positions[ticker] = shares_to_buy
                        cash -= shares_to_buy * price
                        
    returns = portfolio_value.pct_change().dropna()
    return returns

# --- Step 5: Performance Evaluation ---

def evaluate_performance(returns, benchmark_ticker='SPY'):
    """Generates a performance report using quantstats."""
    print("Evaluating performance...")
    qs.extend_pandas()
    # Fetch benchmark data for comparison
    spy_data = qs.utils.download_returns(benchmark_ticker)
    qs.reports.html(returns, benchmark=spy_data, output='strategy_report.html', title='Piotroski + Day-of-Week Strategy')
    print("Performance report generated: strategy_report.html")

# --- Step 6: Deep Analysis ---

def analyze_by_market_regime(strategy_returns, benchmark_ticker='SPY'):
    """Analyzes strategy performance across different market regimes."""
    print("Analyzing performance by market regime...")
    prices = qs.utils.download_prices(benchmark_ticker, period=strategy_returns.index.to_series())
    
    # Define regimes using moving averages
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    
    regimes = pd.Series(np.nan, index=prices.index)
    
    regimes[(prices > ma50) & (ma50 > ma200)] = 'Uptrend'
    regimes[(prices < ma50) & (ma50 < ma200)] = 'Downtrend'
    regimes[(prices > ma200) & (prices < ma50)] = 'Sideways-Down' # Recovering from dip or topping
    regimes[(prices < ma200) & (prices > ma50)] = 'Sideways-Up' # Recovering from crash or falling
    regimes.ffill(inplace=True)
    
    regime_returns = strategy_returns.groupby(regimes).agg(['mean', 'std', 'size'])
    regime_returns['sharpe'] = (regime_returns['mean'] / regime_returns['std']) * np.sqrt(252)
    
    print("\n--- Performance by Market Regime ---")
    print(regime_returns)
    
    # Plot cumulative returns by regime
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for regime_name, group in strategy_returns.groupby(regimes):
        cumulative_returns = (1 + group).cumprod()
        ax.plot(cumulative_returns, label=f'Strategy in {regime_name}')
        
    ax.set_title('Strategy Cumulative Returns by Market Regime')
    ax.set_ylabel('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig('regime_analysis.png')
    print("Regime analysis plot saved: regime_analysis.png")

def plot_monthly_yearly_returns(returns):
    """Plots monthly and yearly return heatmaps."""
    print("Generating monthly and yearly return plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    qs.plots.monthly_heatmap(returns, ax=axes[0], show=False)
    axes[0].set_title('Monthly Return Heatmap')
    
    qs.plots.yearly_returns(returns, ax=axes[1], show=False)
    axes[1].set_title('Annual Returns')
    
    fig.tight_layout()
    plt.savefig('monthly_yearly_returns.png')
    print("Monthly/Yearly return plots saved: monthly_yearly_returns.png")


# --- Main Execution Block ---

if __name__ == '__main__':
    if EODHD_API_KEY == 'YOUR_API_KEY':
        print("Please set your EODHD_API_KEY in the script.")
    else:
        # Step 1
        constituents = get_sp500_constituents(EODHD_API_KEY)
        if constituents:
            all_eod_data, all_fundamentals = load_and_cache_data(constituents, START_DATE, END_DATE, EODHD_API_KEY)
            
            # Step 3 & 4
            strategy_returns = run_backtest(all_eod_data, all_fundamentals, START_DATE, END_DATE)

            if not strategy_returns.empty:
                # Step 5
                evaluate_performance(strategy_returns, benchmark_ticker='SPY')
                
                # Step 6
                analyze_by_market_regime(strategy_returns, benchmark_ticker='SPY')
                plot_monthly_yearly_returns(strategy_returns)
                
                print("\n--- Backtest Complete ---")
            else:
                print("Backtest did not generate any returns.")
        else:
            print("Could not retrieve S&P 500 constituents. Aborting.")
