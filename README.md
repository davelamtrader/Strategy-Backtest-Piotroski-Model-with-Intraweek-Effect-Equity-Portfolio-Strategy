# Strategy-Backtest-Piotroski-Model-with-Intraweek-Effect-Equity-Portfolio-Strategy

## Overview

This repository presents a quantitative investment strategy that combines the principles of value investing, specifically an improved Piotroski F-Score model, with the "day-of-the-week effect" to generate enhanced returns in the equities market . This strategy aims to identify undervalued stocks using a modified Piotroski scoring system and then leverage the statistical anomaly of day-of-the-week price patterns to optimize trading decisions.

## Strategy Components

### 1. Data Acquisition and Preprocessing

-   **Data Source:**  The strategy utilizes historical stock data from the EODHD API (replace 'YOUR_API_KEY' with your actual API key)  .
-   **Data Fetching:**  The provided Python code includes functions to fetch end-of-day (EOD) and fundamental data for S&P 500 constituents, and should be adaptable to other stock universes  . Functions are included to load and cache data, improving efficiency  .
-   **Data Preprocessing:**  Data undergoes lag processing to account for the delay in the availability of financial reports, using previous quarter's data  .

### 2. Enhanced Piotroski F-Score Model

-   **Model Improvement:**  The strategy implements an improved Piotroski F-Score model, adapted from research, using nine fundamental factors  .
-   **Factor Selection:**  The selected factors include return on equity, growth of return on equity, net cash flow from operating activities, OCF > Net Income, growth of long-term debt ratio, growth of current ratio, basic earnings per share, growth of gross profit margin, and growth of total asset turnover  .
-   **Scoring:**  Stocks are scored based on these factors, with higher scores indicating potentially more attractive investment opportunities  .

### 3. Day-of-the-Week Effect Implementation

-   **Analysis:**  The research incorporates analysis of the day-of-the-week effect, which has been observed in the Chinese stock market, with average returns being highest on Mondays  .
-   **Trading Strategy:**  The strategy can be adapted to leverage this effect, potentially buying stocks on Thursday and selling on Monday  .

### 4. Portfolio Construction and Backtesting

-   **Portfolio Selection:**  High-scoring stocks (e.g., scores of 8 or 9) from the improved Piotroski model are selected for the investment portfolio  .
-   **Rebalancing:**  The portfolio is rebalanced quarterly to account for new financial information and market changes  .
-   **Backtesting:**  The provided code includes a framework for backtesting the strategy using historical data, evaluating performance and generating reports  .

## Code Structure

The Python code is modular, with the following key components:

-   get_sp500_constituents(): Fetches S&P 500 constituent tickers  .
-   fetch_eod_data(): Fetches EOD price data  .
-   fetch_fundamental_data(): Fetches fundamental data  .
-   load_and_cache_data(): Loads and caches data for efficiency  .
-   calculate_improved_piotroski(): Calculates the enhanced Piotroski score  .
-   run_backtest(): Executes the backtesting simulation, incorporating day-of-the-week trading logic and portfolio rebalancing  .
-   evaluate_performance(): Generates performance reports using the quantstats library  .
-   analyze_by_market_regime(): Analyzes performance across different market regimes  .
-   plot_monthly_yearly_returns(): Generates plots of monthly and yearly returns  .

## Conclusion

This strategy provides a robust framework for combining value investing principles with market anomalies to potentially generate superior returns. The improved Piotroski model provides a disciplined approach to stock selection, and the incorporation of the day-of-the-week effect seeks to further optimize trading decisions .
