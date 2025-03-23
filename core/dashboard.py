# dashboard.py
import datetime
import dash
import json
import os
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from algo_trading_system import TradingEngine, config, Backtester
from pathlib import Path

# Function to load optimized parameters
def load_optimized_params():
    try:
        # Check if optimized_params.json exists
        params_file = Path("optimized_params.json")
        if params_file.exists():
            with open(params_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded optimized parameters from {params_file}")
            return data
        else:
            print(f"File {params_file} not found, using default config")
            return None
    except Exception as e:
        print(f"Error loading optimized parameters: {e}")
        return None

# Function to update config with optimized parameters
def update_config_with_optimized_params(config, optimized_data, strategy_toggle='all'):
    if not optimized_data or 'params' not in optimized_data:
        return config
        
    # Create a copy of the config to modify
    updated_config = config.copy()
    
    # Make sure strategies dict exists
    if 'strategies' not in updated_config:
        updated_config['strategies'] = {}
    
    # Process optimized parameters
    for param_key, param_values in optimized_data['params'].items():
        # Parse the key format: "('AAPL', 'MovingAverageCrossoverStrategy')"
        try:
            # Strip parentheses and split by comma
            parts = param_key.strip("()").split(", ")
            if len(parts) == 2:
                symbol = parts[0].strip("'\" ")  # Handle any quotes/whitespace
                strategy_name = parts[1].strip("'\" ")
                
                # Map class names to strategy config names
                strategy_name_map = {
                    'MovingAverageCrossoverStrategy': 'moving_average_crossover',
                    'RSIStrategy': 'rsi',
                    'MeanReversionStrategy': 'mean_reversion',
                    'BollingerBandsStrategy': 'bollinger_bands'
                }
                
                # Get the strategy config name
                strategy_config_name = strategy_name_map.get(strategy_name)
                if not strategy_config_name:
                    continue
                    
                # Filter strategies if toggle is set to 'mb_only'
                if strategy_toggle == 'mb_only' and strategy_config_name not in ['mean_reversion', 'bollinger_bands']:
                    continue
                
                # Initialize symbol strategies if needed
                if symbol not in updated_config['strategies']:
                    updated_config['strategies'][symbol] = {}
                
                # Add or update the strategy parameters
                updated_config['strategies'][symbol][strategy_config_name] = param_values
        except Exception as e:
            print(f"Error processing parameter key {param_key}: {e}")
    
    # Set symbols based on what's in the strategies
    updated_config['symbols'] = list(updated_config['strategies'].keys())
    
    print(f"Updated config with optimized parameters, using strategy toggle: {strategy_toggle}")
    return updated_config

# Function to load benchmark data (SPY and QQQ)
def load_benchmark_data():
    benchmark_data = {}
    benchmark_dir = r"C:\Users\jchang427\OneDrive - Georgia Institute of Technology\Random Projects\trading_sim_cursor\src\main\resources\market_data\market_data_export_2020-03-18_to_2022-12-13"
    
    for symbol in ['SPY', 'QQQ']:
        file_path = os.path.join(benchmark_dir, f"{symbol}_data.csv")
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Ensure we have a Date column and it's properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    # Normalize data to 100 at the start for easier comparison
                    df = df.sort_values('Date')
                    df['Normalized'] = df['Close'] / df['Close'].iloc[0] * 100
                    benchmark_data[symbol] = df
                    print(f"Loaded benchmark data for {symbol}")
                else:
                    print(f"No Date column found in {symbol} data")
            else:
                print(f"Benchmark file not found: {file_path}")
        except Exception as e:
            print(f"Error loading benchmark data for {symbol}: {e}")
    
    return benchmark_data

class TradingDashboard:
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.app = dash.Dash(__name__)
        self.benchmark_data = load_benchmark_data()
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                html.H1("Algorithmic Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='mode-selector',
                    options=[
                        {'label': 'Live Trading', 'value': 'live'},
                        {'label': 'Backtest Analysis', 'value': 'backtest'}
                    ],
                    value='backtest',
                    style={'width': '300px', 'margin': '10px'}
                ),
                html.Div([
                    # Add strategy selection toggle
                    html.Div([
                        html.Label("Strategy Selection:"),
                        dcc.RadioItems(
                            id='strategy-toggle',
                            options=[
                                {'label': 'All Strategies', 'value': 'all'},
                                {'label': 'Mean Reversion & Bollinger Bands Only', 'value': 'mb_only'}
                            ],
                            value='all',
                            labelStyle={'display': 'block'}
                        ),
                        html.Button(
                            'Apply Strategy Selection',
                            id='apply-strategy-btn',
                            style={
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin': '10px 0',
                                'borderRadius': '5px'
                            }
                        ),
                        html.Button(
                            'Run Backtest',
                            id='run-backtest-btn',
                            style={
                                'backgroundColor': '#2ecc71',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin': '10px 0',
                                'borderRadius': '5px'
                            }
                        ),
                        html.Div(id='status-display', style={'marginTop': '10px'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '10px'}),
                    
                    dcc.Slider(
                        id='timeline-slider',
                        min=0,
                        max=100,  # Default value, will be updated in callback
                        value=0,
                        marks={},
                        step=1,
                        tooltip={'placement': 'bottom'},
                        disabled=True
                    )
                ], id='timeline-controls', style={'padding': '20px'})
            ], className='banner'),
            
            dcc.Tabs([
                dcc.Tab(label='Performance Overview', children=[
                    html.Div([
                        # Add benchmark display toggles
                        html.Div([
                            html.Label("Benchmark Comparison:"),
                            dcc.Checklist(
                                id='benchmark-toggles',
                                options=[
                                    {'label': 'SPY (S&P 500)', 'value': 'SPY'},
                                    {'label': 'QQQ (NASDAQ-100)', 'value': 'QQQ'}
                                ],
                                value=['SPY'],
                                labelStyle={'display': 'inline-block', 'marginRight': '15px'}
                            )
                        ], style={'marginBottom': '10px'}),
                        dcc.Graph(id='equity-curve'),
                        dcc.Graph(id='drawdown-chart'),
                        html.Div(id='performance-metrics')
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='Positions & Risk', children=[
                    html.Div([
                        html.Div(id='current-positions'),
                        dcc.Graph(id='risk-metrics'),
                        html.Div(id='volatility-display')
                    ], style={'padding': '20px'})
                ]),
                
                dcc.Tab(label='Trading Activity', children=[
                    html.Div([
                        dash_table.DataTable(
                            id='trade-history',
                            columns=[
                                {'name': 'Date', 'id': 'date'},
                                {'name': 'Symbol', 'id': 'symbol'},
                                {'name': 'Action', 'id': 'action'},
                                {'name': 'Amount', 'id': 'amount'},
                                {'name': 'Price', 'id': 'price'},
                                {'name': 'Value', 'id': 'value'}
                            ],
                            style_table={'overflowX': 'auto'},
                            page_size=10
                        )
                    ], style={'padding': '20px'})
                ])
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=1*1000,
                n_intervals=0,
                disabled=False
            ),
            
            dcc.Store(id='backtest-data-store'),
            dcc.Store(id='current-config-store')
        ], style={'fontFamily': 'Arial, sans-serif'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('timeline-slider', 'max'),
             Output('timeline-slider', 'marks'),
             Output('timeline-slider', 'value'),
             Output('timeline-slider', 'disabled'),
             Output('interval-component', 'disabled')],
            [Input('mode-selector', 'value')]
        )
        def update_mode_controls(selected_mode):
            if selected_mode == 'backtest':
                max_days = len(self.engine.backtest_results) if hasattr(self.engine, 'backtest_results') else 0
                if max_days > 0:
                    marks = {i: str(i+1) for i in range(0, max_days, max(1, max_days//10))}
                    return max_days-1, marks, max_days-1, False, True
                else:
                    return 0, {}, 0, True, True
            return 0, {}, 0, True, False

        # Callback for strategy selection toggle
        @self.app.callback(
            [Output('status-display', 'children'),
             Output('current-config-store', 'data')],
            [Input('apply-strategy-btn', 'n_clicks'),
             Input('run-backtest-btn', 'n_clicks')],
            [State('strategy-toggle', 'value'),
             State('status-display', 'children')]
        )
        def handle_button_actions(apply_clicks, run_clicks, strategy_toggle, current_status):
            ctx = callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            if not triggered_id or (not apply_clicks and not run_clicks):
                return "Ready to run backtest", {'strategy_toggle': strategy_toggle}
            
            # Handle Apply Strategy Selection button
            if triggered_id == 'apply-strategy-btn':
                return f"Strategy selection applied: {'All strategies' if strategy_toggle == 'all' else 'Mean Reversion & Bollinger Bands only'}", {'strategy_toggle': strategy_toggle}
            
            # Handle Run Backtest button
            elif triggered_id == 'run-backtest-btn':
                try:
                    # Load optimized parameters
                    optimized_data = load_optimized_params()
                    
                    # Update config with optimized parameters based on toggle
                    updated_config = update_config_with_optimized_params(config, optimized_data, strategy_toggle)
                    
                    # Set max_backtest_days in the config for quicker testing
                    updated_config['max_backtest_days'] = 120  # Limit to 120 days for reasonable testing
                    
                    # Create and run a new trading engine with updated config
                    trading_engine = TradingEngine(updated_config)
                    self.engine = trading_engine  # Update the dashboard's trading engine
                    
                    # Run backtest
                    import asyncio
                    asyncio.run(trading_engine._run_backtest())
                    
                    # Update UI component data
                    self._update_component_data()
                    
                    # Return status message
                    strategies_used = "all strategies" if strategy_toggle == 'all' else "mean reversion and bollinger bands only"
                    return f"Backtest completed using {strategies_used}", {'strategy_toggle': strategy_toggle}
                except Exception as e:
                    print(f"Error in run_backtest_with_strategies: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error running backtest: {str(e)}", {'strategy_toggle': strategy_toggle}
            
            # Default case
            return current_status, {'strategy_toggle': strategy_toggle}

        @self.app.callback(
            [Output('equity-curve', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('performance-metrics', 'children'),
             Output('current-positions', 'children'),
             Output('risk-metrics', 'figure'),
             Output('volatility-display', 'children'),
             Output('trade-history', 'data'),
             Output('backtest-data-store', 'data')],
            [Input('interval-component', 'n_intervals'),
             Input('timeline-slider', 'value'),
             Input('mode-selector', 'value'),
             Input('benchmark-toggles', 'value')]  # Add input for benchmark toggles
        )
        def update_all_components(n, slider_day, mode, selected_benchmarks):
            ctx = callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Initialize backtest data if needed
            backtest_data = None
            if mode == 'backtest' and hasattr(self.engine, 'backtest_data'):
                backtest_data = self.engine.backtest_data
            
            if mode == 'live':
                live_results = self.update_live_mode()
                return *live_results, backtest_data
            else:
                backtest_results = self.update_backtest_mode(slider_day, backtest_data, selected_benchmarks)
                return *backtest_results, backtest_data

    def update_live_mode(self):
        # Replace these with your actual live update logic.
        equity_fig = go.Figure(go.Scatter(
            x=[datetime.datetime.now()], 
            y=[100000],
            mode='lines+markers'
        ))
        equity_fig.update_layout(title='Live Equity Curve')
        
        drawdown_fig = go.Figure()
        drawdown_fig.update_layout(title='Live Drawdown (N/A)')
        
        metrics_html = html.Div("Live performance metrics not available yet.")
        current_positions = html.Div("Live positions not available yet.")
        
        risk_fig = go.Figure()
        risk_fig.update_layout(title='Live Risk Metrics (N/A)')
        
        vol_display = html.Div("Live volatility data not available yet.")
        trade_history = []
        
        return equity_fig, drawdown_fig, metrics_html, current_positions, risk_fig, vol_display, trade_history

    def update_backtest_mode(self, day, backtest_data, selected_benchmarks=None):
        if not backtest_data:
            # Return empty defaults if data is missing
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title='No Backtest Data Available',
                annotations=[{
                    'text': 'Run a backtest first to see results',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return empty_fig, empty_fig, html.Div("No backtest data available"), html.Div(), empty_fig, html.Div(), []
        
        if day >= len(backtest_data['history']):
            day = len(backtest_data['history']) - 1 if backtest_data['history'] else 0
        
        # Get historical state for the selected day.
        if not backtest_data['history']:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title='No Backtest History Available',
                annotations=[{
                    'text': 'No history data available',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return empty_fig, empty_fig, html.Div("No backtest history available"), html.Div(), empty_fig, html.Div(), []
            
        hist_state = backtest_data['history'][day]
        positions = hist_state['positions']
        trades = [t for t in backtest_data['trades'] if t['day'] <= day]
        
        # Equity Curve up to the selected day.
        equity_df = pd.DataFrame(backtest_data['history'][:day+1])
        # Convert string dates to datetime
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Normalize the equity values for percentage comparison
        first_value = equity_df['value'].iloc[0] if not equity_df.empty else 100000
        equity_df['normalized_value'] = equity_df['value'] / first_value * 100
        
        # Create the equity curve figure
        equity_fig = go.Figure()
        
        # Add the portfolio equity curve
        equity_fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['normalized_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add benchmark traces if selected and available
        if selected_benchmarks and self.benchmark_data:
            start_date = equity_df['date'].min() if not equity_df.empty else None
            end_date = equity_df['date'].max() if not equity_df.empty else None
            
            for symbol in selected_benchmarks:
                if symbol in self.benchmark_data:
                    benchmark_df = self.benchmark_data[symbol]
                    # Filter to the backtest date range
                    if start_date and end_date:
                        filtered_df = benchmark_df[(benchmark_df['Date'] >= start_date) & 
                                                  (benchmark_df['Date'] <= end_date)]
                        
                        if not filtered_df.empty:
                            # Normalize to the same starting point
                            first_benchmark_value = filtered_df['Close'].iloc[0]
                            normalized_values = filtered_df['Close'] / first_benchmark_value * 100
                            
                            equity_fig.add_trace(go.Scatter(
                                x=filtered_df['Date'],
                                y=normalized_values,
                                mode='lines',
                                name=symbol,
                                line=dict(
                                    color='#e74c3c' if symbol == 'SPY' else '#2ecc71',
                                    width=1.5,
                                    dash='dot'
                                )
                            ))
        
        equity_fig.update_layout(
            title='Normalized Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Value (Normalized to 100)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode="x unified"
        )
        
        # Compute drawdown: peak, then drawdown percentage.
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['value']) / equity_df['peak'] * 100
        drawdown_fig = go.Figure(go.Scatter(
            x=equity_df['date'],
            y=equity_df['drawdown'],
            line=dict(color='#e74c3c')
        ))
        drawdown_fig.update_layout(title='Drawdown Curve')
        
        # Performance metrics: total return and max drawdown.
        start_value = equity_df['value'].iloc[0]
        end_value = equity_df['value'].iloc[-1]
        total_return = (end_value - start_value) / start_value * 100
        max_drawdown = equity_df['drawdown'].max()
        
        # Calculate additional metrics based on benchmarks
        benchmark_metrics = []
        if selected_benchmarks and self.benchmark_data:
            start_date = equity_df['date'].min() if not equity_df.empty else None
            end_date = equity_df['date'].max() if not equity_df.empty else None
            
            for symbol in selected_benchmarks:
                if symbol in self.benchmark_data:
                    benchmark_df = self.benchmark_data[symbol]
                    # Filter to the backtest date range
                    if start_date and end_date:
                        filtered_df = benchmark_df[(benchmark_df['Date'] >= start_date) & 
                                                  (benchmark_df['Date'] <= end_date)]
                        
                        if not filtered_df.empty:
                            # Calculate benchmark return
                            b_start = filtered_df['Close'].iloc[0]
                            b_end = filtered_df['Close'].iloc[-1]
                            b_return = (b_end - b_start) / b_start * 100
                            
                            benchmark_metrics.append(html.P(f"{symbol} Return: {b_return:.2f}%"))
        
        metrics_html = html.Div([
            html.P(f"Portfolio Total Return: {total_return:.2f}%"),
            html.P(f"Max Drawdown: {max_drawdown:.2f}%"),
            *benchmark_metrics
        ])
        
        # Build positions table.
        position_table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in ['Symbol', 'Amount', 'Price', 'Value']],
            data=[{
                'Symbol': sym,
                'Amount': f"{pos['amount']:.2f}",
                'Price': f"{pos['price']:.2f}",
                'Value': f"{pos['amount'] * pos['price']:.2f}"
            } for sym, pos in positions.items()]
        )
        
        # Calculate risk metrics (example: annualized volatility).
        equity_df['daily_return'] = equity_df['value'].pct_change()
        volatility = equity_df['daily_return'].std() * (252**0.5) * 100 if len(equity_df) > 1 else 0
        risk_fig = go.Figure(go.Bar(
            x=['Volatility'],
            y=[volatility],
            marker_color=['#e74c3c']
        ))
        risk_fig.update_layout(title='Risk Metrics')
        vol_display = html.Div(f"Annualized Volatility: {volatility:.2f}%")
        
        # Build trade history.
        trade_data = [{
            'date': t['date'],
            'symbol': t['symbol'],
            'action': t['action'],
            'amount': f"{t['amount']:.2f}",
            'price': f"{t['price']:.2f}",
            'value': f"{t['value']:.2f}"
        } for t in trades]
        
        return equity_fig, drawdown_fig, metrics_html, position_table, risk_fig, vol_display, trade_data

    def _update_component_data(self):
        """Update the component data after running a backtest"""
        try:
            # Check if engine exists and has backtest data
            if not hasattr(self, 'engine') or not hasattr(self.engine, 'backtest_data'):
                print("No engine or backtest data available")
                return
                
            # Enable the timeline slider and reset its value to the last day
            if 'history' in self.engine.backtest_data and self.engine.backtest_data['history']:
                max_days = len(self.engine.backtest_data['history']) - 1
                
                # This doesn't directly update the UI, but sets up the data
                # The UI will be updated by the interval component or when the user interacts
                # with the slider
                print(f"Updated component data with {max_days+1} days of backtest history")
        except Exception as e:
            print(f"Error in _update_component_data: {e}")
            import traceback
            traceback.print_exc()

    def create_synthetic_dashboard_data(self):
        """Create synthetic data for the dashboard if the backtest fails"""
        import numpy as np
        import pandas as pd
        
        # Create synthetic history data
        dates = pd.date_range(start='2020-01-01', periods=100)
        values = 100000 + np.random.randn(100).cumsum() * 1000
        
        history = []
        for i, (date, value) in enumerate(zip(dates, values)):
            history.append({
                'date': date,
                'value': value,
                'positions': {
                    'AAPL': {'amount': 10 + i % 5, 'price': 150 + np.random.randn() * 10},
                    'MSFT': {'amount': 15 - i % 7, 'price': 250 + np.random.randn() * 15}
                },
                'cash': 50000 - i * 100
            })
        
        # Create synthetic trade data
        trades = []
        for i in range(20):
            trade_date = dates[i * 5]
            symbol = 'AAPL' if i % 2 == 0 else 'MSFT'
            action = 'BUY' if i % 3 != 0 else 'SELL'
            amount = 5 + i % 10
            price = 150 + np.random.randn() * 20 if symbol == 'AAPL' else 250 + np.random.randn() * 30
            
            trades.append({
                'day': i * 5,
                'date': trade_date.strftime('%Y-%m-%d'),
                'timestamp': trade_date.isoformat(),
                'symbol': symbol,
                'action': action,
                'amount': amount,
                'price': price,
                'value': amount * price
            })
        
        return {
            'history': history,
            'trades': trades
        }

if __name__ == "__main__":
    import asyncio
    
    # Initialize trading engine
    engine = TradingEngine(config)
    
    # Run a backtest first to have data available
    async def run_backtest():
        backtester = Backtester(config)
        await backtester.run()
        return backtester.engine
    
    # Run the backtest and get the engine with results
    engine = asyncio.run(run_backtest())
    
    # Create and run dashboard
    dashboard = TradingDashboard(engine)
    
    # You can change the port number below (default is 8050)
    port = 9050  # Change this value to use a different port
    dashboard.app.run_server(debug=True, port=port)