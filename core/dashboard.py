# dashboard.py
import datetime
import dash
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class TradingDashboard:
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.app = dash.Dash(__name__)
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
            
            dcc.Store(id='backtest-data-store')
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
             Input('mode-selector', 'value')]
        )
        def update_all_components(n, slider_day, mode):
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
                backtest_results = self.update_backtest_mode(slider_day, backtest_data)
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

    def update_backtest_mode(self, day, backtest_data):
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
        equity_fig = go.Figure(go.Scatter(
            x=equity_df['date'],
            y=equity_df['value'],
            line=dict(color='#3498db')
        ))
        equity_fig.update_layout(title='Historical Equity Curve')
        
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
        metrics_html = html.Div([
            html.P(f"Total Return: {total_return:.2f}%"),
            html.P(f"Max Drawdown: {max_drawdown:.2f}%")
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
    from algo_trading_system import TradingEngine, config, Backtester
    
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
    dashboard.app.run_server(debug=True)