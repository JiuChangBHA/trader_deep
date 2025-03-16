# dashboard.py
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
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
                       style={'textAlign': 'center', 'color': '#2c3e50'})
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
                                {'name': 'Timestamp', 'id': 'timestamp'},
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
                interval=1*1000,  # Update every second
                n_intervals=0
            )
        ], style={'fontFamily': 'Arial, sans-serif'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('equity-curve', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('performance-metrics', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance(n):
            try:
                # Equity Curve
                history = self.engine.portfolio.history
                
                # Handle empty history
                if not history:
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title='No Data Available',
                        annotations=[{
                            'text': 'No trading data available yet',
                            'showarrow': False,
                            'font': {'size': 20}
                        }]
                    )
                    return empty_fig, empty_fig, html.Div("No performance metrics available")
                
                df = pd.DataFrame(history)
                
                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=df['date'], y=df['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#3498db', width=2)
                ))
                equity_fig.update_layout(
                    title='Portfolio Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_white'
                )
                
                # Drawdown Chart
                df['peak'] = df['value'].cummax()
                df['drawdown'] = (df['value'] - df['peak']) / df['peak']
                
                drawdown_fig = go.Figure()
                drawdown_fig.add_trace(go.Scatter(
                    x=df['date'], y=df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#e74c3c', width=2),
                    fill='tozeroy'
                ))
                drawdown_fig.update_layout(
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown',
                    template='plotly_white'
                )
                
                # Performance Metrics
                if len(df) > 1:
                    returns = df['value'].pct_change().dropna()
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
                    max_dd = df['drawdown'].min() * 100
                    total_return = ((df['value'].iloc[-1] / df['value'].iloc[0]) - 1) * 100
                else:
                    sharpe = 0
                    max_dd = 0
                    total_return = 0
                
                metrics = html.Div([
                    html.Div([
                        html.H4("Performance Metrics"),
                        html.Table([
                            html.Tr([html.Td("Total Return:"), html.Td(f"{total_return:.2f}%")]),
                            html.Tr([html.Td("Sharpe Ratio:"), html.Td(f"{sharpe:.2f}")]),
                            html.Tr([html.Td("Max Drawdown:"), html.Td(f"{max_dd:.2f}%")])
                        ], style={'width': '100%', 'border': '1px solid #ddd'})
                    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px'})
                ])
                
                return equity_fig, drawdown_fig, metrics
            except Exception as e:
                print(f"Error updating performance: {e}")
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title='Error Loading Data',
                    annotations=[{
                        'text': f'Error: {str(e)}',
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                )
                return empty_fig, empty_fig, html.Div(f"Error: {str(e)}")

        @self.app.callback(
            [Output('current-positions', 'children'),
             Output('risk-metrics', 'figure'),
             Output('volatility-display', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_positions_and_risk(n):
            # Current Positions
            positions = self.engine.portfolio.positions
            prices = self.engine.portfolio.current_prices
            position_data = []
            
            for sym, amt in positions.items():
                price = prices.get(sym, 0)
                value = amt * price
                position_data.append({
                    'Symbol': sym,
                    'Amount': f"{amt:.2f}",
                    'Price': f"{price:.2f}",
                    'Value': f"{value:.2f}"
                })
            
            positions_table = dash_table.DataTable(
                columns=[{'name': col, 'id': col} for col in ['Symbol', 'Amount', 'Price', 'Value']],
                data=position_data,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            )
            
            # Risk Metrics
            volatilities = self.engine.risk_manager.volatility_cache
            risk_fig = go.Figure()
            if volatilities:
                risk_fig.add_trace(go.Bar(
                    x=list(volatilities.keys()),
                    y=list(volatilities.values()),
                    name='Annualized Volatility',
                    marker_color='#2ecc71'
                ))
                risk_fig.update_layout(
                    title='Asset Volatility',
                    xaxis_title='Symbol',
                    yaxis_title='Volatility',
                    template='plotly_white'
                )
            
            # Volatility Display
            vol_display = html.Div([
                html.H4("Latest Volatility Updates"),
                html.Ul([
                    html.Li(f"{sym}: {vol:.2%}") 
                    for sym, vol in volatilities.items()
                ])
            ])
            
            return positions_table, risk_fig, vol_display

        @self.app.callback(
            Output('trade-history', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trade_history(n):
            trades = self.engine.portfolio.trade_history
            return trades

if __name__ == "__main__":
    # Example usage
    from algo_trading_system import TradingEngine, config
    
    # Initialize trading engine
    engine = TradingEngine(config)
    
    # Create and run dashboard
    dashboard = TradingDashboard(engine)
    dashboard.app.run_server(debug=True, port=8050)