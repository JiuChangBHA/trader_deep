# algo_trading_system.py
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingMode:
    LIVE = "live"
    BACKTEST = "backtest"

class DataHandler(ABC):
    @abstractmethod
    async def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_latest_data(self, symbol: str) -> pd.DataFrame:
        pass

class AlpacaDataHandler(DataHandler):
    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)
        
    async def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=lookback
        )
        bars = await self.client.get_stock_bars(request)
        return bars.df

    async def get_latest_data(self, symbol: str) -> pd.DataFrame:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = await self.client.get_stock_latest_quote(request)
        return pd.DataFrame([quote.dict()])

class BacktestDataHandler(DataHandler):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()
        self.current_idx = 0
        
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        try:
            # Check if the data path exists
            if not os.path.exists(self.data_path):
                logger.warning(f"Data path {self.data_path} does not exist.")
                return data

            # Assuming data is stored in CSV files named by symbol
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            logger.info(f"Found {len(csv_files)} CSV files in {self.data_path}.")
            if not csv_files:
                logger.warning(f"No CSV files found in {self.data_path}.")
                return data
                
            for file in csv_files:
                if file.endswith('.csv'):
                    symbol = file.split('_')[0]
                    df = pd.read_csv(os.path.join(self.data_path, file))
                    
                    # Try to find a date column (could be 'date', 'Date', 'timestamp', etc.)
                    date_columns = [col for col in df.columns if col.lower() in ['date', 'time', 'timestamp']]
                    
                    if date_columns:
                        # Use the first date column found
                        date_col = date_columns[0]
                        df[date_col] = pd.to_datetime(df[date_col])
                        df.set_index(date_col, inplace=True)
                    else:
                        # If no date column found, create a synthetic one
                        df['date'] = pd.date_range(start='2020-01-01', periods=len(df))
                        df.set_index('date', inplace=True)
                    
                    # Ensure we have the required columns for the trading system
                    if 'Close' not in df.columns and 'close' in df.columns:
                        df['Close'] = df['close']
                    elif 'Close' not in df.columns:
                        logger.warning(f"No Close column found in {file}. Adding synthetic Close data.")
                        df['Close'] = np.random.randn(len(df)).cumsum() + 100
                    
                    data[symbol] = df
            # If we didn't find any data for the symbols we need, create synthetic data
            required_symbols = ['AAPL', 'MSFT']  # These should match the symbols in config
            missing_symbols = [sym for sym in required_symbols if sym not in data.keys()]
            
            if missing_symbols:
                logger.error(f"Missing data for symbols: {missing_symbols}.")
            
            return data
        except Exception as e:
            logger.error(f"Error loading backtest data: {e}")
            return data
    
    async def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame:
        if symbol not in self.data:
            return pd.DataFrame()
        
        end_idx = self.current_idx + 1
        start_idx = max(0, end_idx - lookback)
        
        df = self.data[symbol].iloc[start_idx:end_idx].copy()
        return df
    
    async def get_latest_data(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.data:
            return pd.DataFrame()
        
        latest = self.data[symbol].iloc[self.current_idx:self.current_idx+1].copy()
        return latest

class TradingStrategy(ABC):
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params

    @abstractmethod
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        pass

class MeanReversionStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        closes = data['Close'].values
        lookback = self.params.get('lookback', 20)
        threshold = self.params.get('threshold', 2.0)
        
        if len(closes) < lookback:
            return {'action': 'HOLD', 'strength': 0}
        
        moving_avg = np.mean(closes[-lookback:])
        std_dev = np.std(closes[-lookback:])
        current_price = closes[-1]
        
        z_score = (current_price - moving_avg) / std_dev
        if z_score < -threshold:
            return {'action': 'BUY', 'strength': abs(z_score)}
        elif z_score > threshold:
            return {'action': 'SELL', 'strength': abs(z_score)}
        return {'action': 'HOLD', 'strength': 0}

class BollingerBandsStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        closes = data['Close'].values
        lookback = self.params.get('lookback', 20)
        num_std = self.params.get('num_std', 2)
        
        if len(closes) < lookback:
            return {'action': 'HOLD', 'strength': 0}
        
        moving_avg = np.mean(closes[-lookback:])
        std_dev = np.std(closes[-lookback:])
        upper_band = moving_avg + num_std * std_dev
        lower_band = moving_avg - num_std * std_dev
        current_price = closes[-1]
        
        if current_price > upper_band:
            return {'action': 'SELL', 'strength': (current_price - upper_band)/std_dev}
        elif current_price < lower_band:
            return {'action': 'BUY', 'strength': (lower_band - current_price)/std_dev}
        return {'action': 'HOLD', 'strength': 0}

class StrategyFactory:
    @staticmethod
    def create_strategy(name: str, symbol: str, params: dict) -> TradingStrategy:
        strategies = {
            'mean_reversion': MeanReversionStrategy,
            'bollinger_bands': BollingerBandsStrategy
        }
        return strategies[name](symbol, params)

class RiskManager:
    def __init__(self, initial_equity: float = 100000):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.max_drawdown = 0.2
        self.position_sizes = {}
        self.volatility_cache = {}

    def calculate_position_size(self, symbol: str, price: float) -> float:
        max_risk = self.current_equity * 0.01  # 1% risk per trade
        return max_risk / price

    def get_volatility(self, symbol: str) -> float:
        return self.volatility_cache.get(symbol, 0.01)

    def update_volatility(self, symbol: str, volatility: float):
        self.volatility_cache[symbol] = volatility

class SignalAggregator:
    def __init__(self, strategies: Dict[str, List[TradingStrategy]], risk_manager: RiskManager):
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.signal_history = {}

    def process_signals(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        if symbol not in self.strategies:
            return None

        raw_signals = []
        for strategy in self.strategies[symbol]:
            try:
                signal = strategy.calculate_signal(data)
                normalized = self._normalize_signal(signal)
                weighted = self._apply_strategy_weights(strategy, normalized)
                raw_signals.append(weighted)
            except Exception as e:
                logger.error(f"Signal error in {strategy.__class__.__name__}: {str(e)}")

        if not raw_signals:
            return None

        combined = self._aggregate_signals(raw_signals)
        risk_adjusted = self._apply_risk_constraints(symbol, combined)
        return risk_adjusted

    def _normalize_signal(self, signal: dict) -> float:
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        return action_map[signal['action']] * signal['strength']

    def _apply_strategy_weights(self, strategy: TradingStrategy, signal: float) -> float:
        return signal * strategy.params.get('weight', 1.0)

    def _aggregate_signals(self, signals: List[float]) -> float:
        decay_factor = 0.95
        weights = [decay_factor ** i for i in reversed(range(len(signals)))]
        return np.dot(signals, weights) / sum(weights)

    def _apply_risk_constraints(self, symbol: str, signal: float) -> float:
        volatility = self.risk_manager.get_volatility(symbol)
        scaled = signal / max(volatility, 0.01)
        return np.tanh(scaled)  # Squash to [-1, 1]

class Portfolio:
    def __init__(self, initial_cash: float = 100000):
        self.positions: Dict[str, float] = {}
        self.cash = initial_cash
        self.history = [{
            'date': pd.Timestamp.now(),
            'value': initial_cash
        }]
        self.current_prices = {}
        self.trade_history = []
        self.historical_states = []

    def update_prices(self, prices: Dict[str, float]):
        self.current_prices = prices
        # Update history with current portfolio value
        self.history.append({
            'date': pd.Timestamp.now(),
            'value': self.value()
        })

    def value(self) -> float:
        pos_value = sum(amt * self.current_prices.get(sym, 0) 
                     for sym, amt in self.positions.items())
        return self.cash + pos_value
    
    def _update_portfolio(self, symbol: str, amount: float, price: float):
        self.trade_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY' if amount > 0 else 'SELL',
            'amount': abs(amount),
            'price': price,
            'value': abs(amount) * price
        })

    def snapshot_state(self, current_date):
        self.historical_states.append({
            'date': current_date,
            'value': self.value(),
            'positions': {sym: {'amount': amt, 'price': self.current_prices.get(sym, 0)}
                          for sym, amt in self.positions.items()},
            'cash': self.cash
        })

class TradingEngine:
    def __init__(self, config: dict):
        self.mode = config['mode']
        self.symbols = config['symbols']
        self.strategies = self._init_strategies(config['strategies'])
        self.risk_manager = RiskManager(config.get('initial_equity', 100000))
        self.portfolio = Portfolio(config.get('initial_equity', 100000))
        self.signal_aggregator = SignalAggregator(self.strategies, self.risk_manager)
        
        if self.mode == TradingMode.LIVE:
            self.data_handler = AlpacaDataHandler(config['api_key'], config['secret_key'])
            self.trading_client = TradingClient(config['api_key'], config['secret_key'])
        else:
            self.data_handler = BacktestDataHandler(config['data_path'])
            self.backtest_results = []

    def _init_strategies(self, strategy_config: dict) -> Dict[str, List[TradingStrategy]]:
        strategies = {}
        for symbol, configs in strategy_config.items():
            strategies[symbol] = [
                StrategyFactory.create_strategy(name, symbol, params)
                for name, params in configs.items()
            ]
        return strategies

    async def run(self):
        if self.mode == TradingMode.LIVE:
            while True:
                await self._trading_cycle()
                await asyncio.sleep(300)  # 5 minute intervals
        else:
            await self._run_backtest()

    async def _trading_cycle(self):
        try:
            prices = {}
            for symbol in self.symbols:
                data = await self.data_handler.get_historical_data(symbol, 30)
                latest_data = await self.data_handler.get_latest_data(symbol)
                current_price = latest_data['Close'].iloc[0]
                prices[symbol] = current_price
                
                signal = self.signal_aggregator.process_signals(symbol, data)
                if signal is not None:
                    await self._execute_trade(symbol, signal, current_price)
            
            self.portfolio.update_prices(prices)
            await self._update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    async def _execute_trade(self, symbol: str, signal: float, price: float):
        position_size = self.risk_manager.calculate_position_size(symbol, price)
        amount = signal * position_size
        
        if self.mode == TradingMode.LIVE:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=abs(amount),
                side=OrderSide.BUY if amount > 0 else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            await self.trading_client.submit_order(order)
        else:
            self._update_portfolio(symbol, amount, price)

    def _update_portfolio(self, symbol: str, amount: float, price: float):
        current_amount = self.portfolio.positions.get(symbol, 0)
        cost = amount * price
        if self.portfolio.cash >= cost:
            self.portfolio.positions[symbol] = current_amount + amount
            self.portfolio.cash -= cost
            self.portfolio._update_portfolio(symbol, amount, price)
            logger.info(f"Executed {amount:.2f} shares of {symbol} @ {price:.2f}")

    async def _run_backtest(self):
        handler = self.data_handler
        
        # Make sure we have data for all symbols
        for symbol in self.symbols:
            if symbol not in handler.data or handler.data[symbol].empty:
                logger.error(f"No data available for symbol {symbol}. Cannot run backtest.")
                return
        
        max_days = min(len(handler.data[sym]) for sym in self.symbols)
        
        # Get dates from the first symbol's data
        first_symbol = self.symbols[0]
        self.backtest_dates = handler.data[first_symbol].index[:max_days].tolist()

        # Clear initial history entry (will be replaced with backtest dates)
        self.portfolio.history = []
        self.portfolio.historical_states = []
        self.portfolio.trade_history = []
        
        for day in range(max_days):
            handler.current_idx = day
            await self._trading_cycle()
            
            # Get the current date from the data
            current_date = self.backtest_dates[day]
            self.portfolio.snapshot_state(current_date)
            
            # Update portfolio history with the current date and value
            self.portfolio.history.append({
                'date': current_date,
                'value': self.portfolio.value()
            })
            
            self.backtest_results.append(self.portfolio.value())
            logger.info(f"Day {day+1}: Portfolio Value {self.backtest_results[-1]:.2f}")
        
        # Prepare data for the dashboard
        self.backtest_data = {
            'history': self.portfolio.historical_states,
            'trades': [{'day': i, 'date': self.backtest_dates[min(i, len(self.backtest_dates)-1)].strftime('%Y-%m-%d'), **t} 
                      for i, t in enumerate(self.portfolio.trade_history)]
        }

    async def _update_risk_metrics(self):
        # Update volatility metrics
        for symbol in self.symbols:
            if symbol in self.portfolio.current_prices:
                price_history = await self.data_handler.get_historical_data(symbol, 30)
                returns = np.diff(np.log(price_history['Close']))
                volatility = np.std(returns) * np.sqrt(252)
                self.risk_manager.update_volatility(symbol, volatility)

class Backtester:
    def __init__(self, config: dict):
        self.engine = TradingEngine(config)
        self.results = {}

    async def run(self):
        await self.engine.run()
        self._generate_performance_report()
        return self.engine

    def _generate_performance_report(self):
        if not hasattr(self.engine, 'backtest_results') or not self.engine.backtest_results:
            logger.warning("No backtest results available. Skipping performance report.")
            return
            
        returns = np.diff(self.engine.backtest_results) if len(self.engine.backtest_results) > 1 else [0]
        sharpe = np.mean(returns) / (np.std(returns) or 1) * np.sqrt(252)
        max_dd = self._calculate_max_drawdown()
        
        print(f"\nBacktest Results:")
        print(f"Final Portfolio Value: {self.engine.backtest_results[-1]:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")

    def _calculate_max_drawdown(self) -> float:
        if not hasattr(self.engine, 'backtest_results') or not self.engine.backtest_results:
            return 0.0
            
        peak = self.engine.backtest_results[0]
        max_dd = 0
        for val in self.engine.backtest_results:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

# Define config at module level so it can be imported
config = {
    'mode': TradingMode.BACKTEST,
    'api_key': 'AKGJL44ZUAB56704LFVX',
    'secret_key': 'mPWPeY6T2TfyiAv7ebrFDejhFa0yhH1Xlj57v0lb',
    'data_path': 'C:/Users/jchang427/OneDrive - Georgia Institute of Technology/Random Projects/trading_sim_cursor/src/main/resources/market_data/2025-03-10-07-28_market_data_export_2020-03-11_to_2025-03-10',  # Simplified path that will use synthetic data
    'initial_equity': 100000,
    'symbols': ['AAPL', 'MSFT'],
    'strategies': {
        'AAPL': {
            'mean_reversion': {'lookback': 20, 'threshold': 2.0, 'weight': 0.6},
            'bollinger_bands': {'lookback': 20, 'num_std': 2, 'weight': 0.4}
        },
        'MSFT': {
            'mean_reversion': {'lookback': 30, 'threshold': 1.8, 'weight': 0.5},
            'bollinger_bands': {'lookback': 25, 'num_std': 1.5, 'weight': 0.5}
        }
    }
}

if __name__ == "__main__":
    backtester = Backtester(config)
    asyncio.run(backtester.run())