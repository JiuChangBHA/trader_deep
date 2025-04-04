# algo_trading_system.py
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, ForwardRef
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Import in stages to avoid circular dependencies
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

    def precompute_indicators(self, symbol: str, indicators: dict):
        """Precompute technical indicators for a symbol's data"""
        if symbol not in self.data:
            return
            
        df = self.data[symbol].copy()
        
        # Moving Averages
        for window in indicators.get('moving_averages', []):
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # RSI
        for period in indicators.get('rsi_periods', []):
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in indicators.get('bollinger_windows', []):
            ma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            df[f'BB_Upper_{window}'] = ma + 2*std
            df[f'BB_Lower_{window}'] = ma - 2*std
            
        self.data[symbol] = df

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        try:
            # Check if the data path exists
            if not os.path.exists(self.data_path):
                logger.error(f"Data path {self.data_path} does not exist.")
                return self._generate_synthetic_data()

            # Assuming data is stored in CSV files named by symbol
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            if not csv_files:
                logger.error(f"No CSV files found in {self.data_path}.")
                return self._generate_synthetic_data()
            logger.info(f"Found {len(csv_files)} CSV files in {self.data_path}.")
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

    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market data for testing"""
        logger.warning("Generating synthetic data for backtest")
        data = {}
        symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
        start_date = pd.Timestamp('2020-01-01')
        periods = 1000
        
        for symbol in symbols:
            # Generate random price series with realistic properties
            np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
            
            # Start with a reasonable base price
            base_price = np.random.uniform(50, 200)
            
            # Generate daily returns with slight upward bias and volatility
            daily_returns = np.random.normal(0.0005, 0.015, periods)
            
            # Calculate price series
            price_series = base_price * np.cumprod(1 + daily_returns)
            
            # Create dataframe with OHLCV data
            dates = [start_date + pd.Timedelta(days=i) for i in range(periods)]
            df = pd.DataFrame({
                'Open': price_series * np.random.uniform(0.99, 1.0, periods),
                'High': price_series * np.random.uniform(1.0, 1.03, periods),
                'Low': price_series * np.random.uniform(0.97, 0.99, periods),
                'Close': price_series,
                'Volume': np.random.randint(1000000, 10000000, periods)
            }, index=dates)
            
            data[symbol] = df
            
        return data

class TradingStrategy(ABC):
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params

    @abstractmethod
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        pass

class MovingAverageCrossoverStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        short_window = self.params.get('short_window', 10)
        long_window = self.params.get('long_window', 50)
        
        if len(data) < long_window:
            return {'action': 'HOLD', 'strength': 0}
        
        # Create a proper copy of the DataFrame to avoid SettingWithCopyWarning
        data_copy = data.copy()
        
        # Check if required MA columns exist, calculate them if they don't
        short_ma_col = f'MA_{short_window}'
        long_ma_col = f'MA_{long_window}'
        
        if short_ma_col not in data_copy.columns:
            data_copy[short_ma_col] = data_copy['Close'].rolling(window=short_window, min_periods=1).mean()
            
        if long_ma_col not in data_copy.columns:
            data_copy[long_ma_col] = data_copy['Close'].rolling(window=long_window, min_periods=1).mean()
        
        data_copy['Short_MA'] = data_copy[short_ma_col]
        data_copy['Long_MA'] = data_copy[long_ma_col]

        # Ensure we have at least 2 values for comparison
        if len(data_copy) < 2:
            return {'action': 'HOLD', 'strength': 0}

        if data_copy['Short_MA'].iloc[-1] > data_copy['Long_MA'].iloc[-1] and data_copy['Short_MA'].iloc[-2] <= data_copy['Long_MA'].iloc[-2]:
            return {'action': 'BUY', 'strength': 1.0}
        elif data_copy['Short_MA'].iloc[-1] < data_copy['Long_MA'].iloc[-1] and data_copy['Short_MA'].iloc[-2] >= data_copy['Long_MA'].iloc[-2]:
            return {'action': 'SELL', 'strength': 1.0}
        return {'action': 'HOLD', 'strength': 0}

class RSIStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        period = self.params.get('period', 14)
        overbought = self.params.get('overbought', 70)
        oversold = self.params.get('oversold', 30)
        
        if len(data) < period:
            return {'action': 'HOLD', 'strength': 0}
        
        # Create a proper copy of the DataFrame to avoid SettingWithCopyWarning
        data_copy = data.copy()
        
        # Check if RSI column exists, calculate it if it doesn't
        rsi_col = f'RSI_{period}'
        if rsi_col not in data_copy.columns:
            # Calculate RSI
            delta = data_copy['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-9)  # Add small value to avoid division by zero
            data_copy[rsi_col] = 100 - (100 / (1 + rs))
        
        rsi = data_copy[rsi_col].iloc[-1]
        
        if rsi < oversold:
            return {'action': 'BUY', 'strength': (oversold - rsi) / oversold}
        elif rsi > overbought:
            return {'action': 'SELL', 'strength': (rsi - overbought) / (100 - overbought)}
        return {'action': 'HOLD', 'strength': 0}
    
class MeanReversionStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        lookback = self.params.get('lookback', 20)
        threshold = self.params.get('threshold', 2.0)
        
        if len(data) < lookback:
            return {'action': 'HOLD', 'strength': 0}
        
        # Create a proper copy of the DataFrame
        data_copy = data.copy()
        
        # Calculate mean and standard deviation directly from price data
        # This avoids the need for precomputed indicators
        closes = data_copy['Close'].values
        current_price = closes[-1]
        
        # Use the moving_avg and std_dev columns if they exist
        ma_col = f'MA_{lookback}'
        std_col = f'Std_{lookback}'
        
        if ma_col in data_copy.columns and std_col in data_copy.columns:
            # Use precomputed values
            moving_avg = data_copy[ma_col].iloc[-1]
            std_dev = data_copy[std_col].iloc[-1]
        else:
            # Calculate directly
            moving_avg = np.mean(closes[-lookback:])
            std_dev = np.std(closes[-lookback:])
        
        # Avoid division by zero
        if std_dev < 1e-9:
            return {'action': 'HOLD', 'strength': 0}
        
        z_score = (current_price - moving_avg) / std_dev
        if z_score < -threshold:
            return {'action': 'BUY', 'strength': abs(z_score)}
        elif z_score > threshold:
            return {'action': 'SELL', 'strength': abs(z_score)}
        return {'action': 'HOLD', 'strength': 0}

class BollingerBandsStrategy(TradingStrategy):
    def calculate_signal(self, data: pd.DataFrame) -> dict:
        lookback = self.params.get('lookback', 20)
        num_std = self.params.get('num_std', 2)
        
        if len(data) < lookback:
            return {'action': 'HOLD', 'strength': 0}
        
        # Create a proper copy of the DataFrame
        data_copy = data.copy()
        closes = data_copy['Close'].values
        current_price = closes[-1]
        
        # Use the moving_avg and std_dev columns if they exist
        ma_col = f'MA_{lookback}'
        std_col = f'Std_{lookback}'
        
        if ma_col in data_copy.columns and std_col in data_copy.columns:
            # Use precomputed values
            moving_avg = data_copy[ma_col].iloc[-1]
            std_dev = data_copy[std_col].iloc[-1]
        else:
            # Calculate directly
            moving_avg = np.mean(closes[-lookback:])
            std_dev = np.std(closes[-lookback:])
        
        # Avoid issues with very low volatility
        if std_dev < 1e-9:
            return {'action': 'HOLD', 'strength': 0}
            
        upper_band = moving_avg + num_std * std_dev
        lower_band = moving_avg - num_std * std_dev
        
        if current_price > upper_band:
            return {'action': 'SELL', 'strength': (current_price - upper_band)/std_dev}
        elif current_price < lower_band:
            return {'action': 'BUY', 'strength': (lower_band - current_price)/std_dev}
        return {'action': 'HOLD', 'strength': 0}

class StrategyFactory:
    strategies = {
        'moving_average_crossover': MovingAverageCrossoverStrategy,
        'rsi': RSIStrategy,
        'mean_reversion': MeanReversionStrategy,
        'bollinger_bands': BollingerBandsStrategy
    }

    @staticmethod
    def create_strategy(name: str, symbol: str, params: dict) -> TradingStrategy:
        return StrategyFactory.strategies[name](symbol, params)

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
        total_weight = sum(strategy.params.get('weight', 1.0) for strategy in self.strategies[symbol])

        for strategy in self.strategies[symbol]:
            try:
                signal = strategy.calculate_signal(data)
                normalized = self._normalize_signal(signal)
                weighted = self._apply_strategy_weights(strategy, normalized, total_weight)
                raw_signals.append(weighted)
            except KeyError as e:
                # Log the specific missing key and strategy type
                logger.error(f"Signal error in {strategy.__class__.__name__}: Missing key '{str(e)}' for symbol {symbol}")
                # Continue with other strategies
            except Exception as e:
                # Log the general error with traceback
                logger.error(f"Signal error in {strategy.__class__.__name__} for {symbol}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                # Continue with other strategies

        if not raw_signals:
            return None

        combined = self._aggregate_signals(raw_signals)
        risk_adjusted = self._apply_risk_constraints(symbol, combined)
        return risk_adjusted

    def _normalize_signal(self, signal: dict) -> float:
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        return action_map[signal['action']] * signal['strength']

    def _apply_strategy_weights(self, strategy: TradingStrategy, signal: float, total_weight: float) -> float:
        return signal * strategy.params.get('weight', 1.0) / total_weight

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
            'date': pd.Timestamp.now().isoformat(),
            'value': self.value()
        })

    def value(self) -> float:
        pos_value = sum(amt * self.current_prices.get(sym, 0) 
                     for sym, amt in self.positions.items())
        return self.cash + pos_value
    
    def _update_portfolio(self, symbol: str, amount: float, price: float, current_date: pd.Timestamp = None, day: int = None):
        trade_entry = {
            'timestamp': current_date.isoformat() if current_date else pd.Timestamp.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY' if amount > 0 else 'SELL',
            'amount': abs(amount),
            'price': price,
            'value': abs(amount) * price,
            'day': day  # Track the backtest day
        }
        self.trade_history.append(trade_entry)

    def snapshot_state(self, current_date):
        self.historical_states.append({
            'date': current_date.isoformat(),
            'value': self.value(),
            'positions': {sym: {'amount': amt, 'price': self.current_prices.get(sym, 0)}
                          for sym, amt in self.positions.items()},
            'cash': self.cash
        })

class TradingEngine:
    def __init__(self, config: dict):
        self.mode = config['mode']
        if self.mode == TradingMode.LIVE:
            self.data_handler = AlpacaDataHandler(config['api_key'], config['secret_key'])
            self.trading_client = TradingClient(config['api_key'], config['secret_key'])
        else:
            self.data_handler = BacktestDataHandler(config['data_path'])
            self.backtest_results = []
        self.symbols = list(self.data_handler.data.keys())
        self.strategies = self._init_strategies(StrategyFactory.strategies)
        self.risk_manager = RiskManager(config.get('initial_equity', 100000))
        self.portfolio = Portfolio(config.get('initial_equity', 100000))
        self.signal_aggregator = SignalAggregator(self.strategies, self.risk_manager)
        self.current_date = None
        self.current_day = 0
        self.mvo_enabled = config.get('mvo_enabled', True)
        self.rebalance_frequency = config.get('rebalance_frequency', 30)
        self.mvo_lookback = config.get('mvo_lookback', 252)
        self._precompute_indicators() 

        # Walk-forward optimization parameters
        wf_config = config.get('walk_forward', {})
        self.optimization_window_days = wf_config.get('optimization_window_days', 180)
        self.out_of_sample_window_days = wf_config.get('out_of_sample_window_days', 30)
        self.last_optimization_day = -self.optimization_window_days

    def _precompute_indicators(self):
        """Precompute indicators based on optimized parameters from JSON"""
        try:
            # Load optimized parameters if file exists
            optimized_params = {}
            try:
                if os.path.exists("optimized_params.json"):
                    with open("optimized_params.json") as f:
                        optimized_params = json.load(f)["params"]
                    logger.info("Loaded cached params from optimized_params.json")
            except Exception as e:
                logger.warning(f"Could not load optimized parameters: {e}")
                
            indicator_config = defaultdict(lambda: {
                'moving_averages': set(),
                'rsi_periods': set(),
                'bollinger_windows': set(),
                'meanreversion_windows': set()
            })

            # Parse parameters from JSON structure
            for strategy_key, params in optimized_params.items():
                try:
                    # Convert string key to tuple ("('AAPL', 'Strategy')" -> ("AAPL", "Strategy"))
                    symbol, strategy_name = map(str.strip, strategy_key.strip("()").split(",", 1))
                    symbol = symbol.strip("'")
                    strategy_name = strategy_name.strip(" '")  # Handle quotes in strategy names

                    # Collect parameters for each strategy type
                    if "MovingAverageCrossover" in strategy_name:
                        indicator_config[symbol]['moving_averages'].add(params['short_window'])
                        indicator_config[symbol]['moving_averages'].add(params['long_window'])
                    elif "RSIStrategy" in strategy_name:
                        indicator_config[symbol]['rsi_periods'].add(params['period'])
                    elif "BollingerBandsStrategy" in strategy_name:
                        indicator_config[symbol]['bollinger_windows'].add(params['lookback'])
                    elif "MeanReversionStrategy" in strategy_name:
                        indicator_config[symbol]['meanreversion_windows'].add(params['lookback'])
                except Exception as e:
                    logger.error(f"Error processing parameter key {strategy_key}: {e}")

            # Add default indicators if none found
            for symbol in self.symbols:
                # Add default moving averages if none found
                if not indicator_config[symbol]['moving_averages']:
                    indicator_config[symbol]['moving_averages'].update([10, 20, 50])
                
                # Always include RSI with period 14
                indicator_config[symbol]['rsi_periods'].add(14)
                
                # Add default bollinger and meanreversion windows if none found
                if not indicator_config[symbol]['bollinger_windows']:
                    indicator_config[symbol]['bollinger_windows'].add(20)
                if not indicator_config[symbol]['meanreversion_windows']:
                    indicator_config[symbol]['meanreversion_windows'].add(20)

            # Precompute indicators for each symbol
            for symbol in self.symbols:
                if symbol not in self.data_handler.data:
                    logger.warning(f"No data for symbol {symbol}, skipping indicator calculation")
                    continue
                    
                config = indicator_config[symbol]
                
                # Get all unique windows needed for MA calculations
                ma_windows = set()
                ma_windows.update(config['moving_averages'])
                ma_windows.update(config['bollinger_windows'])
                ma_windows.update(config['meanreversion_windows'])
                
                # Precompute indicators
                df = self.data_handler.data[symbol].copy()
                
                # Moving Averages
                for window in ma_windows:
                    df[f'MA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
                
                # Standard Deviations (for Bollinger Bands and Mean Reversion)
                for window in config['bollinger_windows'].union(config['meanreversion_windows']):
                    df[f'Std_{window}'] = df['Close'].rolling(window=window, min_periods=1).std()
                
                # RSI
                for period in config['rsi_periods']:
                    # Handle division by zero and ensure RSI always has values
                    delta = df['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    # Use min_periods=1 to start calculating as soon as possible
                    avg_gain = gain.rolling(window=period, min_periods=1).mean()
                    avg_loss = loss.rolling(window=period, min_periods=1).mean()
                    # Add a small value to avg_loss to avoid division by zero
                    rs = avg_gain / (avg_loss + 1e-9)  # Add small epsilon to avoid division by zero
                    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                
                # Update data with precomputed indicators
                self.data_handler.data[symbol] = df
                
            logger.info("Successfully precomputed indicators for all symbols")
                
        except Exception as e:
            logger.error(f"Error precomputing indicators: {e}")
            import traceback
            traceback.print_exc()
            # Create basic indicators if optimization fails
            self._create_default_indicators()

    def _init_strategies(self, strategy_config: dict) -> Dict[str, List[TradingStrategy]]:
        strategies = {}
        
        # Handle empty strategy config
        if not strategy_config or isinstance(strategy_config, dict) and not strategy_config:
            # Initialize with default strategies
            for symbol in self.symbols:
                strategies[symbol] = [
                    StrategyFactory.create_strategy(name, symbol, {})
                    for name in StrategyFactory.strategies
                ]
            return strategies
            
        # Handle case where strategy_config is the StrategyFactory.strategies dictionary
        if strategy_config == StrategyFactory.strategies:
            for symbol in self.symbols:
                strategies[symbol] = [
                    StrategyFactory.create_strategy(name, symbol, {})
                    for name in strategy_config
                ]
            return strategies
            
        # Handle normal case where strategy_config is a nested dictionary
        for symbol, symbol_strategies in strategy_config.items():
            if symbol not in self.symbols:
                continue
                
            strategies[symbol] = []
            for strategy_name, params in symbol_strategies.items():
                if strategy_name in StrategyFactory.strategies:
                    strategies[symbol].append(
                        StrategyFactory.create_strategy(strategy_name, symbol, params)
                    )
                    
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
                self.risk_manager.current_equity = self.portfolio.value()
                signal = self.signal_aggregator.process_signals(symbol, data)
                if signal is not None and signal != 0:
                    await self._execute_trade(symbol, signal, current_price)
            
            self.portfolio.update_prices(prices)
            await self._update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    async def _execute_trade(self, symbol: str, signal: float, price: float, override_qty: float = None):
        
        if override_qty is not None:
            position_size = override_qty
        else:
            self.risk_manager.current_equity = self.portfolio.value()
            position_size = self.risk_manager.calculate_position_size(symbol, price)
        position_size = self.risk_manager.calculate_position_size(symbol, price)
        amount = signal * (position_size if override_qty is None else override_qty)
        current_amount = self.portfolio.positions.get(symbol, 0)
        if amount < 0:
            amount = max(amount, -current_amount)
        else:
            max_amount = self.portfolio.cash / price
            amount = min(amount, max_amount)
        # Skip the trade entirely if we have nothing to sell
        if abs(amount) < 1e-6:
            # logger.info(f"Skipping sell for {symbol}: no shares to sell")
            return
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
        if self.portfolio.cash >= cost and cost != 0:
            # Safety check: ensure we have enough shares for sells
            if amount < 0 and abs(amount) > current_amount:
                logger.warning(f"Safety check triggered: Attempted to sell {abs(amount):.2f} shares of {symbol} but only have {current_amount:.2f}")
                return  # Skip this trade
                
            # Safety check: ensure we have enough cash for buys    
            if amount > 0 and cost > self.portfolio.cash:
                logger.warning(f"Safety check triggered: Insufficient cash (${self.portfolio.cash:.2f}) to buy {amount:.2f} shares of {symbol} at ${price:.2f} (${cost:.2f})")
                return  # Skip this trade
                
            self.portfolio.positions[symbol] = current_amount + amount
            self.portfolio.cash -= cost
            self.portfolio._update_portfolio(symbol, amount, price, self.current_date, self.current_day)
            # logger.info(f"Date: {self.current_date.strftime('%Y-%m-%d')}, Day: {self.current_day}, Executed {amount:.2f} shares of {symbol} @ {price:.2f}")

    async def _rebalance_with_mvo(self):
        """Main MVO rebalancing routine"""
        logger.info(f"Performing MVO rebalancing on day {self.current_day}")
        
        # Get optimal weights
        optimal_weights = self._calculate_mvo_weights()
        if not optimal_weights:
            logger.warning("MVO optimization failed, skipping rebalance")
            return
            
        # Execute rebalancing trades
        await self._execute_mvo_rebalance(optimal_weights)
        
    async def _run_backtest(self):
        handler = self.data_handler
        
        # Make sure we have data for all symbols
        for symbol in self.symbols:
            if symbol not in handler.data or handler.data[symbol].empty:
                logger.error(f"No data available for symbol {symbol}. Cannot run backtest.")
                return
        
        # Limit the number of days for faster testing
        max_days = min(len(handler.data[sym]) for sym in self.symbols)
        max_days = min(max_days, config['max_backtest_days'])
        
        # Get dates from the first symbol's data
        first_symbol = self.symbols[0]
        self.backtest_dates = handler.data[first_symbol].index[:max_days].tolist()

        # Clear initial history entry (will be replaced with backtest dates)
        self.portfolio.history = []
        self.portfolio.historical_states = []
        self.portfolio.trade_history = []
        
        logger.info(f"Running backtest with {max_days} days of data")
        
        for day in range(max_days):
            self.current_day = day
            self.current_date = self.backtest_dates[day]
            handler.current_idx = day
            # MVO Rebalancing
            if self.mvo_enabled and day % self.rebalance_frequency == 0 and day != 0:
                await self._rebalance_with_mvo()
            # Walk-forward optimization check - only do this every few days to speed up
            if (day >= self.optimization_window_days and 
                (day - self.last_optimization_day) >= self.out_of_sample_window_days and
                day % 10 == 0):  # Only optimize every 10 days
                window_start = day - self.optimization_window_days
                window_end = day - 1
                self._optimize_weights(window_start, window_end)
                self.last_optimization_day = day
            await self._trading_cycle()
            
            # Get the current date from the data
            self.portfolio.snapshot_state(self.current_date)
            
            # Update portfolio history with the current date and value
            self.portfolio.history.append({
                'date': self.current_date.isoformat(),
                'value': self.portfolio.value()
            })
            
            self.backtest_results.append(self.portfolio.value())
            
            # Log progress occasionally
            if day % 50 == 0 or day == max_days - 1:
                logger.info(f"Backtest progress: Day {day}/{max_days} ({day/max_days:.0%}), Value: ${self.portfolio.value():.2f}")
        
        # Prepare data for the dashboard
        self.backtest_data = {
            'history': self.portfolio.historical_states,
            'trades': [
                {
                    'day': t['day'],
                    'date': pd.to_datetime(t['timestamp']).strftime('%Y-%m-%d'),
                    'symbol': t['symbol'],
                    'action': t['action'],
                    'amount': t['amount'],
                    'price': t['price'],
                    'value': t['value']
                }
                for t in self.portfolio.trade_history
            ]
        }


    def _calculate_mvo_weights(self):
        """Calculate optimal portfolio weights using MVO"""
        try:
            # Collect historical returns for all symbols
            returns_data = {}
            for symbol in self.symbols:
                data = self.data_handler.data[symbol]
                start_idx = max(0, self.current_day - self.mvo_lookback)
                closes = data['Close'].iloc[start_idx:self.current_day]
                returns = closes.pct_change().dropna()
                returns_data[symbol] = returns
                
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                logger.warning("Not enough data for MVO calculation")
                return None


            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Optimization setup
            n_assets = len(self.symbols)
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
                return -portfolio_return / portfolio_volatility  # Minimize negative Sharpe
            
            # Initial guess (equal weights)
            init_weights = np.ones(n_assets) / n_assets
            
            # Run optimization
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_weights /= optimal_weights.sum()  # Ensure normalization
                return dict(zip(self.symbols, optimal_weights))

            logger.error("MVO optimization failed: %s", result.message)
            return None
            
        except Exception as e:
            logger.error("Error in MVO calculation: %s", str(e))
            return None

    async def _execute_mvo_rebalance(self, target_weights: dict):
        """Execute trades to achieve target portfolio weights"""
        current_values = {}
        total_value = self.portfolio.value()
        
        # Get current prices and position values
        for symbol in self.symbols:
            data = await self.data_handler.get_historical_data(symbol, 1)
            price = data['Close'].iloc[-1]
            current_qty = self.portfolio.positions.get(symbol, 0)
            current_values[symbol] = current_qty * price
        
        # Calculate target values and deltas
        trades = []
        for symbol, weight in target_weights.items():
            target_value = total_value * weight
            current_value = current_values.get(symbol, 0)
            delta = target_value - current_value
            trades.append((symbol, delta))
        # Execute trades
        for symbol, delta in trades:
            if abs(delta) < 1e-6:  # Skip negligible trades
                continue
                
            data = await self.data_handler.get_historical_data(symbol, 1)
            price = data['Close'].iloc[-1]
            
            # Calculate quantity to trade
            qty = delta / price
            
            # Execute trade directly (bypass signal system)
            await self._execute_trade(
                symbol=symbol,
                signal=np.sign(qty),  # Direction only, quantity handled differently
                price=price,
                override_qty=abs(qty)  # New parameter to handle MVO quantities
            )

    def _optimize_weights(self, window_start: int, window_end: int):
        """Optimize strategy weights using specified historical window"""
        # logger.info(f"\n=== Optimizing weights using window {window_start}-{window_end} ===")
        
        for symbol in self.symbols:
            if symbol not in self.data_handler.data:
                continue

            full_data = self.data_handler.data[symbol]
            if len(full_data) < window_end:
                continue

            window_data = full_data.iloc[window_start:window_end+1]
            strategies = self.strategies[symbol]
            
            # Generate signals for each strategy in window
            strategy_signals = []
            for strategy in strategies:
                signals = []
                for i in range(len(window_data)):
                    data_slice = window_data.iloc[:i+1]
                    try:
                        signal = strategy.calculate_signal(data_slice)
                        num_signal = self._signal_to_numeric(signal)
                    except Exception as e:
                        logger.error(f"Signal error: {e}")
                        num_signal = 0
                    signals.append(num_signal)
                strategy_signals.append(signals)

            # Find optimal weights
            best_weights = self._find_optimal_weights(strategy_signals, window_data)
            
            # Update strategy weights
            for i, strategy in enumerate(strategies):
                strategy.params['weight'] = best_weights[i]
            # logger.info(f"Updated {symbol} weights: {best_weights}")

    def _signal_to_numeric(self, signal: dict) -> float:
        """Convert signal dict to numeric value"""
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        return action_map[signal['action']] * signal.get('strength', 0)

    def _find_optimal_weights(self, strategy_signals: List[List[float]], data: pd.DataFrame) -> List[float]:
        """Find optimal weights for multiple strategies to maximize Sharpe ratio"""
        best_sharpe = -np.inf
        n_strategies = len(strategy_signals)
        
        # Default to equal weights
        best_weights = [1.0 / n_strategies] * n_strategies  
        
        # Handle special case of single strategy
        if n_strategies == 1:
            return [1.0]
            
        # Handle case of two strategies with grid search (efficient)
        if n_strategies == 2:
            closes = data['Close'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else []
            
            for w1 in np.linspace(0, 1, 21):  # 0.0, 0.05, ..., 1.0
                w2 = 1 - w1
                weights = [w1, w2]
                
                combined = []
                for i in range(len(strategy_signals[0][:-1])):
                    # Weighted sum of signals
                    combined_signal = sum(w * strategy_signals[j][i] for j, w in enumerate(weights))
                    combined.append(combined_signal)
                
                if len(returns) == 0 or len(combined) == 0:
                    continue
                
                strategy_returns = [c*r for c, r in zip(combined, returns)]
                sharpe = self._calculate_sharpe(strategy_returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights
        
        # For 3 or more strategies, use a simplified approach (slower but general)
        elif n_strategies >= 3:
            closes = data['Close'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else []
            
            if len(returns) == 0:
                return best_weights
                
            # Test a limited set of weight combinations
            # For 3+ strategies, we'll try some pre-defined weight distributions
            weight_combinations = []
            
            # Equal weights
            weight_combinations.append([1.0/n_strategies] * n_strategies)
            
            # Single dominant strategy with equal remainder
            for i in range(n_strategies):
                weights = [0.1/(n_strategies-1)] * n_strategies
                weights[i] = 0.9
                weight_combinations.append(weights)
            
            # Pairs of strategies (if we have more than 3)
            if n_strategies > 3:
                for i in range(n_strategies):
                    for j in range(i+1, n_strategies):
                        weights = [0.05/(n_strategies-2)] * n_strategies
                        weights[i] = 0.475
                        weights[j] = 0.475
                        weight_combinations.append(weights)
            
            # Evaluate each weight combination
            for weights in weight_combinations:
                combined = []
                for i in range(len(strategy_signals[0][:-1])):
                    # Weighted sum of signals
                    combined_signal = sum(w * strategy_signals[j][i] for j, w in enumerate(weights) if i < len(strategy_signals[j]))
                    combined.append(combined_signal)
                
                strategy_returns = [c*r for c, r in zip(combined, returns)]
                sharpe = self._calculate_sharpe(strategy_returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights
        
        return best_weights
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio"""
        if not returns or np.std(returns) == 0:
            return -np.inf
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    async def _update_risk_metrics(self):
        # Update volatility metrics
        for symbol in self.symbols:
            if symbol in self.portfolio.current_prices:
                price_history = await self.data_handler.get_historical_data(symbol, 30)
                returns = np.diff(np.log(price_history['Close']))
                volatility = np.std(returns) * np.sqrt(252)
                self.risk_manager.update_volatility(symbol, volatility)

    def _create_default_indicators(self):
        """Create default indicators for all symbols when optimization fails"""
        logger.info("Creating default indicators for all symbols")
        
        # Define default periods and windows
        ma_periods = [10, 20, 50]
        rsi_period = 14
        bollinger_window = 20
        
        for symbol in self.symbols:
            if symbol not in self.data_handler.data:
                logger.warning(f"No data for symbol {symbol}, skipping default indicators")
                continue
                
            # Get the data for this symbol
            df = self.data_handler.data[symbol].copy()
            
            # Calculate moving averages
            for period in ma_periods:
                df[f'MA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
            df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands components
            df[f'MA_{bollinger_window}'] = df['Close'].rolling(window=bollinger_window, min_periods=1).mean()
            df[f'Std_{bollinger_window}'] = df['Close'].rolling(window=bollinger_window, min_periods=1).std()
            
            # Update the data handler with the new data
            self.data_handler.data[symbol] = df
            
        logger.info("Successfully created default indicators for all symbols")

# Modified TimeSeriesSplit for financial data
class PurgedTimeSeriesSplit(TimeSeriesSplit):
    """Version with gap to prevent information leakage"""
    def __init__(self, n_splits=5, purge_gap=5):
        super().__init__(n_splits)
        self.purge_gap = purge_gap

    def split(self, X):
        splits = super().split(X)
        for train_idx, test_idx in splits:
            yield (train_idx[:-self.purge_gap], 
                   test_idx[self.purge_gap:])
            


class StrategyOptimizer:
    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.optimization_results = {}
        self.cache_file = Path("optimized_params.json")
        self.cache_ttl = timedelta(days=30)  # Extended from 7 days to 30 days
        self.progress_bars = {}
        
        # Configure Optuna logging to suppress output
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.WARNING)  # Only show warnings and errors, not info

    def optimize_all(self):
        """Optimize parameters for all strategy/symbol combinations"""
        # Try to load cached results first - more aggressive caching
        
        logger.info("Starting strategy optimization")

        # Collect all tasks
        tasks = []
        for symbol in self.engine.symbols:
            for strategy in self.engine.strategies[symbol]:
                strategy_name = strategy.__class__.__name__
                tasks.append((symbol, strategy, strategy_name))
                
        
        # Use a simpler progress indicator instead of tqdm
        total_tasks = len(tasks)
        logger.info(f"Quick testing: Optimizing {total_tasks} strategy/symbol combinations...")
        
        for i, (symbol, strategy, strategy_name) in enumerate(tasks, 1):
            # Run each optimization sequentially for stability
            (symbol_result, strategy_name_result), params = self.optimize_strategy(symbol, strategy, strategy_name)
            self.optimization_results[(symbol_result, strategy_name_result)] = params
            # Log progress every 25% completion
            if i % max(1, total_tasks // 4) == 0 or i == total_tasks:
                logger.info(f"Optimization progress: {i}/{total_tasks} ({i/total_tasks:.0%} complete)")
            
        self._save_results_to_cache()

    def _save_results_to_cache(self):
        """Save optimized parameters to JSON file"""
        serializable = {
            str(k): v for k, v in self.optimization_results.items()
        }
        with open(self.cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'params': serializable
            }, f, indent=2)
            
    def _load_cached_results(self) -> bool:
        """Load cached results if available and fresh"""
        if not self.cache_file.exists():
            return False
            
        file_age = datetime.now() - datetime.fromtimestamp(
            self.cache_file.stat().st_mtime)
        if file_age > self.cache_ttl:
            return False
            
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                
            self.optimization_results = {
                tuple(k.strip("()").split(", ")): v 
                for k, v in data['params'].items()
            }
            logger.info(f"Loaded cached params from {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False
        
    def optimize_strategy(self, symbol: str, strategy: TradingStrategy, strategy_name: str = None):
        """Optimize parameters for a single strategy/symbol combination"""
        study = optuna.create_study(direction='maximize')
        
        # If strategy_name not provided, derive it from the class
        if strategy_name is None:
            strategy_name = strategy.__class__.__name__
            
        study.optimize(
            lambda trial: self._objective(trial, symbol, strategy),
            n_trials=config['optimization']['n_trials'],
            n_jobs=1,
            show_progress_bar=False  # Disable progress bar
        )
        
        return (symbol, strategy_name), study.best_params
        
    def _objective(self, trial, symbol: str, strategy: TradingStrategy) -> float:
        """Enhanced objective function with robust validation and metrics"""
        try:
            # 1. Get relevant data (last 3 years for optimization)
            full_data = self.engine.data_handler.data[symbol]
            data = full_data.iloc[-750:] if len(full_data) > 750 else full_data
            
            # 2. Suggest parameters with adaptive ranges
            params = self._suggest_adaptive_parameters(trial, strategy, data)
            
            # 3. Configure purged time series validation
            tscv = PurgedTimeSeriesSplit(
                n_splits=3,
                purge_gap=5  # Prevent look-ahead bias
            )
            
            # 4. Parallel evaluation of splits
            results = Parallel(n_jobs=-1)(
                delayed(self._evaluate_split)(data, train_idx, test_idx, strategy, params, trial)
                for train_idx, test_idx in tscv.split(data)
            )
            
            # 5. Aggregate results with robustness checks
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return -np.inf  # All splits failed
                
            sharpe_ratios = [r['sharpe'] for r in valid_results]
            drawdowns = [r['max_drawdown'] for r in valid_results]
            returns = [r['returns'] for r in valid_results]
            
            # 6. Composite objective function
            median_sharpe = np.nanmedian(sharpe_ratios)
            median_drawdown = np.nanmedian(drawdowns)
            median_return = np.nanmedian(returns)
            
            # Penalize high drawdowns and negative returns
            objective_value = median_sharpe * (1 - median_drawdown) 
            if median_return < 0:
                objective_value *= 0.5
                
            # Report intermediate values for pruning
            trial.report(objective_value, step=len(valid_results))
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            return objective_value
            
        except Exception as e:
            logger.debug(f"Trial failed: {str(e)}")
            return -np.inf

    def _evaluate_split(self, data, train_idx, test_idx, strategy, params, trial):
        """Evaluate a single split with enhanced metrics"""
        try:
            # Clone strategy with new parameters
            temp_strategy = strategy.__class__(strategy.symbol, params)
            
            # Train/test split with purge gap
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Simulate realistic execution with forward-looking prevention
            signals = []
            prices = []
            for i in range(len(test_data)):
                # Prevent look-ahead by using only data up to current point
                available_data = pd.concat([train_data, test_data.iloc[:i]])
                signal = temp_strategy.calculate_signal(available_data)
                signals.append(self._signal_to_numeric(signal))
                prices.append(test_data['Close'].iloc[i])
            
            # Calculate returns with transaction costs (0.1% per trade)
            returns = []
            position = 0
            prev_signal = 0
            for i in range(1, len(signals)):
                # Calculate price change
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                
                # Calculate position change
                signal_change = signals[i-1] - prev_signal
                transaction_cost = abs(signal_change) * 0.001  # 0.1% per trade
                ret -= transaction_cost
                
                returns.append(signals[i-1] * ret)
                prev_signal = signals[i-1]
            
            # Calculate performance metrics
            if len(returns) < 5 or np.std(returns) < 1e-6:
                return None
                
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            cum_returns = np.cumprod([1 + r for r in returns])
            max_drawdown = (np.maximum.accumulate(cum_returns) - cum_returns).max()
            total_return = cum_returns[-1] - 1
            
            return {
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'returns': total_return
            }
            
        except Exception as e:
            logger.debug(f"Split evaluation failed: {str(e)}")
            return None

    def _suggest_adaptive_parameters(self, trial, strategy: TradingStrategy, data: pd.DataFrame):
        """Enhanced parameter suggestions with data-driven ranges"""
        params = {}
        
        # Calculate volatility for adaptive parameter ranges
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        if isinstance(strategy, MeanReversionStrategy):
            # Scale lookback with volatility (shorter windows for higher volatility)
            base_lookback = max(10, min(50, int(30 / (volatility * 100 + 0.01))))
            params['lookback'] = trial.suggest_int(
                'lookback', 
                base_lookback - 5, 
                base_lookback + 5,
                step=2
            )
            params['threshold'] = trial.suggest_float(
                'threshold', 
                max(1.0, 1.5 - volatility * 10),
                min(3.0, 2.5 + volatility * 10),
                step=0.1
            )
            
        elif isinstance(strategy, BollingerBandsStrategy):
            params['lookback'] = trial.suggest_int(
                'lookback', 
                10, 
                50,
                step=5
            )
            # Adjust number of std devs based on volatility
            params['num_std'] = trial.suggest_float(
                'num_std', 
                max(1.5, 2.0 - volatility * 5),
                min(2.5, 2.0 + volatility * 5),
                step=0.1
            )
            
        elif isinstance(strategy, MovingAverageCrossoverStrategy):
            # Ensure short window < long window
            short = trial.suggest_int('short_window', 5, 30, step=1)
            params['long_window'] = trial.suggest_int(
                'long_window', 
                short + 10, 
                max(short + 20, 100),
                step=5
            )
            params['short_window'] = short
            
        elif isinstance(strategy, RSIStrategy):
            params['period'] = trial.suggest_int(
                'period', 
                10, 
                21,
                step=2
            )
            # Dynamic overbought/oversold levels based on volatility
            params['overbought'] = trial.suggest_int(
                'overbought', 
                65 + int(volatility * 500), 
                75 + int(volatility * 500)
            )
            params['oversold'] = trial.suggest_int(
                'oversold', 
                25 - int(volatility * 500), 
                35 - int(volatility * 500)
            )
            
        return params

    def _suggest_parameters(self, trial, strategy: TradingStrategy):
        """Parameter space definition with narrower search spaces for faster optimization"""
        params = {}
        if isinstance(strategy, MeanReversionStrategy):
            params['lookback'] = trial.suggest_int('lookback', 10, 50)  # Narrowed from 10-50
            params['threshold'] = trial.suggest_float('threshold', 1.0, 3.0)  # Narrowed from 1.0-3.0
        elif isinstance(strategy, BollingerBandsStrategy):
            params['lookback'] = trial.suggest_int('lookback', 10, 50)  # Narrowed from 10-50
            params['num_std'] = trial.suggest_float('num_std', 1.5, 2.5)  # Narrowed from 1.5-2.5
        elif isinstance(strategy, MovingAverageCrossoverStrategy):
            params['short_window'] = trial.suggest_int('short_window', 5, 20)  # Narrowed from 5-20
            params['long_window'] = trial.suggest_int('long_window', 30, 100)  # Narrowed from 30-100
        elif isinstance(strategy, RSIStrategy):
            params['period'] = trial.suggest_int('period', 10, 21)  # Narrowed from 10-21
            params['overbought'] = trial.suggest_int('overbought', 65, 75)  # Narrowed from 65-75
            params['oversold'] = trial.suggest_int('oversold', 25, 35)  # Narrowed from 25-35
        # if isinstance(strategy, MeanReversionStrategy):
        #     # Narrower parameter ranges
        #     params['lookback'] = trial.suggest_int('lookback', 15, 25)  # Narrowed from 10-50
        #     params['threshold'] = trial.suggest_float('threshold', 1.5, 2.5)  # Narrowed from 1.0-3.0
        # elif isinstance(strategy, BollingerBandsStrategy):
        #     params['lookback'] = trial.suggest_int('lookback', 15, 25)  # Narrowed from 10-50
        #     params['num_std'] = trial.suggest_float('num_std', 1.8, 2.2)  # Narrowed from 1.5-2.5
        # elif isinstance(strategy, MovingAverageCrossoverStrategy):
        #     params['short_window'] = trial.suggest_int('short_window', 8, 15)  # Narrowed from 5-20
        #     params['long_window'] = trial.suggest_int('long_window', 40, 70)  # Narrowed from 30-100
        # elif isinstance(strategy, RSIStrategy):
        #     params['period'] = trial.suggest_int('period', 12, 18)  # Narrowed from 10-21
        #     params['overbought'] = trial.suggest_int('overbought', 68, 72)  # Narrowed from 65-75
        #     params['oversold'] = trial.suggest_int('oversold', 28, 32)  # Narrowed from 25-35
        return params

    def _backtest_strategy(self, strategy: TradingStrategy, data: pd.DataFrame) -> dict:
        """Backtest a single strategy on given data"""
        # Calculate signals for this strategy
        signals = []
        for i in range(len(data)):
            try:
                signal = strategy.calculate_signal(data.iloc[:i+1])
                normalized = self._signal_to_numeric(signal)
                signals.append(normalized)
            except Exception as e:
                logger.error(f"Error calculating signal: {e}")
                signals.append(0)
                
        # Calculate returns based on signals
        returns = []
        for i in range(1, len(data)):
            price_change = (data['Close'].iloc[i] / data['Close'].iloc[i-1]) - 1
            signal_return = signals[i-1] * price_change
            returns.append(signal_return)
            
        # Calculate performance metrics
        if not returns:
            return {'sharpe': -999, 'max_drawdown': 0, 'returns': 0}
            
        sharpe_ratio = np.mean(returns) / (np.std(returns) or 0.0001) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = np.cumprod(np.array(returns) + 1)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate annualized returns
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        days = len(returns)
        annualized_returns = ((1 + total_return) ** (252/days)) - 1 if days > 0 else 0
        
        return {
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns': annualized_returns
        }

    def _signal_to_numeric(self, signal: dict) -> float:
        """Convert signal dict to numeric value"""
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        return action_map[signal['action']] * signal.get('strength', 0)

class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.engine = None
        self.optimizer = None
        self.results = {}
        
    def initialize(self):
        # Use the local import without the core prefix
        from algo_trading_system import TradingEngine
        self.engine = TradingEngine(self.config)
        self.optimizer = StrategyOptimizer(self.engine)

    async def run(self):
        if not self.engine:
            self.initialize()
            
        # Load cached parameters if available
        if not self.optimizer._load_cached_results():
            # Only optimize if no cache available
            self.optimizer.optimize_all()
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
    # 'data_path': 'C:/Users/jchang427/OneDrive - Georgia Institute of Technology/Random Projects/trading_sim_cursor/src/main/resources/market_data/subset',  # Simplified path that will use synthetic data
    'initial_equity': 100000,
    'walk_forward': {
        'optimization_window_days': 180,  # Reduced from 180 days to 90 days
        'out_of_sample_window_days': 21   # Increased from 21 to 60 for less frequent optimization
    },
    'mvo_enabled': False,  # Keep MVO disabled for speed
    'rebalance_frequency': 120,  # Reduced rebalancing frequency (less computation)
    'mvo_lookback': 250,        # Reduced from 252 to 126 (6 months instead of 1 year)
    'max_backtest_days': 250,
    'optimization': {
        'n_trials': 100,  # Reduced from 20
        'cache_ttl_days': 30,  # Increased from 7
    }
}

if __name__ == "__main__":
    backtester = Backtester(config)
    asyncio.run(backtester.run())