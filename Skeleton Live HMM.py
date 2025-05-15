# live_coincatch_bot.py
import sys
print("--- sys.path from SCRIPT START ---")
for p_path in sys.path:
    print(p_path)
print("--- END sys.path from SCRIPT START ---")
import sys
print(f"Script is running with: {sys.executable}") # Add this line

print("--- sys.path from SCRIPT START ---")
# ... (rest of the sys.path printing)
# Your existing imports like import os, etc., will follow here
import os
import hmac
# ... rest of your script
import os
import hmac
import hashlib
import base64
import urllib.parse
import time
import requests
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.signal import find_peaks # From your backtest
from scipy.stats import norm       # From your backtest
import math                        # From your backtest
from datetime import datetime, timedelta # From your backtest
from colorama import init, Fore, Style # For pretty logging
from pathlib import Path  # Import Path for handling file paths
import json  # Add this import for handling JSON serialization
import ccxt  # Add CCXT for exchange API interactions
from typing import Optional, List, Dict, Any
import joblib # For HMM model saving/loading
import pandas_ta as ta # For HMM feature calculation
from sklearn.preprocessing import StandardScaler # For HMM feature scaling

# Import pomegranate with error handling for HMM
try:
    from pomegranate import HiddenMarkovModel, State, DiscreteDistribution, MultivariateGaussianDistribution
    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False
    print("WARNING: pomegranate library not available. Install with: pip install pomegranate>=1.0.0")

# Initialize colorama
init(autoreset=True)

# Update logging configuration for Terminal-Only Output
logging.basicConfig(
    level=logging.INFO,  # Keep INFO or set to DEBUG for more detail during development
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # ONLY Log to console (standard output/error)
    ]
)
logging.info("--- TERMINAL LOGGING INITIALIZED (NO FILE LOG) ---") # New confirmation

# Changed from DEBUG to INFO
logging.getLogger("CoincatchApiClient").setLevel(logging.INFO)
# Set level for HMM logger
logging.getLogger("HMMRegimeDetector").setLevel(logging.INFO)

# --- Configuration (Adapted from Coincatch Example) ---
class Config:
    def __init__(self):
        # Specify the path to the .env file
        project_root = Path.cwd()  # Gets /Users/JorJ/LETSgot/
        dotenv_path = project_root / "A1 trading bots" / ".env"  # Correctly builds the path

        # Check if the .env file exists
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded .env from: {dotenv_path}")  # Debugging message
        else:
            print(f".env file not found at: {dotenv_path}")  # Debugging message

        # Load API credentials
        self.API_KEY = os.getenv("COINCATCH_API_KEY")
        self.API_SECRET = os.getenv("COINCATCH_SECRET")  # Ensure this matches your .env file
        self.PASSPHRASE = os.getenv("COINCATCH_PASSPHRASE")

        self.SYMBOL = "BTCUSDT_UMCBL" # Or whatever Coincatch calls it
        self.MARGIN_COIN = "USDT"
        self.PRODUCT_TYPE = "umcbl" # Or 'mcbl' etc. - check Coincatch docs

        self.KLINE_INTERVAL_PRIMARY = "4h" # e.g., "4h", "1H", "15m" on Coincatch
        self.KLINE_INTERVAL_SECONDARY = "1d" # e.g., "1d", "1D"

        # Strategy Parameters (from your backtest's Test33LIVEStrategy.params)
        self.MIN_QUANTITY = 0.001 # Example, check Coincatch minimums for BTCUSDT
        self.LEVERAGE = 30 # Ensure this is set on Coincatch or can be set via API
        self.TRANSACTION_COST = 0.0006
        self.MARGIN_PERCENTAGE = 0.5
        self.MAX_LOSS_PER_TRADE_USD = 500 # Max loss in USD

        self.RSI_PERIOD = 14
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        self.BB_PERIOD = 20
        self.BB_STD_DEV = 2.0
        self.MIN_SCORE_ENTRY = 1.5
        self.TRAILING_STOP_ACTIVATION_PROFIT_PCT = 0.15  # Reduced from 0.5 to 15%
        self.TRAILING_STOP_DISTANCE_PCT = 0.3      # Distance % (0.3 = 30%)
        self.BREAKEVEN_PROFIT_PCT = 0.02           # Reduced from 0.25 to 2%

        # For UPNL-based Stop Loss
        self.MAX_CANDLES_1D = 110  # Increased from 50 to 110 to ensure at least max_lookback (100) candles for Trend/Fib analysis
        self.MAX_CANDLES_4H = 200  # Add this line to define MAX_CANDLES_4H

        self.COOLDOWN_PERIOD_SECONDS = 300 # Min time between trades
        self.LOOP_SLEEP_SECONDS = 60       # How often the main loop runs

        # HMM Parameters
        self.HMM_N_REGIMES = 6
        self.HMM_N_FEATURES = 7 # Features: returns, volatility, rsi, macd_diff, bollinger_b, atr_val, volume_norm
        self.HMM_MODEL_PATH = "hmm_model.json"  # Path to save/load the HMM model
        self.HMM_SCALER_PATH = "hmm_scaler.joblib" # Path to save/load the feature scaler
        self.HMM_VOLATILITY_WINDOW = 10 # For rolling volatility feature

        # Use existing indicator periods for HMM
        self.HMM_RSI_PERIOD = self.RSI_PERIOD  # Uses existing RSI_PERIOD
        self.HMM_MACD_FAST = 12  # Default value if not defined elsewhere
        self.HMM_MACD_SLOW = 26  # Default value if not defined elsewhere
        self.HMM_MACD_SIGNAL = 9  # Default value if not defined elsewhere
        self.HMM_BB_PERIOD = self.BB_PERIOD  # Uses existing BB_PERIOD
        self.HMM_BB_STD_DEV = self.BB_STD_DEV  # Uses existing BB_STD_DEV
        self.HMM_ATR_PERIOD = 14  # Default value for ATR period

        # Regime names and colors for visualization/logging
        self.HMM_REGIME_NAMES = [
            "Bullish Trend",
            "Bearish Trend",
            "Range-Bound/Chop",
            "High-Volatility Breakout",
            "Pullback/Consolidation",
            "Fakeout/Liquidity Sweep"
        ]
        self.HMM_REGIME_COLORS = [
            Fore.GREEN,     # Bullish
            Fore.RED,       # Bearish
            Fore.YELLOW,    # Range-Bound
            Fore.CYAN,      # High-Vol Breakout
            Fore.MAGENTA,   # Pullback
            Fore.BLUE       # Fakeout
        ]

# --- Coincatch API Client (Simplified to match fibonacciSWI3NG223.py) ---
class CoincatchApiClient:
    def __init__(self, config):  # Ensure 'config' is accepted as a parameter
        self.config = config  # Store the config object for later use
        self.api_key = config.API_KEY
        self.api_secret = config.API_SECRET
        self.passphrase = config.PASSPHRASE
        self.base_url = getattr(config, "API_URL", "https://api.coincatch.com")  # Use API_URL from Config if defined
        self.logger = logging.getLogger("CoincatchApiClient")  # Initialize logger
        self.session = requests.Session()

        # --- CCXT Initialization ---
        self.ccxt_client = None
        try:
            self.ccxt_client = ccxt.coincatch({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.passphrase, # ccxt uses 'password' for passphrase
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap', # For futures/perpetual contracts
                    'adjustForTimeDifference': True, # Let ccxt handle time sync
                }
            })
            # Load markets information from the exchange
            self.ccxt_client.load_markets()
            self.log("CCXT Coincatch client initialized successfully.", level="info", color=Fore.GREEN)

            # Map the standard symbol to the CCXT unified symbol format
            self.market_symbol = self.config.SYMBOL
            # In CCXT, the format might be something like 'BTC/USDT:USDT' for perpetual
            # This is a simplistic mapping - adjust as needed for Coincatch's actual format
            base, quote = self.market_symbol.split('_')[0][:-4], self.market_symbol.split('_')[0][-4:]
            self.unified_symbol = f"{base}/{quote}:{quote}"
            self.log(f"Mapped exchange symbol {self.market_symbol} to CCXT unified symbol {self.unified_symbol}", level="info")

        except ccxt.AuthenticationError as e:
            self.log(f"CCXT Authentication Error: Invalid API Key/Secret/Passphrase. Error: {str(e)}", level="error", color=Fore.RED)
            self.ccxt_client = None
        except Exception as e:
            self.log(f"Failed to initialize CCXT client: {str(e)}", level="error", color=Fore.RED)
            self.ccxt_client = None

        # Ensure API keys are loaded
        if not self.api_key or not self.api_secret or not self.passphrase:
            self.logger.error("API Key, Secret, or Passphrase not loaded into CoincatchApiClient. Check .env and Config class.")

    def log(self, message, level="info", color=None, exc_info=None):
        """
        Enhanced logging method. File logging is primary.
        Direct colored console output from here is minimized to avoid duplication
        if the LiveStrategy layer already provides a colored summary.
        """
        # Log to the dedicated logger (goes to file and basic console via root StreamHandler)
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message, exc_info=exc_info)
        elif level.lower() == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)

        # Only print directly to console with color for ERROR or CRITICAL messages from API client
        # if the 'color' argument is explicitly passed.
        # Most INFO level colored messages should come from LiveStrategy.log
        if color and (level.lower() == "error" or level.lower() == "critical"):
            print(f"{color}{message}{Style.RESET_ALL}")
        elif color and level.lower() == "warning": # Optionally, important warnings too
             print(f"{color}{message}{Style.RESET_ALL}")

    def set_leverage(self, symbol: str, margin_coin: str, leverage: int, position_side: str) -> bool:
        """
        Sets the leverage for a specific symbol and position side (long/short) using CCXT.
        """
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot set leverage", level="error", color=Fore.RED)
            return False

        self.log(f"Attempting to set leverage (via CCXT) to {leverage}x for {symbol} ({position_side})", level="info")

        try:
            # Prepare parameters for CCXT call
            params = {
                'marginCoin': margin_coin,
                'holdSide': position_side.lower()  # 'long' or 'short'
            }

            # Call CCXT set_leverage
            response = self.ccxt_client.set_leverage(leverage, self.unified_symbol, params=params)

            self.log(f"CCXT set_leverage response: {response}", level="info", color=Fore.GREEN)
            return True

        except ccxt.AuthenticationError as e:
            self.log(f"CCXT Authentication Error setting leverage: {e}", level="error", color=Fore.RED)
        except ccxt.ExchangeError as e:
            self.log(f"CCXT Exchange Error setting leverage: {e}", level="error", color=Fore.RED)
        except Exception as e:
            self.log(f"Unexpected error setting leverage via CCXT: {e}", level="error", color=Fore.RED, exc_info=True)

        return False

    def get_position(self, symbol: str, margin_coin: str) -> dict:
        """Gets the open position for a specific symbol using CCXT."""
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot fetch position", level="error", color=Fore.RED)
            return None

        target_symbol = self.unified_symbol # Use the unified symbol
        self.log(f"Fetching position for {target_symbol}", level="info")

        try:
            # Fetch all positions and find the one matching our symbol
            all_positions = self.ccxt_client.fetch_positions([target_symbol])

            if all_positions and isinstance(all_positions, list):
                # Find position matching our symbol
                position = None
                for p in all_positions:
                    if p.get('symbol') == target_symbol:
                        position = p
                        break

                if position:
                    # Extract position details
                    contracts = float(position.get('contracts', position.get('contractSize', position.get('amount', 0))))
                    side = position.get('side', '').lower()
                    entry_price = position.get('entryPrice')

                    # Verify we have a valid position
                    if contracts > 0 and side in ['long', 'short'] and entry_price is not None and float(entry_price) > 0:
                        # Format for strategy
                        strategy_position = {
                            "total": contracts,
                            "holdSide": side,
                            "averageOpenPrice": float(entry_price),
                            "avgEntryPrice": float(entry_price),  # Add this alias as some code might use it
                            "leverage": float(position.get('leverage', 0)),
                            "liquidationPrice": position.get('liquidationPrice'),
                            "marginMode": position.get('marginMode', 'isolated'),
                            "marketPrice": position.get('markPrice'),
                            "unrealizedPL": position.get('unrealizedPnl'),
                        }
                        self.log(f"Found open position: {side.upper()} {contracts} @ {entry_price}", level="info", color=Fore.GREEN)
                        return strategy_position

            self.log("No active position found", level="info")
            return None

        except ccxt.AuthenticationError as e:
            self.log(f"CCXT Authentication Error fetching position: {e}", level="error", color=Fore.RED)
            return None
        except ccxt.ExchangeError as e:
            self.log(f"CCXT Exchange Error fetching position: {e}", level="error", color=Fore.RED)
            return None
        except Exception as e:
            self.log(f"Unexpected error fetching position: {e}", level="error", color=Fore.RED, exc_info=True)
            return None

    def get_account_balance(self, symbol: str, margin_coin: str, product_type: str) -> dict:
        """Gets account details, including available balance using CCXT."""
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot fetch balance", level="error", color=Fore.RED)
            return None

        self.log(f"Fetching balance for {margin_coin}", level="info")

        try:
            # Fetch balance with swap product type
            balance_data = self.ccxt_client.fetch_balance(params={'type': 'swap'})

            # Verify we have data for our margin coin
            if margin_coin in balance_data.get('total', {}):
                # Extract balance components
                total_equity = balance_data['total'].get(margin_coin, 0.0)
                free_balance = balance_data['free'].get(margin_coin, 0.0)
                used_margin = balance_data['used'].get(margin_coin, 0.0)

                # Extract unrealized PNL (location can vary)
                upnl = 0.0
                if 'info' in balance_data:
                    info = balance_data['info']
                    if isinstance(info, list) and len(info) > 0:
                        upnl = float(info[0].get('unrealizedPL', info[0].get('unrealizedPnl', 0.0)))
                    elif isinstance(info, dict):
                        upnl = float(info.get('unrealizedPL', info.get('unrealizedPnl', 0.0)))

                # Format response for strategy
                formatted_balance = {
                    "code": "00000",  # Success code
                    "msg": "success",
                    "data": {
                        "marginCoin": margin_coin,
                        "equity": float(total_equity),
                        "available": float(free_balance),
                        "locked": float(used_margin),
                        "unrealizedPL": float(upnl),
                        # Add other fields if needed
                    }
                }

                self.log(f"Balance: Equity={total_equity}, Available={free_balance}, PNL={upnl}", level="info", color=Fore.GREEN)
                return formatted_balance
            else:
                self.log(f"Margin coin {margin_coin} not found in balance response", level="warning", color=Fore.YELLOW)
                return None

        except ccxt.AuthenticationError as e:
            self.log(f"CCXT Authentication Error fetching balance: {e}", level="error", color=Fore.RED)
            return None
        except ccxt.ExchangeError as e:
            self.log(f"CCXT Exchange Error fetching balance: {e}", level="error", color=Fore.RED)
            return None
        except Exception as e:
            self.log(f"Unexpected error fetching balance: {e}", level="error", color=Fore.RED, exc_info=True)
            return None

    def get_candles(self, symbol: str, interval_from_config: str = "1H", limit: int = 100, fetch_history: bool = False):
        """
        Get historical klines (candlestick data) for a symbol using CCXT.
        """
        if not self.ccxt_client:
             self.log("CCXT client not initialized, cannot fetch candles", level="error", color=Fore.RED)
             return []

        # Map timeframe format for CCXT
        ccxt_timeframe = interval_from_config.lower()
        if 'h' in ccxt_timeframe:
            ccxt_timeframe = interval_from_config.lower()
        elif 'd' in ccxt_timeframe:
            ccxt_timeframe = interval_from_config.lower()
        elif 'm' in ccxt_timeframe:
            ccxt_timeframe = interval_from_config.lower()

        # Use unified symbol
        target_symbol = self.unified_symbol
        self.log(f"Fetching {limit} {ccxt_timeframe} candles for {target_symbol}", level="info")

        try:
            # Fetch OHLCV data
            ohlcv_list = self.ccxt_client.fetch_ohlcv(target_symbol, ccxt_timeframe, limit=limit)

            if not ohlcv_list:
                self.log(f"Received empty candle list from CCXT", level="warning")
                return []

            # Parse candles
            parsed_candles = []
            for candle in ohlcv_list:
                if isinstance(candle, list) and len(candle) >= 6:
                    try:
                        # CCXT format: [timestamp_ms, open, high, low, close, volume]
                        candle_obj = CandleData(
                            timestamp=int(candle[0] / 1000),  # ms to s
                            open_price=float(candle[1]),
                            high=float(candle[2]),
                            low=float(candle[3]),
                            close=float(candle[4]),
                            volume=float(candle[5])
                        )
                        parsed_candles.append(candle_obj)
                    except (ValueError, TypeError, IndexError) as e:
                        self.log(f"Error parsing candle data: {e}", level="error")
                        continue

            if parsed_candles:
                self.log(f"Successfully fetched {len(parsed_candles)} candles", level="info")
                return sorted(parsed_candles, key=lambda c: c.timestamp)
            else:
                self.log("Failed to parse any valid candles", level="warning", color=Fore.YELLOW)
                return []

        except ccxt.NetworkError as e:
            self.log(f"CCXT Network Error fetching candles: {e}", level="error", color=Fore.RED)
        except ccxt.ExchangeError as e:
            self.log(f"CCXT Exchange Error fetching candles: {e}", level="error", color=Fore.RED)
            if 'not found' in str(e).lower() or 'invalid symbol' in str(e).lower():
                self.log(f"Check if symbol '{target_symbol}' or timeframe '{ccxt_timeframe}' is valid", level="error", color=Fore.RED)
        except Exception as e:
            self.log(f"Unexpected error fetching candles: {e}", level="error", color=Fore.RED, exc_info=True)

        return []

    def place_market_order(self, symbol: str, side: str, amount_base_currency: float,
                           stop_loss_price: Optional[float] = None,
                           take_profit_price: Optional[float] = None,
                           params={}):
        """
        Places a market order using CCXT, passing Coincatch-specific parameters.
        :param symbol: Unified symbol (ignored, uses self.unified_symbol)
        :param side: Standard 'buy' or 'sell'
        :param amount_base_currency: Amount of the base currency (e.g., BTC)
        :param stop_loss_price: Optional stop loss price
        :param take_profit_price: Optional take profit price
        :param params: Base extra parameters for CCXT create_order (e.g., {'reduceOnly': True})
        :return: Order response from CCXT or None on error
        """
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot place order", level="error", color=Fore.RED)
            return None

        # --- Coincatch Specific Side Mapping ---
        coincatch_side_map = {
            "buy": "OPEN_LONG",
            "sell": "OPEN_SHORT"
        }
        if side.lower() not in coincatch_side_map:
            self.log(f"Invalid order side '{side}' for placing market order.", level="error", color=Fore.RED)
            return None
        coincatch_side = coincatch_side_map[side.lower()]
        # --- End Mapping ---

        order_type = 'market'
        target_symbol = self.unified_symbol

        # --- Build CCXT Params ---
        ccxt_params = params.copy()
        ccxt_params['side'] = coincatch_side

        market_details = self.get_market_details(target_symbol)
        price_precision = None
        if market_details and market_details.get('precision'):
            precision_val = market_details['precision'].get('price')
            if isinstance(precision_val, float) and precision_val < 1:
                price_precision = abs(int(math.log10(precision_val)))
            elif isinstance(precision_val, int):
                price_precision = precision_val

        def format_price(price, precision):
            if price is None: return None
            try:
                if precision is not None:
                    return f"{float(price):.{precision}f}"
                else:
                    return f"{float(price):.1f}"
            except (ValueError, TypeError): return None

        sl_price_str = format_price(stop_loss_price, price_precision)
        tp_price_str = format_price(take_profit_price, price_precision)

        if sl_price_str:
            ccxt_params['presetStopLossPrice'] = sl_price_str
        if tp_price_str:
            ccxt_params['presetTakeProfitPrice'] = tp_price_str
        # --- End Build CCXT Params ---

        sl_log = sl_price_str if sl_price_str else "N/A"
        tp_log = tp_price_str if tp_price_str else "N/A"
        self.log(f"Attempting CCXT place order: Symbol={target_symbol}, Type={order_type}, Amount={amount_base_currency}, Params={ccxt_params}", level="info")
        self.log(f"Placing MARKET {side.upper()} order ({coincatch_side}): Size={amount_base_currency}, SL={sl_log}, TP={tp_log}", color=Fore.YELLOW)

        try:
            order = self.ccxt_client.create_order(
                symbol=target_symbol,
                type=order_type,
                side=side,
                amount=amount_base_currency,
                params=ccxt_params
            )
            self.log(f"Market {side.upper()} order placed successfully: {order.get('id', 'N/A')}. Full response: {order}", level="info", color=Fore.GREEN)
            return order
        except ccxt.InsufficientFunds as e:
            self.log(f"CCXT Insufficient Funds: {e}", level="error", color=Fore.RED)
        except ccxt.NetworkError as e:
            self.log(f"CCXT Network Error: {e}", level="error", color=Fore.RED)
        except ccxt.ExchangeError as e:
            self.log(f"CCXT Exchange Error placing order on {target_symbol}: {e}", level="error", color=Fore.RED)
        except Exception as e:
            self.log(f"Unexpected error placing order on {target_symbol}: {e}", level="error", color=Fore.RED, exc_info=True)
        return None

    def close_position_market(self, symbol: str, position_side: str, position_amount_base: float):
        """
        Closes an existing position with a market order using Coincatch-specific side.
        :param symbol: Unified symbol (ignored, uses self.unified_symbol)
        :param position_side: 'long' or 'short' (the side of the position you want to close)
        :param position_amount_base: The size of the position in base currency (e.g., BTC)
        """
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot close position", level="error", color=Fore.RED)
            return None

        coincatch_close_side_map = {
            "long": "CLOSE_LONG",
            "short": "CLOSE_SHORT"
        }
        if position_side.lower() not in coincatch_close_side_map:
            self.log(f"Invalid position_side '{position_side}' for closing.", level="error", color=Fore.RED)
            return None
        coincatch_close_side = coincatch_close_side_map[position_side.lower()]

        standard_close_side = 'sell' if position_side.lower() == 'long' else 'buy'

        self.log(f"Attempting to CLOSE {position_side.upper()} position of {position_amount_base} {self.unified_symbol.split('/')[0]} using Coincatch side: {coincatch_close_side}", level="info", color=Fore.YELLOW)

        try:
            params = {
                'reduceOnly': True,
                'side': coincatch_close_side
            }
            order = self.place_market_order(
                symbol=self.unified_symbol,
                side=standard_close_side,
                amount_base_currency=position_amount_base,
                params=params
            )

            if order:
                self.log(f"Position close order ({order.get('id', 'N/A')}) placed for {position_side.upper()}.", level="info", color=Fore.GREEN)
                return order
            else:
                self.log(f"Failed to place position close order for {position_side.upper()}.", level="warning", color=Fore.YELLOW)
                return None

        except Exception as e:
            self.log(f"Error in close_position_market: {e}", level="error", color=Fore.RED, exc_info=True)
            return None

    def get_market_details(self, symbol: str) -> dict:
        """Fetches market details like precision and limits for self.unified_symbol using CCXT."""
        if not self.ccxt_client:
            self.log("CCXT client not initialized, cannot fetch market details", level="error", color=Fore.RED)
            return None
        try:
            market = self.ccxt_client.market(self.unified_symbol)
            return market
        except Exception as e:
            self.log(f"Error fetching market details for {self.unified_symbol}: {e}", level="error", color=Fore.RED)
            return None

# --- CandleData Class (From your backtest) ---
class CandleData:
    def __init__(self, timestamp, open_price, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self):
        dt_object = datetime.fromtimestamp(self.timestamp)
        return (f"Candle(T={dt_object.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"O={self.open}, H={self.high}, L={self.low}, C={self.close}, V={self.volume})")

# --- HMM Regime Detector ---
class HMMRegimeDetector:
    def __init__(self, config):
        """
        Initialize HMM Regime Detector.

        Args:
            config: Configuration object with HMM parameters
        """
        self.logger = logging.getLogger("HMMRegimeDetector")

        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Pomegranate library not available. HMM functionality disabled.")
            return

        # Extract configuration from config object
        self.config = config
        self.n_regimes = config.HMM_N_REGIMES
        self.n_features = config.HMM_N_FEATURES
        self.model_path = Path(config.HMM_MODEL_PATH)
        self.scaler_path = Path(config.HMM_SCALER_PATH)
        self.volatility_window = config.HMM_VOLATILITY_WINDOW
        self.rsi_period = config.HMM_RSI_PERIOD
        self.macd_fast = config.HMM_MACD_FAST
        self.macd_slow = config.HMM_MACD_SLOW
        self.macd_signal = config.HMM_MACD_SIGNAL
        self.bb_period = config.HMM_BB_PERIOD
        self.bb_std_dev = config.HMM_BB_STD_DEV
        self.atr_period = config.HMM_ATR_PERIOD
        self.regime_names = config.HMM_REGIME_NAMES
        self.regime_colors = config.HMM_REGIME_COLORS

        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._load_model_and_scaler()

    def _build_model(self):
        """Build initial HMM model structure (not fitted)"""
        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Cannot build model: pomegranate library not available")
            return

        # For pomegranate >= 1.0.0
        self.model = HiddenMarkovModel(name="MarketRegimeHMM")
        states = []
        for i in range(self.n_regimes):
            # Dummy initialization for MultivariateGaussianDistribution
            distribution = MultivariateGaussianDistribution.from_samples(
                np.random.randn(self.n_regimes * 10, self.n_features)
            )
            state = State(distribution, name=f"regime_{i}")
            states.append(state)

        self.model.add_states(states)

        # Initialize transitions (uniform start and transitions)
        for state_obj in states:  # Start probabilities
            self.model.add_transition(self.model.start, state_obj, 1.0 / self.n_regimes)
        for from_state_obj in states:  # Transition probabilities
            for to_state_obj in states:
                self.model.add_transition(from_state_obj, to_state_obj, 1.0 / self.n_regimes)

        self.model.bake(verbose=True)
        self.is_fitted = False
        self.logger.info("Built new HMM model structure (pomegranate >= 1.0.0).")

    def prepare_features(self, candles_data: List[CandleData], fit_scaler=False) -> Optional[np.ndarray]:
        """
        Prepare features for HMM from candle data

        Args:
            candles_data: List of CandleData objects
            fit_scaler: Whether to fit the scaler on this data

        Returns:
            Normalized feature array or None if error
        """
        min_len_rsi = self.rsi_period
        min_len_macd = self.macd_slow + self.macd_signal
        min_len_bb = self.bb_period
        min_len_atr = self.atr_period
        min_len_vol_rolling = self.volatility_window

        # Make sure we have enough candles for all features
        required_data_length = max(min_len_rsi, min_len_macd, min_len_bb, min_len_atr, min_len_vol_rolling) + 20

        if not candles_data or len(candles_data) < required_data_length:
            self.logger.warning(f"Not enough candle data ({len(candles_data) if candles_data else 0}) for feature preparation. Need at least {required_data_length}.")
            return None

        # Convert candle data to DataFrame
        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        } for c in candles_data])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")

        # Calculate features
        try:
            # 1. Log Returns
            df["returns"] = np.log(df["close"] / df["close"].shift(1))

            # 2. Volatility (rolling std dev of returns)
            df["volatility"] = df["returns"].rolling(window=self.volatility_window).std()

            # 3. RSI
            df.ta.rsi(length=self.rsi_period, append=True)
            df["rsi"] = df[f"RSI_{self.rsi_period}"]

            # 4. MACD difference (histogram)
            macd_df = df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=False)
            df["macd_diff"] = macd_df[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"]

            # 5. Bollinger %B
            bbands_df = df.ta.bbands(length=self.bb_period, std=self.bb_std_dev, append=False)
            lower_col = f"BBL_{self.bb_period}_{self.bb_std_dev:.1f}"
            upper_col = f"BBU_{self.bb_period}_{self.bb_std_dev:.1f}"
            df["bb_lower"] = bbands_df[lower_col]
            df["bb_upper"] = bbands_df[upper_col]
            df["bollinger_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            df["bollinger_b"] = df["bollinger_b"].replace([np.inf, -np.inf], np.nan).fillna(0.5)

            # 6. ATR
            df.ta.atr(length=self.atr_period, append=True)
            df["atr_val"] = df[f"ATR_{self.atr_period}"]

            # 7. Normalized Volume
            volume_mean = df["volume"].rolling(window=self.volatility_window).mean()
            volume_std = df["volume"].rolling(window=self.volatility_window).std()
            df["volume_norm"] = (df["volume"] - volume_mean) / volume_std
            df["volume_norm"] = df["volume_norm"].replace([np.inf, -np.inf], np.nan).fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}", exc_info=True)
            return None

        # Select feature columns and drop rows with NaN
        feature_columns = ["returns", "volatility", "rsi", "macd_diff", "bollinger_b", "atr_val", "volume_norm"]
        df_features = df[feature_columns].copy()
        df_features = df_features.dropna()

        if df_features.empty:
            self.logger.warning("DataFrame empty after NaN drop in feature preparation.")
            return None

        raw_features = df_features.values

        if raw_features.shape[1] != self.n_features:
            self.logger.error(f"Expected {self.n_features} features, but got {raw_features.shape[1]}.")
            return None

        # Normalize features
        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(raw_features)
            self._save_scaler()
            self.logger.info("Fitted and saved new scaler.")
        else:
            try:
                normalized_features = self.scaler.transform(raw_features)
            except Exception as e:
                self.logger.error(f"Error transforming features: {e}. Re-fitting scaler.")
                self.scaler = StandardScaler()
                normalized_features = self.scaler.fit_transform(raw_features)
                self._save_scaler()

        return normalized_features

    def train(self, historical_candles_data: List[CandleData], max_iterations=100):
        """
        Train the HMM model with historical candle data

        Args:
            historical_candles_data: List of CandleData objects
            max_iterations: Maximum number of EM iterations for HMM training
        """
        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Cannot train model: pomegranate library not available")
            return

        self.logger.info("Preparing features for training...")
        features = self.prepare_features(historical_candles_data, fit_scaler=True)

        min_training_samples = self.n_regimes * 20  # Heuristic: e.g., 20 samples per regime
        if features is None or features.shape[0] < min_training_samples:
            self.logger.error(f"Not enough data for training: {features.shape[0] if features is not None else 0} samples (need {min_training_samples}).")
            return

        self.logger.info(f"Training HMM with {features.shape[0]} samples and {features.shape[1]} features...")

        if self.model is None:
            self._build_model()

        try:
            # Train the model (fit expects a list of sequences)
            self.model.fit([features], max_iterations=max_iterations, verbose=True, n_jobs=-1)
            self.is_fitted = True
            self._save_model()
            self.logger.info("Training complete and model saved.")
            self.analyze_regimes()
        except Exception as e:
            self.logger.error(f"Error training HMM model: {e}", exc_info=True)
            self.is_fitted = False

    def predict_regime(self, current_candles_data: List[CandleData]) -> Optional[int]:
        """
        Predict the current market regime using the trained HMM

        Args:
            current_candles_data: List of CandleData objects

        Returns:
            Index of the predicted regime or None on error
        """
        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Cannot predict: pomegranate library not available")
            return None

        if not self.is_fitted:
            self.logger.warning("Model not fitted. Cannot predict.")
            return None

        if self.scaler is None:
            self.logger.error("Scaler not available. Cannot normalize features.")
            return None

        features = self.prepare_features(current_candles_data, fit_scaler=False)

        if features is None or features.shape[0] == 0:
            self.logger.warning("Feature preparation failed or yielded no data.")
            return None

        try:
            # For pomegranate >= 1.0.0, predict returns the state indices directly
            if not isinstance(features, list):
                features = [features]
            predicted_states_indices = self.model.predict(features)[0]
            if not predicted_states_indices:
                self.logger.warning("HMM: Prediction yielded empty state sequence.")
                return None
            current_regime_index = predicted_states_indices[-1]
            return int(current_regime_index)
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            return None

    def _save_model(self):
        """Save the trained HMM model to a file"""
        if not self.model:
            return

        try:
            with open(self.model_path, "w") as f:
                f.write(self.model.to_json())
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}", exc_info=True)

    def _save_scaler(self):
        """Save the feature scaler to a file"""
        if not self.scaler:
            return

        try:
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info(f"Scaler saved to {self.scaler_path}")
        except Exception as e:
            self.logger.error(f"Failed to save scaler: {e}", exc_info=True)

    def _load_model_and_scaler(self):
        """Load the HMM model and scaler from files"""
        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Cannot load model: pomegranate library not available")
            return

        # Try to load the model
        model_loaded = False
        if self.model_path.exists():
            try:
                with open(self.model_path, "r") as f:
                    model_json_str = f.read()
                # For pomegranate >= 1.0.0, from_json is a class method
                self.model = HiddenMarkovModel.from_json(model_json_str)
                self.is_fitted = True
                model_loaded = True
                self.logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.model = None
                self.is_fitted = False
        else:
            self.logger.info(f"Model file not found at {self.model_path}")

        # If model failed to load, build a new (unfitted) model
        if not model_loaded:
            self._build_model()

        # Try to load the scaler
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"Scaler loaded from {self.scaler_path}")
            except Exception as e:
                self.logger.error(f"Error loading scaler: {e}")
                self.scaler = None
        else:
            self.logger.info(f"Scaler file not found at {self.scaler_path}")
            self.scaler = None

    def analyze_regimes(self):
        """Analyze and log characteristics of the trained regimes"""
        if not POMEGRANATE_AVAILABLE:
            self.logger.error("Cannot analyze regimes: pomegranate library not available")
            return

        if not self.model or not self.is_fitted:
            self.logger.warning("Model is not trained or loaded. Cannot analyze regimes.")
            return

        self.logger.info("\n" + "="*15 + " HMM REGIME ANALYSIS " + "="*15)

        # Get all states with distributions (actual observable states)
        actual_states = [s for s in self.model.states if hasattr(s, 'distribution') and s.distribution is not None]

        feature_names = ["returns", "volatility", "rsi", "macd_diff", "bollinger_b", "atr_val", "volume_norm"]

        for i, state in enumerate(actual_states):
            # Try to get regime index from state name
            regime_idx = -1
            try:
                regime_idx = int(state.name.split("_")[-1])
            except (ValueError, IndexError):
                regime_idx = i

            # Get regime name and color from config
            regime_name = self.regime_names[regime_idx] if 0 <= regime_idx < len(self.regime_names) else f"Unnamed Regime {regime_idx}"
            regime_color = self.regime_colors[regime_idx] if self.regime_colors and 0 <= regime_idx < len(self.regime_colors) else ""

            self.logger.info(f"{regime_color}--- {regime_name} (State {i}, Name: {state.name}) ---{Style.RESET_ALL}")

            if isinstance(state.distribution, MultivariateGaussianDistribution):
                dist_params = state.distribution.parameters
                means = dist_params[0]

                self.logger.info(f"{regime_color}  Mean Feature Values (Normalized):{Style.RESET_ALL}")
                for feat_idx, mean_val in enumerate(means):
                    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx+1}"
                    self.logger.info(f"{regime_color}    {feat_name:<15}: {mean_val:.3f}{Style.RESET_ALL}")
            else:
                self.logger.info(f"{regime_color}  Distribution Type: {type(state.distribution).__name__ if state.distribution else 'None'}{Style.RESET_ALL}")

        self.logger.info("="*40 + "\n")

# --- Indicator Classes (TrendLineDetector, FibonacciCircles, BollingerBands, RSI - From your backtest) ---
class TrendLineDetector:
    def __init__(self, min_points=3, max_lookback=100, min_strength=2):
        self.min_points = min_points
        self.max_lookback = max_lookback
        self.min_strength = min_strength

    def detect_trend_lines(self, candles):
        """
        Detect trend lines from price data.
        Returns a list of trend lines, each trend line is a tuple (slope, intercept, strength, points).
        """
        if len(candles) < self.min_points:
            return []

        # Find peaks (highs and lows) in the price data
        closes = np.array([candle.close for candle in candles])
        highs = np.array([candle.high for candle in candles])
        lows = np.array([candle.low for candle in candles])

        # Find peak indices
        high_peaks, _ = find_peaks(highs)
        low_peaks, _ = find_peaks(-lows)  # Negate lows to find troughs

        # Create candidate points for trend lines
        high_points = [(i, highs[i]) for i in high_peaks]
        low_points = [(i, lows[i]) for i in low_peaks]

        # Detect trend lines using linear regression
        resistance_lines = self._fit_trend_lines(high_points)
        support_lines = self._fit_trend_lines(low_points)

        # Combine and filter trend lines
        all_lines = resistance_lines + support_lines
        filtered_lines = [line for line in all_lines if line[2] >= self.min_strength]

        return sorted(filtered_lines, key=lambda x: x[2], reverse=True)

    def _fit_trend_lines(self, points):
        """Helper method to fit trend lines to a set of points."""
        trend_lines = []

        for i in range(len(points)):
            for j in range(i+1, len(points)):
                # Ensure points are not too far apart
                if points[j][0] - points[i][0] > self.max_lookback:
                    continue

                # Calculate slope and intercept of the line connecting these two points
                x1, y1 = points[i]
                x2, y2 = points[j]

                if x2 == x1:  # Avoid division by zero
                    continue

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Count points that are close to this line
                strength = self._count_supporting_points(points, slope, intercept)

                # Add to trend lines if strong enough
                if strength >= self.min_strength:
                    trend_lines.append((slope, intercept, strength, [points[i], points[j]]))

        return trend_lines

    def _count_supporting_points(self, points, slope, intercept, threshold=0.01):
        """Count how many points support a trend line."""
        count = 0
        for x, y in points:
            y_line = slope * x + intercept
            # If point is close to the line, it supports the trend line
            if abs(y - y_line) / y < threshold:
                count += 1
        return count

class FibonacciCircles:
    def generate_circles(self, trend_lines, max_circles=5):
        """
        Generate Fibonacci circles from trend lines.
        Returns a list of circles, each circle is a tuple (center_x, center_y, radius).
        """
        if not trend_lines:
            return []

        circles = []
        for slope, intercept, strength, points in trend_lines[:max_circles]:
            # Use the endpoints of the trend line as the diameter of a circle
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Calculate center of the circle
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Calculate radius based on distance between points
            base_radius = ((x2 - x1)**2 + (y2 - y1)**2)**0.5 / 2

            # Generate Fibonacci circles with different radii
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.27, 1.618, 2.0, 2.618]
            for level in fib_levels:
                radius = base_radius * level
                circles.append((center_x, center_y, radius, level))

        return circles

    def get_support_resistance_levels(self, circles, current_idx):
        """
        Calculate potential support and resistance levels based on Fibonacci circles.
        Returns two lists: support levels and resistance levels.
        """
        if not circles:
            return [], []

        # For simplicity, just extract the y-coordinates where the circles cross vertical line at current_idx
        support_levels = []
        resistance_levels = []

        for center_x, center_y, radius, level in circles:
            # Calculate vertical distance from center to current index
            h_dist = current_idx - center_x

            # Skip if current index is outside the circle
            if abs(h_dist) > radius:
                continue

            # Calculate vertical intersections using Pythagorean theorem
            v_dist = (radius**2 - h_dist**2)**0.5

            # Add to support and resistance levels
            resistance_level = center_y + v_dist
            support_level = center_y - v_dist

            support_levels.append(support_level)
            resistance_levels.append(resistance_level)

        return sorted(set(support_levels)), sorted(set(resistance_levels))

class BollingerBands:
    def __init__(self, period=20, std_dev=2.0):  # Ensure 'period' and 'std_dev' are accepted as parameters
        self.period = period
        self.std_dev = std_dev

    def calculate(self, candles):
        if len(candles) < self.period:
            return {'upper': np.array([]), 'middle': np.array([]), 'lower': np.array([])}

        closes = np.array([c.close for c in candles])
        middle_band = np.convolve(closes, np.ones(self.period) / self.period, mode='valid')
        rolling_std = np.array([np.std(closes[i:i + self.period]) for i in range(len(closes) - self.period + 1)])
        upper_band = middle_band + (rolling_std * self.std_dev)
        lower_band = middle_band - (rolling_std * self.std_dev)

        pad_length = len(closes) - len(middle_band)
        middle_band_padded = np.pad(middle_band, (pad_length, 0), 'constant', constant_values=np.nan)
        upper_band_padded = np.pad(upper_band, (pad_length, 0), 'constant', constant_values=np.nan)
        lower_band_padded = np.pad(lower_band, (pad_length, 0), 'constant', constant_values=np.nan)

        return {'upper': upper_band_padded, 'middle': middle_band_padded, 'lower': lower_band_padded}

    def is_near_band(self, price, bands_dict, index, threshold=0.005):
        """
        Check if price is near any Bollinger Band.

        :param price: Current price
        :param bands_dict: Dictionary with 'upper', 'middle', 'lower' bands
        :param index: Index to check in the bands arrays
        :param threshold: Max distance as percentage of price to be considered "near" (default: 0.5%)
        :return: (is_near, band_type) tuple - is_near is boolean, band_type is 'upper', 'middle', 'lower' or None
        """
        if bands_dict is None or index >= len(bands_dict['upper']) or np.isnan(bands_dict['upper'][index]):
            return False, None

        upper = bands_dict['upper'][index]
        middle = bands_dict['middle'][index]
        lower = bands_dict['lower'][index]

        if price == 0:
            return False, None

        distances = {
            'upper': abs(price - upper) / price if upper != 0 else float('inf'),
            'middle': abs(price - middle) / price if middle != 0 else float('inf'),
            'lower': abs(price - lower) / price if lower != 0 else float('inf'),
        }
        min_band = min(distances, key=distances.get)

        # Log the calculations for debugging
        self.log_debug = getattr(self, 'log_debug', False)
        if hasattr(self, 'log_debug') and self.log_debug:
            print(f"BB Check: Price={price:.2f}, Upper={upper:.2f}, Middle={middle:.2f}, Lower={lower:.2f}")
            print(f"BB Distances: Upper={distances['upper']:.6f}, Middle={distances['middle']:.6f}, Lower={distances['lower']:.6f}, Threshold={threshold}")

        # Fix: Use min_band as the band type in the return statement
        return (distances[min_band] <= threshold, min_band if distances[min_band] <= threshold else None)


class RSI:
    def __init__(self, period=14):  # Ensure 'period' is accepted as a parameter
        self.period = period

    def calculate(self, candles):  # Ensure it returns the latest RSI value
        if len(candles) < self.period + 1:  # Need at least period + 1 prices for 1 delta
            return None

        closes = np.array([c.close for c in candles])
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        if len(gains) < self.period:  # Ensure enough data for the calculation
            return None

        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0  # Handle edge cases

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi


# --- Live Strategy Logic ---
class LiveStrategy:
    def __init__(self, config: Config, api_client: CoincatchApiClient):
        self.config = config
        self.api = api_client # Use self.api internally for consistency
        self.logger = logging.getLogger("LiveStrategy") # Initialize logger first

        self.candles_4h: List[CandleData] = []
        self.candles_1d: List[CandleData] = []

        # Base Strategy Indicators
        self.base_trend_detector = TrendLineDetector(min_points=config.RSI_PERIOD, max_lookback=100, min_strength=2)
        self.base_fib_circles = FibonacciCircles()
        self.base_bollinger = BollingerBands(period=config.BB_PERIOD, std_dev=config.BB_STD_DEV)
        self.base_rsi_indicator = RSI(period=config.RSI_PERIOD)

        # HMM Detector
        self.hmm_detector = HMMRegimeDetector(config=self.config)
        self.current_market_regime_idx: Optional[int] = None
        self.current_market_regime_name: str = "Unknown"

        # Log HMM status (self.log is now defined)
        if POMEGRANATE_AVAILABLE and hasattr(self.hmm_detector, 'model') and hasattr(self.hmm_detector, 'is_fitted') and not self.hmm_detector.is_fitted:
            self.log("HMM model is not fitted/loaded. Predictions will be None until trained or loaded.", "warning", Fore.YELLOW)
        elif not POMEGRANATE_AVAILABLE:
             self.log("Pomegranate not available, HMM features disabled for this session.", "critical", Fore.RED)


        # Position State
        self.position_open = False; self.entry_price = 0.0; self.position_size_btc = 0.0
        self.trade_type = None; self.current_stop_loss = 0.0; self.current_take_profit = 0.0
        self.allocated_margin_usd = 0.0; self.api_order_id = None; self.trailing_sl_activated = False
        self.breakeven_sl_activated = False; self.initial_sl_set_by_logic = 0.0; self.last_trade_time = 0

    def log(self, message, level="info", color=None, bold=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plain_log_message = f"STRATEGY - {message}" # For file logger

        # Log to the logger instance (goes to StreamHandler with basic formatting)
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(plain_log_message)

        # Pretty print to console
        console_msg = f"{timestamp} - STRATEGY - {message}"
        if color: console_msg = f"{color}{console_msg}{Style.RESET_ALL}"
        if bold: console_msg = f"{Style.BRIGHT}{console_msg}{Style.RESET_ALL}"
        print(console_msg)

    def _fetch_and_prepare_candles(self) -> bool:
        # Uses self.api (which is self.api_client passed in __init__)
        new_candles_4h = self.api.get_candles(
            symbol=self.config.SYMBOL,
            interval_from_config=self.config.KLINE_INTERVAL_PRIMARY,
            limit=self.config.MAX_CANDLES_4H
        )
        if new_candles_4h:
            self.candles_4h = new_candles_4h # Assuming get_candles returns sorted
        else:
            self.log(f"Failed to fetch 4H candles. Primary data missing.", "warning", Fore.YELLOW)
            self.candles_4h = []
            return False

        new_candles_1d = self.api.get_candles(
            symbol=self.config.SYMBOL,
            interval_from_config=self.config.KLINE_INTERVAL_SECONDARY,
            limit=self.config.MAX_CANDLES_1D
        )
        if new_candles_1d:
            self.candles_1d = new_candles_1d # Assuming get_candles returns sorted
        else:
            self.log(f"Failed to fetch 1D candles. Proceeding without 1D data for this cycle.", "warning", Fore.YELLOW)
            self.candles_1d = []
        return True

    def _update_account_and_position_status(self):
        position_data = self.api.get_position(self.config.SYMBOL, self.config.MARGIN_COIN)
        if position_data:
            size = float(position_data.get("total", position_data.get("contracts", 0)))
            entry_px_str = position_data.get("avgEntryPrice", position_data.get("entryPrice"))
            entry_px = float(entry_px_str) if entry_px_str is not None else 0.0
            side = position_data.get("holdSide", position_data.get("side", "")).lower()

            if size > 0 and entry_px > 0 and side in ["long", "short"]:
                if not self.position_open:
                    self.log(f"Found existing {side.upper()} position on exchange: {size:.8f} @ {entry_px:.2f}. Syncing state.", color=Fore.YELLOW)
                    self.position_open = True
                    self.entry_price = entry_px
                    self.position_size_btc = size
                    self.trade_type = "Long" if side == "long" else "Short"
                    if self.current_stop_loss == 0 or self.current_take_profit == 0: # If bot SL/TP are not set
                        sl, tp = self.calculate_sl_tp(entry_px, self.trade_type)
                        self.current_stop_loss = sl if sl is not None else 0.0
                        self.current_take_profit = tp if tp is not None else 0.0
                        self.initial_sl_set_by_logic = sl if sl is not None else 0.0
                        self.log(f"Safety SL/TP set for existing position: SL={self.current_stop_loss:.2f}, TP={self.current_take_profit:.2f}", color=Fore.YELLOW)
                else: # Position already known by bot, update if necessary
                    if abs(self.entry_price - entry_px) > 0.01 or abs(self.position_size_btc - size) > 1e-8 :
                         self.log(f"Updating known position details from API: Entry {self.entry_price:.2f}->{entry_px:.2f}, Size {self.position_size_btc:.8f}->{size:.8f}", color=Fore.CYAN)
                         self.entry_price = entry_px
                         self.position_size_btc = size
                self.position_open = True # Ensure it's True if valid position found
            else: # Position data from API is invalid (e.g., size 0 but was open in bot)
                if self.position_open:
                    self.log("Position reported by API has size 0 or is invalid. Resetting state.", color=Fore.YELLOW)
                    self._reset_position_state()
                self.position_open = False # Explicitly set to False
        else: # No position found by API
            if self.position_open:
                self.log("Position no longer found on exchange. Resetting state.", color=Fore.YELLOW)
                self._reset_position_state()
            self.position_open = False # Explicitly set to False


    def _reset_position_state(self):
        self.log("Resetting internal position state.", color=Fore.MAGENTA)
        self.position_open = False
        self.entry_price = 0.0
        self.position_size_btc = 0.0
        self.trade_type = None
        self.current_stop_loss = 0.0
        self.current_take_profit = 0.0
        self.allocated_margin_usd = 0.0
        self.api_order_id = None
        self.trailing_sl_activated = False
        self.breakeven_sl_activated = False
        self.initial_sl_set_by_logic = 0.0
        self.last_trade_time = time.time() # Cooldown starts now

    def calculate_rsi_value(self, candles: List[CandleData]) -> float:
        if not candles or len(candles) < self.config.RSI_PERIOD + 1:
            self.log(f"Not enough candles ({len(candles) if candles else 0}) for RSI. Need {self.config.RSI_PERIOD + 1}. Returning neutral 50.", level="debug")
            return 50.0 # Return float
        rsi_val = self.base_rsi_indicator.calculate(candles) # Uses base_rsi_indicator
        if rsi_val is None:
            self.log(f"RSI calculation returned None. Returning neutral 50.", level="debug")
            return 50.0 # Return float
        return rsi_val

    def calculate_standard_fib_levels(self, candles_for_tf: List[CandleData], current_price: float, lookback_period=50) -> Dict[str, List[float]]:
        if not candles_for_tf or len(candles_for_tf) < lookback_period :
            self.log(f"Std Fib: Not enough candles ({len(candles_for_tf)}) for lookback {lookback_period}. Skipping.", level="debug")
            return {"supports": [], "resistances": []}
        recent_candles = candles_for_tf[-lookback_period:]
        if not recent_candles:
            self.log("Std Fib: recent_candles list is empty. Skipping.", level="debug")
            return {"supports": [], "resistances": []}
        highs = np.array([c.high for c in recent_candles])
        lows = np.array([c.low for c in recent_candles])
        if highs.size == 0 or lows.size == 0:
            self.log("Std Fib: highs or lows array is empty. Skipping.", level="debug")
            return {"supports": [], "resistances": []}
        period_high = np.max(highs)
        period_low = np.min(lows)
        if period_high <= period_low or abs(period_high - period_low) / max(period_high, period_low, 1e-9) < 0.001: # Check for valid range
            self.log(f"Std Fib: Swing range too small or invalid. High={period_high:.2f}, Low={period_low:.2f}. Skipping.", level="debug")
            return {"supports": [], "resistances": []}
        price_range = period_high - period_low
        idx_min_in_recent = np.argmin(lows)
        idx_max_in_recent = np.argmax(highs)
        primary_move_is_up = idx_max_in_recent > idx_min_in_recent
        fib_ratios_retracement = [0.236, 0.382, 0.5, 0.618, 0.786]
        supports = []
        resistances = []
        if primary_move_is_up:
            for ratio in fib_ratios_retracement:
                supports.append(period_high - (price_range * ratio))
            resistances.append(period_high)
        else:
            for ratio in fib_ratios_retracement:
                resistances.append(period_low + (price_range * ratio))
            supports.append(period_low)
        final_supports = sorted([s for s in supports if s > 0 and s < current_price], reverse=True)[:3]
        final_resistances = sorted([r for r in resistances if r > 0 and r > current_price])[:3]
        self.log(f"Std Fib ({lookback_period} candles): Swing Low={period_low:.2f}, Swing High={period_high:.2f}, PrimaryMoveUP={primary_move_is_up}", level="debug")
        if final_supports: self.log(f"  Std Fib Supports (closest below current): {[f'{s:.2f}' for s in final_supports]}", level="debug")
        if final_resistances: self.log(f"  Std Fib Resistances (closest above current): {[f'{r:.2f}' for r in final_resistances]}", level="debug")
        return {"supports": final_supports, "resistances": final_resistances}

    def calculate_sl_tp_based_on_backtest(self, entry_price: float, direction: str) -> tuple[Optional[float], Optional[float]]:
        self.log(f"Calculating SL/TP (backtest logic) for {direction} from entry {entry_price:.2f}", color=Fore.BLUE)
        if not self.candles_4h or len(self.candles_4h) < self.base_trend_detector.min_points: # Use base_trend_detector
            self.log("Not enough 4H candles for Fib Circle SL/TP. Fallback.", "warning", Fore.YELLOW)
            return self.calculate_sl_tp_percentage_fallback(entry_price, direction)

        trend_lines_for_sl = self.base_trend_detector.detect_trend_lines(self.candles_4h)
        fib_circles_for_sl = self.base_fib_circles.generate_circles(trend_lines_for_sl)

        idx_within_lookback = len(self.candles_4h) -1 # current_index relative to self.candles_4h
        support_levels, resistance_levels = self.base_fib_circles.get_support_resistance_levels(
            fib_circles_for_sl,
            idx_within_lookback
        )

        stop_loss_price = None
        if direction == 'Long':
            valid_supports = [level for level in support_levels if level < entry_price and level > 0]
            if valid_supports:
                stop_loss_price = max(valid_supports)
                self.log(f"SL (Long) based on Fib Circle Support: {stop_loss_price:.2f}", "debug", Fore.BLUE)
            else: # Fallback if no valid Fib Circle support
                stop_loss_price = entry_price * (1 - 0.015) # Example: 1.5% SL
                self.log(f"SL (Long) no valid Fib Circle support. Using % fallback: {stop_loss_price:.2f}", "debug", Fore.YELLOW)
        else:  # Short
            valid_resistances = [level for level in resistance_levels if level > entry_price]
            if valid_resistances:
                stop_loss_price = min(valid_resistances)
                self.log(f"SL (Short) based on Fib Circle Resistance: {stop_loss_price:.2f}", "debug", Fore.BLUE)
            else: # Fallback
                stop_loss_price = entry_price * (1 + 0.015)
                self.log(f"SL (Short) no valid Fib Circle resistance. Using % fallback: {stop_loss_price:.2f}", "debug", Fore.YELLOW)

        take_profit_price = None
        std_fib_levels = self.calculate_standard_fib_levels(self.candles_4h, entry_price) # Use 4H for TP as well

        if direction == 'Long':
            if std_fib_levels and std_fib_levels["resistances"]:
                valid_resistances_tp = [r for r in std_fib_levels["resistances"] if r > entry_price * 1.002] # TP slightly away
                if valid_resistances_tp:
                    take_profit_price = min(valid_resistances_tp)
                    self.log(f"TP (Long) based on Standard Fib Resistance: {take_profit_price:.2f}", "debug", Fore.BLUE)
        else:  # Short
            if std_fib_levels and std_fib_levels["supports"]:
                valid_supports_tp = [s for s in std_fib_levels["supports"] if 0 < s < entry_price * 0.998] # TP slightly away
                if valid_supports_tp:
                    take_profit_price = max(valid_supports_tp)
                    self.log(f"TP (Short) based on Standard Fib Support: {take_profit_price:.2f}", "debug", Fore.BLUE)

        # Fallback to R:R if TP not found or invalid
        if take_profit_price is None or \
           (direction == "Long" and (stop_loss_price is None or take_profit_price <= entry_price)) or \
           (direction == "Short" and (stop_loss_price is None or take_profit_price >= entry_price)):
            risk_reward_ratio = 2.0
            if stop_loss_price is not None: # Ensure SL is set before calculating R:R based TP
                if direction == 'Long' and entry_price > stop_loss_price:
                    risk = entry_price - stop_loss_price
                    take_profit_price = entry_price + (risk * risk_reward_ratio)
                    self.log(f"TP (Long) No valid Std Fib or too close. Using R:R fallback: {take_profit_price:.2f}", "debug", Fore.YELLOW)
                elif direction == 'Short' and entry_price < stop_loss_price:
                    risk = stop_loss_price - entry_price
                    take_profit_price = entry_price - (risk * risk_reward_ratio)
                    self.log(f"TP (Short) No valid Std Fib or too close. Using R:R fallback: {take_profit_price:.2f}", "debug", Fore.YELLOW)
                else: # SL is on the wrong side of entry, cannot calculate R:R
                    self.log(f"Invalid SL {stop_loss_price} for {direction} from {entry_price} for R:R TP calc. TP remains None.", "warning", Fore.RED)
                    take_profit_price = None # Ensure it's None
            else: # SL itself was None
                 self.log("SL is None, cannot calculate R:R based TP. TP remains None.", "warning", Fore.YELLOW)
                 take_profit_price = None

        if stop_loss_price is not None and take_profit_price is not None:
             risk_amount = abs(stop_loss_price - entry_price)
             reward_amount = abs(take_profit_price - entry_price)
             current_rr = reward_amount / risk_amount if risk_amount > 0 else 0
             self.log(f"Final SL/TP from backtest logic: SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, R:R={current_rr:.2f}", "debug", Fore.BLUE)
        elif stop_loss_price is not None:
             self.log(f"Final SL/TP from backtest logic: SL={stop_loss_price:.2f}, TP=N/A (Calculation failed or invalid)", "debug", Fore.BLUE)
        else:
             self.log(f"Final SL/TP from backtest logic: SL=N/A, TP=N/A (Calculation failed or invalid)", "debug", Fore.BLUE)

        return stop_loss_price, take_profit_price

    def _update_dynamic_sl(self, current_price: float):
        if not self.position_open or not self.trade_type or self.entry_price == 0 or self.current_stop_loss == 0: # SL must be initially set
            return

        asset_profit_pct = ((current_price - self.entry_price) / self.entry_price) if self.trade_type == "Long" else \
                           ((self.entry_price - current_price) / self.entry_price)

        # Breakeven logic
        if not self.breakeven_sl_activated and asset_profit_pct >= self.config.BREAKEVEN_PROFIT_PCT:
            new_sl = self.entry_price * (1.0005 if self.trade_type == "Long" else 0.9995) # Small profit buffer
            if (self.trade_type == "Long" and new_sl > self.current_stop_loss) or \
               (self.trade_type == "Short" and new_sl < self.current_stop_loss):
                self.current_stop_loss = new_sl
                self.breakeven_sl_activated = True
                self.log(f"Breakeven SL activated at {asset_profit_pct:.2%} profit. SL moved to {new_sl:.2f}", color=Fore.GREEN, bold=True)

        # Trailing stop logic
        if not self.trailing_sl_activated and asset_profit_pct >= self.config.TRAILING_STOP_ACTIVATION_PROFIT_PCT:
            self.trailing_sl_activated = True
            self.log(f"Trailing SL activated at {asset_profit_pct:.2%} profit", color=Fore.GREEN, bold=True)

        if self.trailing_sl_activated:
            potential_new_sl = current_price * (1 - self.config.TRAILING_STOP_DISTANCE_PCT) if self.trade_type == "Long" else \
                               current_price * (1 + self.config.TRAILING_STOP_DISTANCE_PCT)

            # If breakeven is also active, ensure trailing SL doesn't move SL back into loss
            if self.breakeven_sl_activated:
                breakeven_level = self.entry_price * (1.0005 if self.trade_type == "Long" else 0.9995)
                if self.trade_type == "Long": potential_new_sl = max(potential_new_sl, breakeven_level)
                else: potential_new_sl = min(potential_new_sl, breakeven_level)

            # Only move SL if it's more favorable (further in profit direction)
            if (self.trade_type == "Long" and potential_new_sl > self.current_stop_loss) or \
               (self.trade_type == "Short" and potential_new_sl < self.current_stop_loss):
                prev_sl = self.current_stop_loss
                self.current_stop_loss = potential_new_sl
                self.log(f"Trailing SL updated: {prev_sl:.2f} -> {self.current_stop_loss:.2f} (Trailing from price {current_price:.2f})", color=Fore.GREEN)


    def _log_market_analysis(self, timeframe_name: str, current_price: float, candles_for_tf: List[CandleData], current_idx: int):
        self.log(f"----- {timeframe_name} Analysis (Price: {current_price:.2f}) -----", color=Fore.CYAN)
        rsi_value = self.calculate_rsi_value(candles_for_tf) # Uses base_rsi_indicator
        self.log(f"RSI ({self.config.RSI_PERIOD}): {rsi_value:.2f}", color=Fore.CYAN)

        bbands = self.base_bollinger.calculate(candles_for_tf) # Uses base_bollinger
        latest_valid_bb_idx = -1
        if bbands and bbands['upper'].size > 0 : # Check if not empty
             latest_valid_idx_arr = np.where(~np.isnan(bbands['upper']))[0]
             if latest_valid_idx_arr.size > 0:
                 latest_valid_bb_idx = latest_valid_idx_arr[-1]

        if latest_valid_bb_idx != -1:
            self.log(f"Bollinger ({self.config.BB_PERIOD},{self.config.BB_STD_DEV}): L={bbands['lower'][latest_valid_bb_idx]:.2f} M={bbands['middle'][latest_valid_bb_idx]:.2f} U={bbands['upper'][latest_valid_bb_idx]:.2f}", color=Fore.CYAN)
        else: self.log("Bollinger Bands: Not enough data or all NaN values.", color=Fore.YELLOW)

        try:
            std_fib_result = self.calculate_standard_fib_levels(candles_for_tf, current_price)
            if std_fib_result:
                std_fib_supports, std_fib_resistances = std_fib_result["supports"], std_fib_result["resistances"]
                if std_fib_supports: self.log(f"Standard Fib Supports: {[f'{s:.2f}' for s in std_fib_supports]}", color=Fore.GREEN)
                else: self.log("No Standard Fibonacci Support Levels calculated.", color=Fore.YELLOW)
                if std_fib_resistances: self.log(f"Standard Fib Resistances: {[f'{r:.2f}' for r in std_fib_resistances]}", color=Fore.RED)
                else: self.log("No Standard Fibonacci Resistance Levels calculated.", color=Fore.YELLOW)
            else: self.log("Standard Fibonacci analysis did not return levels.", color=Fore.YELLOW)
        except Exception as e:
            self.log(f"Error during Standard Fibonacci analysis for {timeframe_name}: {e}", level="error", color=Fore.RED)
            self.logger.error(f"Full traceback for Standard Fibonacci error on {timeframe_name}:", exc_info=True)

        try:
            lookback = self.base_trend_detector.max_lookback # Use base_trend_detector
            if len(candles_for_tf) >= lookback:
                trend_lines = self.base_trend_detector.detect_trend_lines(candles_for_tf[-lookback:])
                if trend_lines:
                    self.log(f"Detected {len(trend_lines)} trendline(s) on {timeframe_name} (Lookback: {lookback})", color=Fore.BLUE)
                    adjusted_current_idx = lookback - 1 # Index for fib circles is relative to the slice used
                    fib_circles_data = self.base_fib_circles.generate_circles(trend_lines, max_circles=3) # Use base_fib_circles
                    if fib_circles_data:
                        support_levels, resistance_levels = self.base_fib_circles.get_support_resistance_levels(fib_circles_data, adjusted_current_idx)
                        if support_levels: self.log(f"Fib Circle Supports: {[f'{s:.2f}' for s in support_levels[:3]]}", color=Fore.GREEN)
                        else: self.log("No Fibonacci Circle Support Levels calculated.", color=Fore.YELLOW)
                        if resistance_levels: self.log(f"Fib Circle Resistances: {[f'{r:.2f}' for r in resistance_levels[:3]]}", color=Fore.RED)
                        else: self.log("No Fibonacci Circle Resistance Levels calculated.", color=Fore.YELLOW)
                    else: self.log("No Fibonacci circles generated.", color=Fore.YELLOW)
                else: self.log(f"No significant trendlines detected on {timeframe_name}.", color=Fore.YELLOW)
            else: self.log(f"Skipping Trend/Fib Circle analysis on {timeframe_name} (need {lookback} candles, have {len(candles_for_tf)}).", color=Fore.YELLOW)
        except Exception as e:
            self.log(f"Error during Trend/Fib Circle analysis for {timeframe_name}: {e}", level="error", color=Fore.RED)
            self.logger.error(f"Full traceback for Trend/Fib Circle error on {timeframe_name}:", exc_info=True)
        self.log(f"----- End {timeframe_name} Analysis -----\n", color=Fore.CYAN)

    def _log_signal_scores(self, long_4h, short_4h, reasons_4h, long_1d, short_1d, reasons_1d, combined_long, combined_short):
        self.log("\n===== TRADING SIGNAL ANALYSIS =====", color=Fore.MAGENTA, bold=True)
        self.log(f"Minimum Score for Entry Threshold: {self.config.MIN_SCORE_ENTRY}", color=Fore.MAGENTA)
        self.log("--- 4H Timeframe Signal Calculation ---", color=Fore.CYAN)
        self.log(f"  Raw Scores: Long={long_4h:.2f}, Short={short_4h:.2f}", color=Fore.CYAN)
        if reasons_4h: self.log("  Contributing Reasons (4H):", color=Fore.CYAN); [self.log(f"    • {r}", color=Fore.CYAN) for r in reasons_4h]
        else: self.log("  No specific 4H signals triggered scoring.", color=Fore.CYAN)
        self.log("--- 1D Timeframe Signal Calculation ---", color=Fore.CYAN)
        self.log(f"  Raw Scores: Long={long_1d:.2f}, Short={short_1d:.2f}", color=Fore.CYAN)
        if reasons_1d: self.log("  Contributing Reasons (1D):", color=Fore.CYAN); [self.log(f"    • {r}", color=Fore.CYAN) for r in reasons_1d]
        else: self.log("  No specific 1D signals triggered scoring.", color=Fore.CYAN)
        self.log(f"--- Combined Signal (Weights: 4H={0.6}, 1D={0.4}) ---", color=Fore.YELLOW)
        self.log(f"  FINAL Combined Scores: Long={combined_long:.2f}, Short={combined_short:.2f}", color=Fore.YELLOW, bold=True)

        action_text = "NO TRADE"
        action_color = Fore.YELLOW
        if combined_long >= self.config.MIN_SCORE_ENTRY and combined_long > combined_short:
            action_text = f"LONG. Score ({combined_long:.2f}) >= Threshold and > Short ({combined_short:.2f})."
            action_color = Fore.GREEN
        elif combined_short >= self.config.MIN_SCORE_ENTRY and combined_short > combined_long:
            action_text = f"SHORT. Score ({combined_short:.2f}) >= Threshold and > Long ({combined_long:.2f})."
            action_color = Fore.RED
        else:
            action_text = f"NO TRADE. Long {combined_long:.2f}, Short {combined_short:.2f}, Threshold {self.config.MIN_SCORE_ENTRY}"

        self.log(f"  Potential Action (Base Strategy): {action_text}", color=action_color, bold=True)
        self.log("======================================\n", color=Fore.MAGENTA, bold=True)

    def combine_signals(self, long_4h: float, short_4h: float, long_1d: float, short_1d: float) -> tuple[float, float]:
        weight_4h = 0.6; weight_1d = 0.4
        return (long_4h * weight_4h) + (long_1d * weight_1d), (short_4h * weight_4h) + (short_1d * weight_1d)

    def _apply_price_precision(self, price: Optional[float], direction_for_rounding: Optional[str] = None, is_sl: bool = False) -> Optional[float]:
        if price is None: return None
        market_details = self.api.get_market_details(self.config.SYMBOL) # Use unified_symbol from api
        price_tick_size_str = market_details.get('precision', {}).get('price') if market_details else None

        if price_tick_size_str is not None:
            try:
                price_tick_size = float(price_tick_size_str)
                if price_tick_size > 0:
                    if direction_for_rounding == "Long":
                        return math.floor(price / price_tick_size) * price_tick_size if is_sl else math.ceil(price / price_tick_size) * price_tick_size
                    elif direction_for_rounding == "Short":
                        return math.ceil(price / price_tick_size) * price_tick_size if is_sl else math.floor(price / price_tick_size) * price_tick_size
                    else: # General rounding (e.g., for entry price if needed, though market fills determine that)
                        return round(price / price_tick_size) * price_tick_size
            except ValueError:
                self.log(f"Could not convert price tick size '{price_tick_size_str}' to float. Using default rounding.", "warning", Fore.YELLOW)

        # Fallback default rounding if precision info is unavailable or invalid (e.g. 1 decimal for BTCUSDT)
        return round(price, 1)


    def calculate_sl_tp(self, entry_price: float, direction: str, risk_factor_sl=0.015, reward_factor_tp=0.03) -> tuple[Optional[float], Optional[float]]:
        sl_price, tp_price = None, None
        try:
            sl_bt, tp_bt = self.calculate_sl_tp_based_on_backtest(entry_price, direction)
            valid_bt_sl = (sl_bt is not None) and ((direction == "Long" and sl_bt < entry_price) or (direction == "Short" and sl_bt > entry_price))
            valid_bt_tp = (tp_bt is not None) and ((direction == "Long" and tp_bt > entry_price) or (direction == "Short" and tp_bt < entry_price))

            if valid_bt_sl and valid_bt_tp:
                sl_price, tp_price = sl_bt, tp_bt
                self.log(f"Using backtest-derived SL/TP: SL={sl_price:.2f}, TP={tp_price:.2f}", "debug", Fore.CYAN)
            else:
                if not valid_bt_sl: self.log("Backtest SL was invalid or None.", "debug", Fore.YELLOW)
                if not valid_bt_tp: self.log("Backtest TP was invalid or None.", "debug", Fore.YELLOW)
                self.log("Reverting to percentage-based SL/TP for missing/invalid parts.", "debug", Fore.YELLOW)
        except Exception as e: self.log(f"Error in backtest-based SL/TP calculation: {e}. Using fallback.", "error", Fore.RED, exc_info=True)

        # Fallback for any part (SL or TP) that wasn't successfully determined by backtest logic
        if sl_price is None or tp_price is None: # If either is still None
            sl_fb, tp_fb = self.calculate_sl_tp_percentage_fallback(entry_price, direction, risk_factor_sl, reward_factor_tp)
            if sl_price is None: sl_price = sl_fb
            if tp_price is None: tp_price = tp_fb
            self.log(f"Using percentage-based SL/TP (or parts): SL={sl_price:.2f}, TP={tp_price:.2f}", "debug", Fore.CYAN)

        # Apply exchange precision requirements AT THE END
        final_sl_price = self._apply_price_precision(sl_price, direction_for_rounding=direction, is_sl=True)
        final_tp_price = self._apply_price_precision(tp_price, direction_for_rounding=direction, is_sl=False)

        self.log(f"Final Calculated & Precision-Adjusted SL: {final_sl_price}, TP: {final_tp_price} for {direction} entry {entry_price:.2f}", color=Fore.CYAN)
        return final_sl_price, final_tp_price

    def calculate_sl_tp_percentage_fallback(self, entry_price: float, direction: str, risk_sl_factor=0.015, reward_tp_factor=0.03) -> tuple[float, float]:
        if direction == "Long":
            sl = entry_price * (1 - risk_sl_factor)
            tp = entry_price * (1 + reward_tp_factor)
        else: # Short
            sl = entry_price * (1 + risk_sl_factor)
            tp = entry_price * (1 - reward_tp_factor)
        # Precision is applied by the calling method (calculate_sl_tp)
        return sl, tp

    def _apply_amount_precision(self, amount: float) -> Optional[float]:
        market_details = self.api.get_market_details(self.config.SYMBOL) # Use unified_symbol from API client
        amount_tick_size_str = market_details.get('precision', {}).get('amount') if market_details else None

        if amount_tick_size_str is not None:
            try:
                amount_tick_size = float(amount_tick_size_str)
                if amount_tick_size > 0:
                    # For order quantities, always round down (floor) to the nearest valid step size
                    return math.floor(amount / amount_tick_size) * amount_tick_size
            except ValueError:
                 self.log(f"Could not convert amount tick size '{amount_tick_size_str}' to float. Using default rounding for amount.", "warning", Fore.YELLOW)

        # Fallback if precision info is unavailable or invalid (e.g. 3 decimals for BTC amount)
        return round(amount, 3) # Adjust default decimals as needed for the asset


    def calculate_position_size_and_margin(self, current_price: float, available_balance_usd: float) -> tuple[Optional[float], Optional[float]]:
        if available_balance_usd <= 0 or current_price <= 0: return None, None

        margin_usd_based_on_pct = available_balance_usd * self.config.MARGIN_PERCENTAGE
        margin_for_trade_usd = min(margin_usd_based_on_pct, self.config.MAX_LOSS_PER_TRADE_USD)

        if margin_for_trade_usd <= 0.01: # Margin too small
            self.log(f"Calculated margin for trade ${margin_for_trade_usd:.2f} is too small.", "debug")
            return None, None

        pos_size_base_raw = (margin_for_trade_usd * self.config.LEVERAGE) / current_price

        market_details = self.api.get_market_details(self.config.SYMBOL)
        min_qty_from_exchange = self.config.MIN_QUANTITY # Default from config

        if market_details:
            limits_amount_min_str = market_details.get('limits',{}).get('amount',{}).get('min')
            if limits_amount_min_str is not None:
                min_qty_from_exchange = float(limits_amount_min_str)

        pos_size_base_adjusted = pos_size_base_raw
        if pos_size_base_raw < min_qty_from_exchange:
            self.log(f"Calculated size {pos_size_base_raw:.8f} < min exchange qty {min_qty_from_exchange:.8f}. Attempting to use min qty.", "debug")
            pos_size_base_adjusted = min_qty_from_exchange
            required_margin_for_min_size = (pos_size_base_adjusted * current_price) / self.config.LEVERAGE if self.config.LEVERAGE > 0 else float('inf')
            if required_margin_for_min_size > available_balance_usd or required_margin_for_min_size > self.config.MAX_LOSS_PER_TRADE_USD:
                self.log(f"Cannot afford min qty ({pos_size_base_adjusted:.8f}). Required margin ${required_margin_for_min_size:.2f} vs Avail ${available_balance_usd:.2f} & MaxLossCap ${self.config.MAX_LOSS_PER_TRADE_USD:.2f}", "warning", Fore.YELLOW)
                return None, None
            margin_for_trade_usd = required_margin_for_min_size

        final_pos_size_base = self._apply_amount_precision(pos_size_base_adjusted)

        if final_pos_size_base is None or final_pos_size_base < min_qty_from_exchange:
            self.log(f"Position size after precision ({final_pos_size_base}) is less than min exchange qty ({min_qty_from_exchange}). Cannot trade.", "warning", Fore.YELLOW)
            return None, None

        final_margin_usd = (final_pos_size_base * current_price) / self.config.LEVERAGE if self.config.LEVERAGE > 0 else float('inf')
        if final_margin_usd > available_balance_usd :
            self.log(f"Final margin ${final_margin_usd:.2f} for precise size {final_pos_size_base:.8f} exceeds available ${available_balance_usd:.2f}", "warning", Fore.YELLOW)
            return None, None

        self.log(f"Calculated Final Pos Size: {final_pos_size_base:.8f} {self.config.SYMBOL.split('_')[0][:3]}, Margin: ${final_margin_usd:.2f} USDT", "debug")
        return final_pos_size_base, final_margin_usd

    def _update_market_regime(self):
        if not POMEGRANATE_AVAILABLE :
            self.current_market_regime_idx = None
            self.current_market_regime_name = "HMM Disabled"
            if not POMEGRANATE_AVAILABLE: self.log("Pomegranate not available, HMM disabled.", "warning", Fore.RED)
            return

        if not hasattr(self.hmm_detector, 'model') or not self.hmm_detector.model or \
           not hasattr(self.hmm_detector, 'scaler') or not self.hmm_detector.scaler or \
           not hasattr(self.hmm_detector, 'is_fitted') or not self.hmm_detector.is_fitted:
            self.current_market_regime_idx = None
            self.current_market_regime_name = "HMM Not Ready"
            self.log("HMM model/scaler not loaded or model not fitted. Regime detection skipped.", "warning", Fore.YELLOW)
            return

        min_candles_for_hmm_features = max(self.config.HMM_RSI_PERIOD, self.config.HMM_MACD_SLOW + self.config.HMM_MACD_SIGNAL, self.config.HMM_BB_PERIOD, self.config.HMM_ATR_PERIOD, self.config.HMM_VOLATILITY_WINDOW) + 25

        candles_for_hmm = self.candles_4h
        if not candles_for_hmm or len(candles_for_hmm) < min_candles_for_hmm_features:
             self.log(f"Not enough 4H candles ({len(candles_for_hmm)}) for HMM prediction. Need ~{min_candles_for_hmm_features}. Skipping.", "warning", Fore.YELLOW)
             self.current_market_regime_idx = None
             self.current_market_regime_name = "Insufficient Data for HMM"
             return

        predicted_regime = self.hmm_detector.predict_regime(candles_for_hmm)
        if predicted_regime is not None:
            self.current_market_regime_idx = predicted_regime
            safe_regime_idx = predicted_regime % len(self.config.HMM_REGIME_NAMES)
            self.current_market_regime_name = self.config.HMM_REGIME_NAMES[safe_regime_idx]
            regime_color = self.config.HMM_REGIME_COLORS[safe_regime_idx]
            self.log(f"Current HMM Market Regime: {self.current_market_regime_name} (Index: {self.current_market_regime_idx})",
                     color=regime_color, bold=True)
        else:
            self.current_market_regime_idx = None
            self.current_market_regime_name = "HMM Prediction Failed"
            self.log("HMM Regime prediction failed for current step.", "warning", Fore.YELLOW)


    def run_strategy_step(self):
        self.log("="*50, color=Fore.MAGENTA, bold=True)
        self.log("Running Strategy Step", color=Fore.MAGENTA, bold=True)

        if not self._fetch_and_prepare_candles():
            self.log("Halting step due to candle fetch failure.", "error", Fore.RED); return
        if not self.candles_4h:
            self.log("No 4H candles available after fetch. Halting step.", "error", Fore.RED); return

        current_price_4h = self.candles_4h[-1].close; current_idx_4h = len(self.candles_4h) - 1
        self.log(f"Current 4H Price: {current_price_4h:.2f} USDT", color=Fore.CYAN)
        current_price_1d = self.candles_1d[-1].close if self.candles_1d else current_price_4h
        current_idx_1d = len(self.candles_1d) - 1 if self.candles_1d else -1

        self._update_market_regime()
        self._update_account_and_position_status()

        balance_info = self.api.get_account_balance(self.config.SYMBOL, self.config.MARGIN_COIN, self.config.PRODUCT_TYPE)
        available_balance_usd = 0.0
        if balance_info and balance_info.get("data"):
            acc_data = balance_info["data"]
            available_balance_usd = float(acc_data.get("available", 0))
            equity_bal = float(acc_data.get("equity", 0))
            unrealized_pnl = float(acc_data.get("unrealizedPL", 0.0))
            self.log(f"ACCOUNT: Equity={equity_bal:.2f}, Avail={available_balance_usd:.2f}, UPNL={unrealized_pnl:.2f} USDT", color=Fore.YELLOW, bold=True)
        else:
            self.log(f"Failed to get account balance or parse data. Response: {balance_info}", "error", Fore.RED); return

        if self.position_open:
            pos_data_live = self.api.get_position(self.config.SYMBOL, self.config.MARGIN_COIN)
            mark_px_str = str(pos_data_live.get("marketPrice", "N/A")) if pos_data_live else "N/A"
            liq_px_str = str(pos_data_live.get("liquidationPrice", "N/A")) if pos_data_live else "N/A"
            pos_upnl_str = str(pos_data_live.get("unrealizedPL", "N/A")) if pos_data_live else "N/A"
            self.log(f"OPEN POSITION: {self.trade_type} | Size: {self.position_size_btc:.6f} | Entry: {self.entry_price:.2f} | "
                     f"MarginAlloc: {self.allocated_margin_usd:.2f} | MarkPx: {mark_px_str} | LiqPx: {liq_px_str} | UPNL: {pos_upnl_str} | "
                     f"Bot SL: {self.current_stop_loss:.2f} | Bot TP: {self.current_take_profit:.2f}",
                     color=(Fore.GREEN if self.trade_type == "Long" else Fore.RED), bold=True)
        else: self.log("No open position.", color=Fore.YELLOW)
        self.log("-" * 30, color=Fore.YELLOW)

        self._log_market_analysis("4H", current_price_4h, self.candles_4h, current_idx_4h)
        if self.candles_1d and current_idx_1d != -1:
            self._log_market_analysis("1D", current_price_1d, self.candles_1d, current_idx_1d)

        time_since_last_trade = time.time() - self.last_trade_time
        if not self.position_open and time_since_last_trade >= self.config.COOLDOWN_PERIOD_SECONDS:
            self.log("Checking entry conditions (cooldown passed)...", color=Fore.BLUE)
            price_1d_sig = self.candles_1d[-1].close if self.candles_1d and current_idx_1d!=-1 else current_price_4h
            idx_1d_sig = current_idx_1d if self.candles_1d and current_idx_1d!=-1 else -1
            candles_1d_sig = self.candles_1d if self.candles_1d and current_idx_1d!=-1 else []

            long_4h, short_4h, reasons_4h = self.check_entry_conditions(current_price_4h, current_idx_4h, self.candles_4h)
            long_1d, short_1d, reasons_1d = (0.0, 0.0, [])
            if candles_1d_sig: long_1d, short_1d, reasons_1d = self.check_entry_conditions(price_1d_sig, idx_1d_sig, candles_1d_sig)

            combined_long, combined_short = self.combine_signals(long_4h, short_4h, long_1d, short_1d)
            self._log_signal_scores(long_4h, short_4h, reasons_4h, long_1d, short_1d, reasons_1d, combined_long, combined_short)

            base_trade_direction = None
            if combined_long >= self.config.MIN_SCORE_ENTRY and combined_long > combined_short: base_trade_direction = "Long"
            elif combined_short >= self.config.MIN_SCORE_ENTRY and combined_short > combined_long: base_trade_direction = "Short"

            final_trade_direction = None
            hmm_allows_trade = True
            if POMEGRANATE_AVAILABLE and hasattr(self.hmm_detector, 'is_fitted') and self.hmm_detector.is_fitted and self.current_market_regime_idx is not None:
                if base_trade_direction == "Long":
                    if not (self.current_market_regime_name in ["Bullish Trend", "High-Volatility Breakout", "Pullback/Consolidation"]):
                        hmm_allows_trade = False
                elif base_trade_direction == "Short":
                    if not (self.current_market_regime_name in ["Bearish Trend", "High-Volatility Breakout", "Fakeout/Liquidity Sweep"]):
                        hmm_allows_trade = False

                if not hmm_allows_trade:
                     self.log(f"Base strategy signals {base_trade_direction}, but HMM regime '{self.current_market_regime_name}' is unfavorable. Skipping.", color=Fore.YELLOW)

            if base_trade_direction and hmm_allows_trade:
                final_trade_direction = base_trade_direction

            if final_trade_direction:
                self.log(f"FINAL TRADE SIGNAL: {final_trade_direction.upper()} (Base Score: {combined_long if final_trade_direction=='Long' else combined_short:.2f}, HMM: {self.current_market_regime_name})",
                         color=(Fore.GREEN if final_trade_direction=="Long" else Fore.RED), bold=True)

                pos_size_btc, margin_usd = self.calculate_position_size_and_margin(current_price_4h, available_balance_usd)
                if pos_size_btc and margin_usd and pos_size_btc > 0 and margin_usd > 0:
                    sl_price, tp_price = self.calculate_sl_tp(current_price_4h, final_trade_direction)

                    if sl_price is None or tp_price is None or \
                       (final_trade_direction == "Long" and (sl_price >= current_price_4h or tp_price <= current_price_4h)) or \
                       (final_trade_direction == "Short" and (sl_price <= current_price_4h or tp_price >= current_price_4h)):
                        self.log(f"Invalid SL/TP: Entry={current_price_4h:.2f}, SL={sl_price}, TP={tp_price} for {final_trade_direction}. Skipping.", "error", Fore.RED)
                    else:
                        ccxt_side = "buy" if final_trade_direction == "Long" else "sell"
                        order_sl = float(sl_price)
                        order_tp = float(tp_price)

                        order_info = self.api.place_market_order(self.config.SYMBOL, ccxt_side, pos_size_btc, order_sl, order_tp)
                        if order_info and order_info.get('id'):
                            self.log(f"{final_trade_direction.upper()} ORDER PLACED: ID {order_info['id']}", color=(Fore.GREEN if final_trade_direction=="Long" else Fore.RED), bold=True)
                            self.position_open = True
                            filled_price_str = order_info.get('average', order_info.get('price'))
                            self.entry_price = float(filled_price_str) if filled_price_str is not None else current_price_4h
                            filled_amount_str = order_info.get('filled', order_info.get('amount'))
                            self.position_size_btc = float(filled_amount_str) if filled_amount_str is not None else pos_size_btc

                            self.trade_type = final_trade_direction;
                            self.current_stop_loss = sl_price
                            self.current_take_profit = tp_price
                            self.allocated_margin_usd = margin_usd; self.api_order_id = order_info['id']
                            self.last_trade_time = time.time(); self.trailing_sl_activated = False; self.breakeven_sl_activated = False
                            self.initial_sl_set_by_logic = sl_price
                            self.log(f"NEW POS: {self.trade_type} {self.position_size_btc:.8f} @ {self.entry_price:.2f}, SL={self.current_stop_loss:.2f}, TP={self.current_take_profit:.2f}, Margin={margin_usd:.2f}", color=Fore.CYAN)
                        else: self.log(f"FAILED TO PLACE {final_trade_direction.upper()} ORDER. Response: {order_info}", "error", Fore.RED, bold=True)
                else: self.log(f"Invalid size/margin from calc: Size={pos_size_btc}, Margin={margin_usd}. Skipping trade.", "warning", Fore.YELLOW)

        elif not self.position_open and time_since_last_trade < self.config.COOLDOWN_PERIOD_SECONDS:
             self.log(f"In cooldown. Remaining: {self.config.COOLDOWN_PERIOD_SECONDS - time_since_last_trade:.0f}s", color=Fore.YELLOW)

        elif self.position_open:
            self._update_dynamic_sl(current_price_4h)
            closed_this_cycle = False
            if self.current_stop_loss > 0 and (
               (self.trade_type == "Long" and current_price_4h <= self.current_stop_loss) or \
               (self.trade_type == "Short" and current_price_4h >= self.current_stop_loss)):
                self.log(f"STOP LOSS HIT for {self.trade_type} at {current_price_4h:.2f} (SL: {self.current_stop_loss:.2f})", color=Fore.RED, bold=True)
                closed_this_cycle = True
            elif self.current_take_profit > 0 and (
                 (self.trade_type == "Long" and current_price_4h >= self.current_take_profit) or \
                 (self.trade_type == "Short" and current_price_4h <= self.current_take_profit)):
                self.log(f"TAKE PROFIT HIT for {self.trade_type} at {current_price_4h:.2f} (TP: {self.current_take_profit:.2f})", color=Fore.GREEN, bold=True)
                closed_this_cycle = True

            if closed_this_cycle:
                close_order_info = self.api.close_position_market(self.config.SYMBOL, self.trade_type.lower(), self.position_size_btc)
                if close_order_info and close_order_info.get('id'):
                    self.log(f"Position close order ({close_order_info['id']}) placed successfully.", color=Fore.YELLOW)
                else:
                    self.log(f"ERROR: Failed to place position close order! Response: {close_order_info}", "error", Fore.RED, bold=True)
                self._reset_position_state()

        self.log("Strategy Step Completed.", color=Fore.MAGENTA, bold=True)
        self.log("="*50 + "\n", color=Fore.MAGENTA, bold=True)


    def check_entry_conditions(self, current_price: float, current_index: int, candles: List[CandleData]) -> tuple[float, float, List[str]]:
        long_score, short_score = 0.0, 0.0
        reasons = []
        if not candles or len(candles) < max(self.config.BB_PERIOD, self.config.RSI_PERIOD +1, 20):
            return long_score, short_score, reasons
        std_fibs = self.calculate_standard_fib_levels(candles, current_price)
        if std_fibs:
            near_std_sup = any(abs(current_price - s_level) / current_price <= 0.005 for s_level in std_fibs.get("supports",[]))
            near_std_res = any(abs(current_price - r_level) / current_price <= 0.005 for r_level in std_fibs.get("resistances",[]))
            if near_std_sup: long_score += 0.75; reasons.append(f"Near Std Fib S")
            if near_std_res: short_score += 0.75; reasons.append(f"Near Std Fib R")
        trends = self.base_trend_detector.detect_trend_lines(candles)
        fib_circs_data = self.base_fib_circles.generate_circles(trends)
        s_circ_levels, r_circ_levels = self.base_fib_circles.get_support_resistance_levels(fib_circs_data, current_index)

        if s_circ_levels and any(abs(current_price - s_level)/current_price <= 0.01 for s_level in s_circ_levels):
            long_score += 0.5; reasons.append("Near Fib Circle S")
        if r_circ_levels and any(abs(current_price - r_level)/current_price <= 0.01 for r_level in r_circ_levels):
            short_score += 0.5; reasons.append("Near Fib Circle R")

        rsi_val = self.calculate_rsi_value(candles)
        if rsi_val is not None:
            if rsi_val >= self.config.RSI_OVERBOUGHT: short_score += 1.0; reasons.append(f"RSI OB ({rsi_val:.1f})")
            elif rsi_val <= self.config.RSI_OVERSOLD: long_score += 1.0; reasons.append(f"RSI OS ({rsi_val:.1f})")
        bbands = self.base_bollinger.calculate(candles)
        latest_valid_idx = -1
        if bbands and bbands['upper'].size > 0:
            latest_valid_idx = next((i for i in range(len(bbands['upper']) - 1, -1, -1) if not np.isnan(bbands['upper'][i])), -1)

        if latest_valid_idx != -1:
            is_near_bb, band_type = self.base_bollinger.is_near_band(current_price, bbands, latest_valid_idx, threshold=0.01)
            if is_near_bb:
                if band_type == 'lower': long_score += 1.0; reasons.append("Near Lower BB")
                elif band_type == 'upper': short_score += 1.0; reasons.append("Near Upper BB")
        return long_score, short_score, reasons

# --- Main Bot Runner ---
if __name__ == "__main__":
    main_logger = logging.getLogger("MainBot")
    main_logger.info("Bot Starting Up...")

    if not POMEGRANATE_AVAILABLE:
        main_logger.critical("Pomegranate library is NOT INSTALLED. HMM features will be disabled. Install with: pip install pomegranate>=1.0.0")
        # exit(1) # Uncomment if you want to force exit

    config = Config()
    api = CoincatchApiClient(config)
    strategy = LiveStrategy(config, api)

    # HMM model training if not fitted
    if POMEGRANATE_AVAILABLE and not strategy.hmm_detector.is_fitted:
        main_logger.warning("HMM model not found or not fitted. Attempting to train initial model...")
        historical_candles = api.get_candles(
            symbol=config.SYMBOL,
            interval_from_config=config.KLINE_INTERVAL_PRIMARY,
            limit=config.HMM_MIN_TRAIN_CANDLES + 50
        )
        if historical_candles and len(historical_candles) >= config.HMM_MIN_TRAIN_CANDLES:
            main_logger.info(f"Fetched {len(historical_candles)} candles for HMM training.")
            strategy.hmm_detector.train(historical_candles)
            if strategy.hmm_detector.is_fitted:
                main_logger.info("HMM model trained successfully.")
            else:
                main_logger.error("HMM model training FAILED. Bot will run with HMM features disabled.")
        else:
            main_logger.error(f"Not enough historical data ({len(historical_candles) if historical_candles else 0}) to train HMM model (need {config.HMM_MIN_TRAIN_CANDLES}). HMM disabled.")

    # Set initial leverage
    if api.ccxt_client:
        main_logger.info(f"Attempting to set initial leverage to {config.LEVERAGE}x for {config.SYMBOL}")
        api.set_leverage(config.SYMBOL, config.MARGIN_COIN, config.LEVERAGE, "long")
        api.set_leverage(config.SYMBOL, config.MARGIN_COIN, config.LEVERAGE, "short")
    else:
        main_logger.error("CCXT client not initialized in API. Cannot set leverage.")

    main_logger.info("Entering main trading loop...")
    while True:
        try:
            strategy.run_strategy_step()
            main_logger.info(f"Loop finished. Sleeping for {config.LOOP_SLEEP_SECONDS} seconds...")
            time.sleep(config.LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            main_logger.info("KeyboardInterrupt detected. Shutting down...")
            if strategy.position_open:
                main_logger.warning("Position is open. Consider manual review on exchange.")
            break
        except Exception as e:
            main_logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
            main_logger.info(f"Sleeping for {config.LOOP_SLEEP_SECONDS*2} seconds after error...")
            time.sleep(config.LOOP_SLEEP_SECONDS * 2)

