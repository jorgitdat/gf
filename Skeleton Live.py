# live_coincatch_bot.py

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
from typing import Optional

# Initialize colorama
init(autoreset=True)

# Update logging configuration at the top of the file
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_bot.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Changed from DEBUG to INFO
logging.getLogger("CoincatchApiClient").setLevel(logging.INFO)

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
        return (f"Candle(T={datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M')}, "
                f"O={self.open}, H={self.high}, L={self.low}, C={self.close}, V={self.volume})")

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


# --- Live Strategy Logic (Ported and Adapted from Test33LIVEStrategy) ---
class LiveStrategy:
    def __init__(self, config, api_client):
        self.config = config
        self.api_client = api_client
        self.logger = logging.getLogger("LiveStrategy")

        self.candles_4h = []
        self.candles_1d = []

        self.trend_detector = TrendLineDetector(
            min_points=config.RSI_PERIOD,
            max_lookback=100,
            min_strength=2
        )
        self.fib_circles = FibonacciCircles()
        self.bollinger = BollingerBands(period=config.BB_PERIOD, std_dev=config.BB_STD_DEV)
        self.rsi_indicator = RSI(period=config.RSI_PERIOD)

        # Position State
        self.position_open = False
        self.entry_price = 0.0
        self.position_size_btc = 0.0
        self.trade_type = None
        self.current_stop_loss = 0.0
        self.current_take_profit = 0.0
        self.allocated_margin_usd = 0.0
        self.api_order_id = None

        # Dynamic SL/TP state
        self.trailing_sl_activated = False
        self.breakeven_sl_activated = False
        self.initial_sl_set_by_logic = 0.0

        self.last_trade_time = 0

    def log(self, message, level="info", color=None, bold=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"{timestamp} - {message}"
        if color:
            log_msg = f"{color}{log_msg}{Style.RESET_ALL}"
        if bold:
            log_msg = f"{Style.BRIGHT}{log_msg}{Style.RESET_ALL}"

        print(log_msg)
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)

    def _fetch_and_prepare_candles(self):
        new_candles_4h = self.api_client.get_candles(
            symbol=self.config.SYMBOL,
            interval_from_config=self.config.KLINE_INTERVAL_PRIMARY,
            limit=self.config.MAX_CANDLES_4H
        )
        if new_candles_4h:
            self.candles_4h = sorted(new_candles_4h, key=lambda c: c.timestamp)[-self.config.MAX_CANDLES_4H:]
        else:
            self.log(f"Failed to fetch 4H candles. Primary data missing.", level="warning", color=Fore.YELLOW)
            self.candles_4h = []
            return False

        new_candles_1d = self.api_client.get_candles(
            symbol=self.config.SYMBOL,
            interval_from_config=self.config.KLINE_INTERVAL_SECONDARY,
            limit=self.config.MAX_CANDLES_1D
        )
        if new_candles_1d:
            self.candles_1d = sorted(new_candles_1d, key=lambda c: c.timestamp)[-self.config.MAX_CANDLES_1D:]
        else:
            self.log(f"Failed to fetch 1D candles. Proceeding without 1D data for this cycle.", level="warning", color=Fore.YELLOW)
            self.candles_1d = []
        return True

    def _update_account_and_position_status(self):
        position_data = self.api_client.get_position(self.config.SYMBOL, self.config.MARGIN_COIN)
        if position_data:
            size = float(position_data.get("total", position_data.get("contracts", 0)))
            entry_px = float(position_data.get("avgEntryPrice", position_data.get("entryPrice", 0)))
            side = position_data.get("holdSide", position_data.get("side", "")).lower()
            if size > 0 and entry_px > 0 and side in ["long", "short"]:
                if not self.position_open:
                    self.log(f"Found existing {side.upper()} position on exchange: {size:.8f} @ {entry_px:.2f}. Syncing state.", color=Fore.YELLOW)
                    self.position_open = True
                    self.entry_price = entry_px
                    self.position_size_btc = size
                    self.trade_type = "Long" if side == "long" else "Short"
                    if self.current_stop_loss == 0 or self.current_take_profit == 0:
                        safety_sl, safety_tp = self.calculate_sl_tp(entry_px, self.trade_type)
                        self.current_stop_loss = safety_sl
                        self.current_take_profit = safety_tp
                        self.initial_sl_set_by_logic = safety_sl
                        self.log(f"Safety SL/TP set for existing position: SL={self.current_stop_loss:.2f}, TP={self.current_take_profit:.2f}", color=Fore.YELLOW)
                else:
                    if abs(self.entry_price - entry_px) > 0.01 or abs(self.position_size_btc - size) > 1e-8 :
                         self.log(f"Updating position details from API: Entry {self.entry_price:.2f}->{entry_px:.2f}, Size {self.position_size_btc:.8f}->{size:.8f}", color=Fore.CYAN)
                         self.entry_price = entry_px
                         self.position_size_btc = size
                self.position_open = True
            else:
                if self.position_open:
                    self.log("Position reported by API has size 0 or is invalid. Resetting state.", color=Fore.YELLOW)
                    self._reset_position_state()
                self.position_open = False
        else:
            if self.position_open:
                self.log("Position no longer found on exchange. Resetting state.", color=Fore.YELLOW)
                self._reset_position_state()
            self.position_open = False

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
        self.last_trade_time = time.time()

    def calculate_rsi_value(self, candles):
        if not candles or len(candles) < self.config.RSI_PERIOD + 1:
            self.log(f"Not enough candles ({len(candles) if candles else 0}) for RSI. Need {self.config.RSI_PERIOD + 1}. Returning neutral 50.", level="debug")
            return 50
        rsi_val = self.rsi_indicator.calculate(candles)
        if rsi_val is None:
            self.log(f"RSI calculation returned None. Returning neutral 50.", level="debug")
            return 50
        return rsi_val

    def calculate_standard_fib_levels(self, candles_for_tf, current_price, lookback_period=50):
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
        if period_high <= period_low or abs(period_high - period_low) / max(period_high, period_low, 1e-9) < 0.001:
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

    def calculate_sl_tp_based_on_backtest(self, entry_price, direction):
        """
        Calculate SL/TP aiming to replicate the backtest's logic with improvements:
        1. SL based on Fibonacci Circle S/R levels.
        2. TP based on Standard Fibonacci levels or Risk:Reward ratio as fallback.
        """
        self.log(f"Calculating SL/TP based on improved backtest logic for {direction} from entry {entry_price:.2f}", color=Fore.BLUE)

        # Ensure we have 4H candles for the TrendDetector
        if not self.candles_4h or len(self.candles_4h) < self.trend_detector.min_points:
            self.log("Not enough 4H candles for Fib Circle SL/TP. Falling back.", color=Fore.YELLOW)
            return self.calculate_sl_tp_percentage_fallback(entry_price, direction)

        # 1. Calculate Stop Loss based on Fibonacci Circle S/R (from backtest logic)
        trend_lines_for_sl = self.trend_detector.detect_trend_lines(self.candles_4h)
        fib_circles_for_sl = self.fib_circles.generate_circles(trend_lines_for_sl)

        idx_within_lookback = min(len(self.candles_4h), self.trend_detector.max_lookback) - 1
        support_levels, resistance_levels = self.fib_circles.get_support_resistance_levels(
            fib_circles_for_sl,
            idx_within_lookback
        )

        stop_loss_price = None
        if direction == 'Long':
            valid_supports = [level for level in support_levels if level < entry_price and level > 0]
            if valid_supports:
                stop_loss_price = max(valid_supports)
                self.log(f"SL (Long) based on Fib Circle Support: {stop_loss_price:.2f}", color=Fore.BLUE)
            else:
                stop_loss_price = entry_price * 0.99
                self.log(f"SL (Long) no valid Fib Circle support. Using fallback: {stop_loss_price:.2f}", color=Fore.YELLOW)
        else:  # Short
            valid_resistances = [level for level in resistance_levels if level > entry_price]
            if valid_resistances:
                stop_loss_price = min(valid_resistances)
                self.log(f"SL (Short) based on Fib Circle Resistance: {stop_loss_price:.2f}", color=Fore.BLUE)
            else:
                stop_loss_price = entry_price * 1.01
                self.log(f"SL (Short) no valid Fib Circle resistance. Using fallback: {stop_loss_price:.2f}", color=Fore.YELLOW)

        # 2. Calculate Take Profit based on Standard Fibonacci levels
        take_profit_price = None
        # Get Standard Fibonacci levels using entry price as current_price
        std_fib_levels = self.calculate_standard_fib_levels(self.candles_4h, entry_price)

        if direction == 'Long':
            # For longs, look for resistance levels above entry price
            if std_fib_levels and std_fib_levels["resistances"]:
                # Get the first (closest) resistance level above entry
                valid_resistances = [r for r in std_fib_levels["resistances"] if r > entry_price]
                if valid_resistances:
                    take_profit_price = min(valid_resistances)
                    self.log(f"TP (Long) based on Standard Fib Resistance: {take_profit_price:.2f}", color=Fore.BLUE)
        else:  # Short
            # For shorts, look for support levels below entry price
            if std_fib_levels and std_fib_levels["supports"]:
                # Get the first (closest) support level below entry
                valid_supports = [s for s in std_fib_levels["supports"] if s < entry_price]
                if valid_supports:
                    take_profit_price = max(valid_supports)
                    self.log(f"TP (Short) based on Standard Fib Support: {take_profit_price:.2f}", color=Fore.BLUE)

        # If no suitable Standard Fib level found for TP, fall back to R:R based TP
        if take_profit_price is None or (direction == "Long" and take_profit_price <= entry_price) or (direction == "Short" and take_profit_price >= entry_price):
            # Use Risk:Reward ratio as fallback
            risk_reward_ratio = 2.0

            if direction == 'Long':
                if entry_price > stop_loss_price:  # Valid SL
                    risk = entry_price - stop_loss_price
                    take_profit_price = entry_price + (risk * risk_reward_ratio)
                    self.log(f"TP (Long) No valid Std Fib resistance. Using R:R fallback: {take_profit_price:.2f}", color=Fore.YELLOW)
                else:  # SL is invalid
                    self.log(f"Invalid SL {stop_loss_price:.2f} for LONG from entry {entry_price:.2f}. Cannot calc R:R TP.", color=Fore.RED)
                    return None, None
            else:  # Short
                if entry_price < stop_loss_price:  # Valid SL
                    risk = stop_loss_price - entry_price
                    take_profit_price = entry_price - (risk * risk_reward_ratio)
                    self.log(f"TP (Short) No valid Std Fib support. Using R:R fallback: {take_profit_price:.2f}", color=Fore.YELLOW)
                else:  # SL is invalid
                    self.log(f"Invalid SL {stop_loss_price:.2f} for SHORT from entry {entry_price:.2f}. Cannot calc R:R TP.", color=Fore.RED)
                    return None, None

        # Log final calculated values
        risk_amount = abs(stop_loss_price - entry_price)
        reward_amount = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        self.log(f"Final SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, R:R = 1:{risk_reward_ratio:.2f}", color=Fore.BLUE)
        return stop_loss_price, take_profit_price

    def _update_dynamic_sl(self, current_price):
        """
        Updates the stop loss dynamically based on price movement.
        Modified to use a more sensitive breakeven trigger.
        """
        if not self.position_open or not self.trade_type or self.entry_price == 0:
            return

        # Get the current position data to check actual UPNL
        current_pos_data = self.api_client.get_position(self.config.SYMBOL, self.config.MARGIN_COIN)
        unrealized_pnl = 0.0
        if current_pos_data and "unrealizedPL" in current_pos_data:
            unrealized_pnl = float(current_pos_data.get("unrealizedPL", 0.0))

        # Calculate profit percentage based on asset price movement
        asset_price_profit_pct = ((current_price - self.entry_price) / self.entry_price) if self.trade_type == "Long" else \
                                 ((self.entry_price - current_price) / self.entry_price)

        # Log both metrics for transparency
        if asset_price_profit_pct > 0.01:  # Only log if there's some profit (avoid log spam)
            self.log(f"Position status: Asset price movement: {asset_price_profit_pct:.2%}, UPNL: {unrealized_pnl:.2f} USDT",
                    level="debug", color=Fore.CYAN)

        # Update breakeven stop loss (using asset price movement %)
        if not self.breakeven_sl_activated and asset_price_profit_pct >= self.config.BREAKEVEN_PROFIT_PCT:
            new_sl = self.entry_price * (1.001 if self.trade_type == "Long" else 0.999)
            if (self.trade_type == "Long" and new_sl > self.current_stop_loss) or \
               (self.trade_type == "Short" and new_sl < self.current_stop_loss):
                self.current_stop_loss = new_sl
                self.breakeven_sl_activated = True
                self.log(f"Breakeven SL activated at {asset_price_profit_pct:.2%} profit. SL moved to {new_sl:.2f}", color=Fore.GREEN, bold=True)

        # Update trailing stop (using asset price movement %)
        if not self.trailing_sl_activated and asset_price_profit_pct >= self.config.TRAILING_STOP_ACTIVATION_PROFIT_PCT:
            self.trailing_sl_activated = True
            self.log(f"Trailing SL activated at {asset_price_profit_pct:.2%} profit", color=Fore.GREEN, bold=True)

        if self.trailing_sl_activated:
            new_sl = current_price * (1 - self.config.TRAILING_STOP_DISTANCE_PCT) if self.trade_type == "Long" else \
                    current_price * (1 + self.config.TRAILING_STOP_DISTANCE_PCT)

            # Ensure trailing SL respects breakeven if activated
            if self.breakeven_sl_activated:
                new_sl = max(new_sl, self.entry_price * 1.001) if self.trade_type == "Long" else \
                        min(new_sl, self.entry_price * 0.999)

            # Only move SL in favorable direction
            if (self.trade_type == "Long" and new_sl > self.current_stop_loss) or \
               (self.trade_type == "Short" and new_sl < self.current_stop_loss):
                prev_sl = self.current_stop_loss
                self.current_stop_loss = new_sl
                self.log(f"Trailing SL updated: {prev_sl:.2f} → {new_sl:.2f} ({self.config.TRAILING_STOP_DISTANCE_PCT:.2%} from price {current_price:.2f})",
                        color=Fore.GREEN)

    def _log_market_analysis(self, timeframe_name, current_price, candles_for_tf, current_idx):
        self.log(f"----- {timeframe_name} Analysis -----", color=Fore.CYAN)
        self.log(f"Current Price: {current_price:.2f}", color=Fore.CYAN)
        rsi_value = self.calculate_rsi_value(candles_for_tf)
        if rsi_value is not None:
            self.log(f"RSI ({self.config.RSI_PERIOD}): {rsi_value:.2f}", color=Fore.CYAN)
        bbands = self.bollinger.calculate(candles_for_tf)
        if bbands and bbands['upper'].size > 0 and not np.all(np.isnan(bbands['upper'])):
             latest_valid_idx = -1
             for i in range(len(bbands['upper']) - 1, -1, -1):
                 if not np.isnan(bbands['upper'][i]):
                     latest_valid_idx = i
                     break
             if latest_valid_idx != -1:
                 self.log(f"Bollinger ({self.config.BB_PERIOD},{self.config.BB_STD_DEV}): L={bbands['lower'][latest_valid_idx]:.2f} M={bbands['middle'][latest_valid_idx]:.2f} U={bbands['upper'][latest_valid_idx]:.2f}", color=Fore.CYAN)
             else: self.log("Bollinger Bands: All NaN values.", color=Fore.YELLOW)
        else: self.log("Bollinger Bands: Not available or empty.", color=Fore.YELLOW)
        try:
            std_fib_result = self.calculate_standard_fib_levels(candles_for_tf, current_price)
            if std_fib_result:
                std_fib_supports, std_fib_resistances = std_fib_result["supports"], std_fib_result["resistances"]
                if std_fib_supports: self.log(f"Standard Fib Supports: {std_fib_supports}", color=Fore.GREEN)
                else: self.log("No Standard Fibonacci Support Levels calculated.", color=Fore.YELLOW)
                if std_fib_resistances: self.log(f"Standard Fib Resistances: {std_fib_resistances}", color=Fore.RED)
                else: self.log("No Standard Fibonacci Resistance Levels calculated.", color=Fore.YELLOW)
            else: self.log("Standard Fibonacci analysis did not return levels.", color=Fore.YELLOW)
        except Exception as e:
            self.log(f"Error during Standard Fibonacci analysis for {timeframe_name}: {e}", level="error", color=Fore.RED)
            self.logger.error(f"Full traceback for Standard Fibonacci error on {timeframe_name}:", exc_info=True)
        try:
            lookback = getattr(self.trend_detector, 'max_lookback', 100)
            if len(candles_for_tf) >= lookback:
                trend_lines = self.trend_detector.detect_trend_lines(candles_for_tf[-lookback:])
                if trend_lines:
                    self.log(f"Detected {len(trend_lines)} trendline(s) on {timeframe_name} (Lookback: {lookback})", color=Fore.BLUE)
                    adjusted_current_idx = lookback - 1
                    fib_circles_data = self.fib_circles.generate_circles(trend_lines, max_circles=3)
                    if fib_circles_data:
                        support_levels, resistance_levels = self.fib_circles.get_support_resistance_levels(fib_circles_data, adjusted_current_idx)
                        if support_levels: self.log(f"Fib Circle Supports: {support_levels[:3]}", color=Fore.GREEN)
                        else: self.log("No Fibonacci Circle Support Levels calculated.", color=Fore.YELLOW)
                        if resistance_levels: self.log(f"Fib Circle Resistances: {resistance_levels[:3]}", color=Fore.RED)
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
        self.log(f"  Raw Scores: Long={long_4h}, Short={short_4h}", color=Fore.CYAN)
        if reasons_4h: self.log("  Contributing Reasons (4H):", color=Fore.CYAN); [self.log(f"    • {r}", color=Fore.CYAN) for r in reasons_4h]
        else: self.log("  No specific 4H signals triggered scoring.", color=Fore.CYAN)
        self.log("--- 1D Timeframe Signal Calculation ---", color=Fore.CYAN)
        self.log(f"  Raw Scores: Long={long_1d}, Short={short_1d}", color=Fore.CYAN)
        if reasons_1d: self.log("  Contributing Reasons (1D):", color=Fore.CYAN); [self.log(f"    • {r}", color=Fore.CYAN) for r in reasons_1d]
        else: self.log("  No specific 1D signals triggered scoring.", color=Fore.CYAN)
        self.log(f"--- Combined Signal (Weights: 4H={0.6}, 1D={0.4}) ---", color=Fore.YELLOW)
        self.log(f"  FINAL Combined Scores: Long={combined_long:.2f}, Short={combined_short:.2f}", color=Fore.YELLOW, bold=True)
        if combined_long >= self.config.MIN_SCORE_ENTRY and combined_long > combined_short: self.log(f"  Potential Action: LONG. Score ({combined_long:.2f}) >= Threshold and > Short ({combined_short:.2f}).", color=Fore.GREEN, bold=True)
        elif combined_short >= self.config.MIN_SCORE_ENTRY and combined_short > combined_long: self.log(f"  Potential Action: SHORT. Score ({combined_short:.2f}) >= Threshold and > Long ({combined_long:.2f}).", color=Fore.RED, bold=True)
        else: self.log(f"  Potential Action: NO TRADE. Long {combined_long:.2f}, Short {combined_short:.2f}, Threshold {self.config.MIN_SCORE_ENTRY}", color=Fore.YELLOW, bold=True)
        self.log("======================================\n", color=Fore.MAGENTA, bold=True)

    def combine_signals(self, long_4h, short_4h, long_1d, short_1d):
        weight_4h = 0.6; weight_1d = 0.4
        return (long_4h * weight_4h) + (long_1d * weight_1d), (short_4h * weight_4h) + (short_1d * weight_1d)

    def calculate_sl_tp(self, entry_price, direction, risk_factor_sl=0.01, reward_factor_tp=0.02):
        """Main SL/TP calculation method, now using backtest-based logic with fallback"""
        sl_price, tp_price = None, None # Initialize
        try:
            # Try the advanced backtest-based calculation first
            sl_price_backtest, tp_price_backtest = self.calculate_sl_tp_based_on_backtest(entry_price, direction)

            # Validate the calculated values from backtest logic
            if sl_price_backtest is not None and tp_price_backtest is not None:
                valid = ((direction == "Long" and sl_price_backtest < entry_price and tp_price_backtest > entry_price) or
                         (direction == "Short" and sl_price_backtest > entry_price and tp_price_backtest < entry_price))
                if valid:
                    sl_price, tp_price = sl_price_backtest, tp_price_backtest
                    self.log(f"Using backtest SL/TP calc: {direction} Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}", color=Fore.CYAN)
                else:
                    self.log(f"Backtest SL/TP invalid (SL={sl_price_backtest}, TP={tp_price_backtest}). Reverting to percentage.", color=Fore.YELLOW)
            else:
                self.log(f"Backtest SL/TP method returned None. Reverting to percentage.", color=Fore.YELLOW)

        except Exception as e:
            self.log(f"Error in backtest-based SL/TP: {e}. Using fallback.", level="error", color=Fore.RED, exc_info=True)

        # Fall back to percentage-based calculation if advanced method failed or returned invalid values
        if sl_price is None or tp_price is None:
            sl_price, tp_price = self.calculate_sl_tp_percentage_fallback(entry_price, direction, risk_factor_sl, reward_factor_tp)
            self.log(f"Using fallback SL/TP calc: {direction} Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}", color=Fore.CYAN)

        # Apply precision at the very end
        market_details = self.api_client.get_market_details(self.config.SYMBOL)
        price_precision = None
        if market_details and market_details.get('precision') and market_details['precision'].get('price'):
            precision_val = market_details['precision']['price']
            if isinstance(precision_val, float) and precision_val < 1:
                price_precision = abs(int(math.log10(precision_val)))
            elif isinstance(precision_val, int):
                price_precision = precision_val

        if price_precision is not None:
            factor = 10 ** price_precision
            if sl_price is not None:
                sl_price = (math.floor(sl_price * factor) / factor) if direction == "Long" else (math.ceil(sl_price * factor) / factor)
            if tp_price is not None:
                tp_price = (math.ceil(tp_price * factor) / factor) if direction == "Long" else (math.floor(tp_price * factor) / factor)

        self.log(f"Final Calculated SL: {sl_price}, TP: {tp_price} for entry {entry_price} ({direction})", color=Fore.CYAN)
        return sl_price, tp_price

    def calculate_position_size_and_margin(self, current_price, available_balance_usd):
        if available_balance_usd <= 0: return None, None
        margin_for_trade_usd = min(available_balance_usd * self.config.MARGIN_PERCENTAGE, self.config.MAX_LOSS_PER_TRADE_USD)
        if margin_for_trade_usd <= 0.01: return None, None
        if current_price == 0: return None, None
        position_size_btc = (margin_for_trade_usd * self.config.LEVERAGE) / current_price
        market_details = self.api_client.get_market_details(self.config.SYMBOL)
        min_qty_from_exchange = float(self.config.MIN_QUANTITY)
        amount_precision = None
        if market_details:
            if market_details.get('limits') and market_details['limits'].get('amount'):
                min_qty_val = market_details['limits']['amount'].get('min')
                if min_qty_val is not None: min_qty_from_exchange = float(min_qty_val)
            if market_details.get('precision') and market_details['precision'].get('amount'):
                precision_val = market_details['precision']['amount']
                if isinstance(precision_val, float) and precision_val < 1: amount_precision = abs(int(math.log10(precision_val)))
                elif isinstance(precision_val, int): amount_precision = precision_val
        if position_size_btc < min_qty_from_exchange:
            position_size_btc = min_qty_from_exchange
            if self.config.LEVERAGE == 0: return None, None
            required_margin_for_min_size = (position_size_btc * current_price) / self.config.LEVERAGE
            if required_margin_for_min_size > available_balance_usd: return None, None
            if required_margin_for_min_size > self.config.MAX_LOSS_PER_TRADE_USD: return None, None
            margin_for_trade_usd = required_margin_for_min_size
        if amount_precision is not None:
            factor = 10 ** amount_precision
            position_size_btc = math.floor(position_size_btc * factor) / factor
        if position_size_btc < min_qty_from_exchange: return None, None
        return position_size_btc, margin_for_trade_usd

    def calculate_sl_tp_percentage_fallback(self, entry_price, direction, risk_factor_sl=0.01, reward_factor_tp=0.02):
        if direction == "Long": sl_price = entry_price * (1 - risk_factor_sl); tp_price = entry_price * (1 + reward_factor_tp)
        else: sl_price = entry_price * (1 + risk_factor_sl); tp_price = entry_price * (1 - reward_factor_tp)
        # Precision adjustment
        market_details = self.api_client.get_market_details(self.config.SYMBOL)
        price_precision = None
        if market_details and market_details.get('precision') and market_details['precision'].get('price'):
            precision_val = market_details['precision']['price']
            if isinstance(precision_val, float) and precision_val < 1: price_precision = abs(int(math.log10(precision_val)))
            elif isinstance(precision_val, int): price_precision = precision_val
        if price_precision is not None:
            factor = 10 ** price_precision
            if direction == "Long": sl_price = math.floor(sl_price * factor)/factor; tp_price = math.ceil(tp_price*factor)/factor
            else: sl_price = math.ceil(sl_price * factor)/factor; tp_price = math.floor(tp_price*factor)/factor
        return sl_price, tp_price

    # --- MAIN STRATEGY EXECUTION METHOD ---
    def run_strategy_step(self):
        self.log("="*50, color=Fore.MAGENTA, bold=True)
        self.log("Running Strategy Step", color=Fore.MAGENTA, bold=True)

        if not self._fetch_and_prepare_candles():
            self.log("Halting strategy step due to candle fetch failure.", level="error", color=Fore.RED)
            return

        if not self.candles_4h:
            self.log("No 4H candles available after fetch. Halting step.", level="error", color=Fore.RED)
            return

        current_price_4h = self.candles_4h[-1].close
        current_idx_4h = len(self.candles_4h) - 1

        self.log(f"Current 4H Price: {current_price_4h:.2f} USDT", color=Fore.CYAN)
        current_price_1d = self.candles_1d[-1].close if self.candles_1d else current_price_4h
        current_idx_1d = len(self.candles_1d) - 1 if self.candles_1d else -1

        self._update_account_and_position_status()

        balance_info = self.api_client.get_account_balance(self.config.SYMBOL, self.config.MARGIN_COIN, self.config.PRODUCT_TYPE)
        available_balance_usd = 0
        if balance_info and balance_info.get("code") == "00000" and balance_info.get("data"):
            acc_data = balance_info["data"]
            if acc_data:
                available_balance_usd = float(acc_data.get("available", 0))
                equity_bal = float(acc_data.get("equity", 0))
                unrealized_pnl = float(acc_data.get("unrealizedPL", acc_data.get("unrealizedPnl", 0.0)))
                self.log(f"ACCOUNT BALANCE: Equity={equity_bal:.2f} USDT, Available={available_balance_usd:.2f} USDT, UPNL={unrealized_pnl:.2f} USDT", color=Fore.YELLOW, bold=True)
            else:
                self.log("Could not extract detailed balance info.", color=Fore.YELLOW)
                return
        else:
            self.log(f"Failed to get account balance. Response: {balance_info}", color=Fore.RED)
            return

        if self.position_open:
            current_pos_data = self.api_client.get_position(self.config.SYMBOL, self.config.MARGIN_COIN)
            mark_price_str = current_pos_data.get("marketPrice", "N/A") if current_pos_data else "N/A"
            liq_price_str = current_pos_data.get("liquidationPrice", "N/A") if current_pos_data else "N/A"
            pos_upnl_str = current_pos_data.get("unrealizedPL", "N/A") if current_pos_data else "N/A"
            self.log(f"OPEN POSITION: {self.trade_type} | Size: {self.position_size_btc:.6f} BTC | Entry: {self.entry_price:.2f} | "
                     f"Margin Alloc: {self.allocated_margin_usd:.2f} USDT | Leverage: {self.config.LEVERAGE}x | "
                     f"MarkPx: {mark_price_str} | LiqPx: {liq_price_str} | UPNL: {pos_upnl_str} USDT | "
                     f"Bot SL: {self.current_stop_loss:.2f} | Bot TP: {self.current_take_profit:.2f}",
                     color=(Fore.GREEN if self.trade_type == "Long" else Fore.RED), bold=True)
        else:
            self.log("No open position.", color=Fore.YELLOW)

        self.log("-" * 30, color=Fore.YELLOW)

        # Analyze market conditions on both timeframes
        self._log_market_analysis("4H", current_price_4h, self.candles_4h, current_idx_4h)
        if self.candles_1d and current_idx_1d != -1:
            self._log_market_analysis("1D", current_price_1d, self.candles_1d, current_idx_1d)

        # Position management logic - check for new entries if no position open
        if not self.position_open and (time.time() - self.last_trade_time) >= self.config.COOLDOWN_PERIOD_SECONDS:
            self.log("Checking entry conditions...", color=Fore.BLUE)

            # Get signals for both timeframes
            current_price_1d_for_signal = self.candles_1d[-1].close if self.candles_1d and current_idx_1d != -1 else current_price_4h
            idx_1d_for_signal = current_idx_1d if self.candles_1d and current_idx_1d != -1 else -1
            candles_for_1d_signal = self.candles_1d if self.candles_1d and current_idx_1d != -1 else []

            long_4h, short_4h, reasons_4h = self.check_entry_conditions(current_price_4h, current_idx_4h, self.candles_4h)
            long_1d, short_1d, reasons_1d = (0, 0, [])
            if candles_for_1d_signal:
                long_1d, short_1d, reasons_1d = self.check_entry_conditions(current_price_1d_for_signal, idx_1d_for_signal, candles_for_1d_signal)

            # Combine signals from different timeframes
            combined_long, combined_short = self.combine_signals(long_4h, short_4h, long_1d, short_1d)
            self._log_signal_scores(long_4h, short_4h, reasons_4h, long_1d, short_1d, reasons_1d, combined_long, combined_short)

            # Determine trade direction based on signal strength and threshold
            trade_direction = None
            if combined_long >= self.config.MIN_SCORE_ENTRY and combined_long > combined_short:
                trade_direction = "Long"
            elif combined_short >= self.config.MIN_SCORE_ENTRY and combined_short > combined_long:
                trade_direction = "Short"

            # Place order if we have a valid trade direction
            if trade_direction:
                self.log(f"TRADE SIGNAL: {trade_direction.upper()} | Score: {combined_long if trade_direction == 'Long' else combined_short:.2f}",
                         color=(Fore.GREEN if trade_direction=="Long" else Fore.RED), bold=True)

                # Calculate position size based on available balance
                self.log(f"DEBUG: Calling calculate_position_size_and_margin with current_price_4h={current_price_4h}, available_balance_usd={available_balance_usd}", level="info")
                result_tuple = self.calculate_position_size_and_margin(current_price_4h, available_balance_usd)
                self.log(f"DEBUG: calculate_position_size_and_margin returned: {result_tuple}", level="info")

                if result_tuple is not None and result_tuple != (None, None):
                    pos_size_btc, margin_usd = result_tuple
                    if pos_size_btc > 0 and margin_usd > 0:
                        # Calculate SL/TP levels
                        sl_price, tp_price = self.calculate_sl_tp(current_price_4h, trade_direction)

                        # Validate SL/TP levels
                        if (trade_direction == "Long" and (sl_price >= current_price_4h or tp_price <= current_price_4h)) or \
                           (trade_direction == "Short" and (sl_price <= current_price_4h or tp_price >= current_price_4h)):
                            self.log(f"Invalid SL/TP: Entry={current_price_4h:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f} for {trade_direction}. Skipping trade.", color=Fore.RED)
                        else:
                            # Place order
                            ccxt_side = "buy" if trade_direction == "Long" else "sell"
                            order_info = self.api_client.place_market_order(self.config.SYMBOL, ccxt_side, pos_size_btc, sl_price, tp_price)

                            if order_info and order_info.get('id'):
                                self.log(f"{trade_direction.upper()} ORDER PLACED: ID {order_info['id']}",
                                         color=(Fore.GREEN if trade_direction=="Long" else Fore.RED), bold=True)

                                # Update position state
                                self.position_open = True
                                filled_price_str = order_info.get('average', order_info.get('price'))
                                self.entry_price = float(filled_price_str) if filled_price_str is not None else current_price_4h
                                filled_amount_str = order_info.get('filled', order_info.get('amount'))
                                self.position_size_btc = float(filled_amount_str) if filled_amount_str is not None else pos_size_btc
                                self.trade_type = trade_direction
                                self.current_stop_loss = sl_price
                                self.current_take_profit = tp_price
                                self.allocated_margin_usd = margin_usd
                                self.api_order_id = order_info['id']
                                self.last_trade_time = time.time()
                                self.trailing_sl_activated = False
                                self.breakeven_sl_activated = False
                                self.initial_sl_set_by_logic = sl_price

                                self.log(f"NEW POSITION: {self.trade_type} {self.position_size_btc:.8f} @ {self.entry_price:.2f}, "
                                        f"SL={self.current_stop_loss:.2f}, TP={self.current_take_profit:.2f}, Margin={self.allocated_margin_usd:.2f}",
                                        color=Fore.CYAN)
                            else:
                                self.log(f"FAILED TO PLACE {trade_direction.upper()} ORDER. Response: {order_info}", color=Fore.RED, bold=True)
                    else:
                        self.log(f"Invalid size/margin from calc: Size={pos_size_btc}, Margin={margin_usd}. Skipping.", color=Fore.RED)
                else:
                    self.log("calculate_position_size_and_margin returned None. Skipping trade.", color=Fore.YELLOW)

        # Position management if we have an open position
        elif self.position_open:
            # Update dynamic stop-loss
            self._update_dynamic_sl(current_price_4h)

            closed_this_cycle = False

            # Check if stop-loss has been hit
            if (self.trade_type == "Long" and current_price_4h <= self.current_stop_loss) or \
               (self.trade_type == "Short" and current_price_4h >= self.current_stop_loss):
                self.log(f"STOP LOSS HIT for {self.trade_type} at {current_price_4h:.2f} (SL: {self.current_stop_loss:.2f})",
                         color=Fore.RED, bold=True)
                closed_this_cycle = True

            # Check if take-profit has been hit
            elif (self.trade_type == "Long" and current_price_4h >= self.current_take_profit) or \
                 (self.trade_type == "Short" and current_price_4h <= self.current_take_profit):
                self.log(f"TAKE PROFIT HIT for {self.trade_type} at {current_price_4h:.2f} (TP: {self.current_take_profit:.2f})",
                         color=Fore.GREEN, bold=True)
                closed_this_cycle = True

            # Close position if either SL or TP was hit
            if closed_this_cycle:
                order_info = self.api_client.close_position_market(self.config.SYMBOL, self.trade_type.lower(), self.position_size_btc)
                if order_info:
                    self.log("Position close order placed.", color=Fore.YELLOW)
                else:
                    self.log("ERROR: Failed to place position close order!", color=Fore.RED, bold=True)

                # Reset position state
                self._reset_position_state()

        # In cooldown period after a trade
        elif (time.time() - self.last_trade_time) < self.config.COOLDOWN_PERIOD_SECONDS:
            self.log(f"In cooldown. Time remaining: {self.config.COOLDOWN_PERIOD_SECONDS - (time.time() - self.last_trade_time):.0f}s", color=Fore.YELLOW)

        self.log("Strategy Step Completed.", color=Fore.MAGENTA, bold=True)
        self.log("="*50, color=Fore.MAGENTA, bold=True)

    def check_entry_conditions(self, current_price, current_index, candles):
        """
        Analyzes market conditions to determine entry signals for both long and short positions.
        Returns scores for long and short signals along with reasons for the signals.
        """
        long_score, short_score, reasons = 0, 0, []  # Basic initialization

        if not candles or len(candles) < 20:
            return long_score, short_score, reasons

        # Standard Fibs
        std_fib_levels = self.calculate_standard_fib_levels(candles, current_price)
        if std_fib_levels:
            if std_fib_levels["supports"]:
                for level in std_fib_levels["supports"]:
                    if abs(current_price - level) / current_price <= 0.005:
                        long_score += 0.75
                        reasons.append(f"Near Std Fib S: {level:.2f}")
                        break
            if std_fib_levels["resistances"]:
                for level in std_fib_levels["resistances"]:
                    if abs(current_price - level) / current_price <= 0.005:
                        short_score += 0.75
                        reasons.append(f"Near Std Fib R: {level:.2f}")
                        break

        # Fib Circles
        trend_lines = self.trend_detector.detect_trend_lines(candles)
        fib_circles_data = self.fib_circles.generate_circles(trend_lines)
        support_levels, resistance_levels = self.fib_circles.get_support_resistance_levels(fib_circles_data, current_index)

        for level in support_levels:
            if abs(current_price - level) / current_price <= 0.01:
                long_score += 0.5
                reasons.append(f"Near Fib Circle S: {level:.2f}")
                break

        for level in resistance_levels:
            if abs(current_price - level) / current_price <= 0.01:
                short_score += 0.5
                reasons.append(f"Near Fib Circle R: {level:.2f}")
                break

        # RSI
        rsi_value = self.calculate_rsi_value(candles)
        if rsi_value is not None:
            if rsi_value >= self.config.RSI_OVERBOUGHT:
                short_score += 1
                reasons.append(f"RSI Overbought: {rsi_value:.2f}")
            elif rsi_value <= self.config.RSI_OVERSOLD:
                long_score += 1
                reasons.append(f"RSI Oversold: {rsi_value:.2f}")

        # Bollinger Bands
        bbands = self.bollinger.calculate(candles)
        if bbands and bbands['upper'].size > 0 and not np.all(np.isnan(bbands['upper'])):
            latest_valid_idx = -1
            for i in range(len(bbands['upper']) - 1, -1, -1):
                if not np.isnan(bbands['upper'][i]):
                    latest_valid_idx = i
                    break

            if latest_valid_idx != -1:
                is_near, band_type = self.bollinger.is_near_band(current_price, bbands, latest_valid_idx, threshold=0.01)
                if is_near:
                    if band_type == 'lower':
                        long_score += 1
                        reasons.append("Near Lower BB")
                    elif band_type == 'upper':
                        short_score += 1
                        reasons.append("Near Upper BB")

        return long_score, short_score, reasons

# --- Main Bot Runner ---
if __name__ == "__main__":
    log_file = 'live_trading_bot.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger("MainBot")
    main_logger.info("Bot Starting Up...")

    config = Config()
    api_client = CoincatchApiClient(config)
    strategy = LiveStrategy(config, api_client)

    main_logger.info(f"Attempting to set initial leverage to {config.LEVERAGE}x for {config.SYMBOL}")
    api_client.set_leverage(config.SYMBOL, config.MARGIN_COIN, config.LEVERAGE, "long")
    api_client.set_leverage(config.SYMBOL, config.MARGIN_COIN, config.LEVERAGE, "short")

    main_logger.info("Entering main trading loop...")
    while True:
        try:
            strategy.run_strategy_step()
            main_logger.info(f"Loop finished. Sleeping for {config.LOOP_SLEEP_SECONDS} seconds...")
            time.sleep(config.LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            main_logger.info("KeyboardInterrupt detected. Shutting down...")
            if strategy.position_open:
                main_logger.warning("Position is open. Consider manual review on exchange or implement auto-close on exit.")
            break
        except Exception as e:
            main_logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
            main_logger.info(f"Sleeping for {config.LOOP_SLEEP_SECONDS*2} seconds after error...")
            time.sleep(config.LOOP_SLEEP_SECONDS * 2)

