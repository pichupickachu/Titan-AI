import pandas as pd
import titan_math  # Ensure titan_math.pyd (or .so) is in the same directory

class Analyzer:

    # =========================================================
    # 1. ORDER BOOK WALLS (C++)
    # =========================================================
    @staticmethod
    def detect_liquidity_walls(orderbook, current_price):
        if not orderbook: return {'support': [], 'resistance': []}

        try:
            # Pass order book to C++
            cpp_res = titan_math.detect_liquidity_walls(
                orderbook['bids'],
                orderbook['asks'],
                float(current_price)
            )

            # Convert result back to Python dictionaries
            return {
                'support': [{'price': w.price, 'vol': w.vol} for w in cpp_res.support],
                'resistance': [{'price': w.price, 'vol': w.vol} for w in cpp_res.resistance]
            }
        except Exception as e:
            print(f"⚠️ Wall Detect Error: {e}")
            return {'support': [], 'resistance': []}

    # =========================================================
    # 2. MARKET REGIME (C++ EMA + ADX + Z-SCORE)
    # =========================================================
    @staticmethod
    def get_market_instruction(btc_df):
        if btc_df.empty:
            return {
                "regime": "FLAT",
                "min_score": 60,
                "strategy_allowed": ["SAFE_REVERSAL", "SNIPE", "SCALPING", "VOL_BREAKOUT"],
                "ai_confidence": 7
            }

        highs = btc_df['high'].tolist()
        lows = btc_df['low'].tolist()
        closes = btc_df['close'].tolist()

        # Get analysis from C++
        res = titan_math.get_market_instruction(highs, lows, closes)

        # If regime is CRASH - disable all strategies
        if res.regime != "CRASH":
            strategies = ["SAFE_REVERSAL", "SNIPE", "SCALPING", "VOL_BREAKOUT", "BB_BREAKOUT", "CHANNEL_BOUNCE"]
        else:
            strategies = []  # Do not trade during crash

        return {
            "regime": res.regime,
            "strategy_allowed": strategies,
            "min_score": res.min_score if res.min_score < 100 else 60,
            "ai_confidence": 7
        }

    # =========================================================
    # 3. HEATMAP (C++)
    # =========================================================
    @staticmethod
    def estimate_liquidation_heatmap(df, window=50):
        if len(df) < window: return []
        highs = df['high'].tolist()
        lows = df['low'].tolist()

        try:
            cpp_levels = titan_math.get_liquidation_heatmap(highs, lows, window)
            return [{'price': x.price, 'type': x.type, 'leverage': x.leverage} for x in cpp_levels]
        except:
            return []

    @staticmethod
    def get_market_context(symbol, df_large):
        """
        Finds global Smart Money levels via C++
        """
        if df_large.empty: return []
        all_prices = df_large['close'].tolist()
        smart_levels = titan_math.find_smart_money_levels(all_prices, 0.005)
        return [lvl.price for lvl in smart_levels]

    # =========================================================
    # 4. 🔥 MAIN SIGNAL (TELESCOPE EDITION)
    # =========================================================
    @staticmethod
    def calculate_signal(df, obi_ratio, btc_regime, btc_df, current_price=None):
        """
        current_price: Live price from streamer (WebSocket).
        If passed, we create a "Ghost Candle" for C++ to reduce latency.
        """
        empty_debug = {'rsi': 50, 'score': 0, 'strat': 'WAIT', 'candle': 'GRAY'}

        data_len = len(df)
        if data_len < 50:
            return 0, empty_debug, "WAIT_DATA"

        try:
            # 1. Prepare data for C++
            # .tolist() creates a copy, so the original df is safe
            opens = df['open'].tolist()
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            closes = df['close'].tolist()
            volumes = df['volume'].tolist()

            # 2. 🔥 GHOST CANDLE LOGIC
            # Append current price as a new candle so indicators (like RSI) react immediately
            if current_price is not None and current_price > 0:
                last_close = closes[-1]

                opens.append(last_close)         # New Open = Old Close
                closes.append(current_price)     # New Close = Current Price

                # Adapt High/Low to the current moment
                current_high = max(last_close, current_price)
                current_low = min(last_close, current_price)
                highs.append(current_high)
                lows.append(current_low)

                # Use last volume to avoid breaking formulas
                volumes.append(volumes[-1])

            # 3. Calculate BTC change
            btc_change = 0.0
            if not btc_df.empty:
                last_btc = btc_df.iloc[-1]
                btc_change = (last_btc['close'] - last_btc['open']) / last_btc['open']

            # 4. 🔥 Call C++ Core
            res = titan_math.get_trading_signal(
                opens, highs, lows, closes, volumes,
                btc_regime,
                btc_change
            )

            # Normalize ATR to percentage
            last_p = closes[-1]
            atr_pct = (res.atr / last_p) * 100 if last_p > 0 else 0

            debug_info = {
                "reasons": res.reasons,
                "rsi": round(res.rsi, 2),
                "z_rsi": round(res.z_score_rsi, 2),
                "z_vol": round(res.z_score_vol, 2),
                "strat": res.strategy,
                "score": res.score,
                "atr_pct": round(atr_pct, 2),
                "candle": "GREEN" if res.is_green else "RED"
            }

            return res.score, debug_info, res.strategy

        except Exception as e:
            print(f"⚠️ Analyzer Error: {e}")
            return 0, empty_debug, "ERR"

    # =========================================================
    # 5. 🔥 MICROSTRUCTURE (ORDER FLOW)
    # =========================================================
    @staticmethod
    def get_microstructure_signal(orderbook):
        """
        Analyzes order book microstructure (imbalance, spread, wmid) via C++
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return {'imbalance': 0, 'wmid': 0, 'spread': 0}

        try:
            res = titan_math.analyze_order_book_microstructure(
                orderbook['bids'],
                orderbook['asks']
            )

            return {
                'imbalance': res.imbalance,
                'wmid': res.wmid,
                'spread': res.spread
            }
        except Exception as e:
            # print(f"⚠️ Microstructure Error: {e}") # Uncomment for debug
            return {'imbalance': 0, 'wmid': 0, 'spread': 0}

    # =========================================================
    # 6. WHALE CHASING
    # =========================================================
    @staticmethod
    def check_whale_activity(trades):
        """
        Searches for whales in the trade tape.
        """
        if not trades: return 0

        # Format data for C++: [price, amount, side(1/-1)]
        cpp_data = []
        for t in trades:
            side = 1.0 if t['side'] == 'buy' else -1.0
            cpp_data.append([t['price'], t['amount'], side])

        try:
            tape = titan_math.analyze_tape_momentum(cpp_data)
            # If buy pressure > 70% OR > 2 large buys observed
            if tape.pressure > 70 or tape.whale_buys >= 2:
                return tape.whale_buys
        except:
            return 0

        return 0
