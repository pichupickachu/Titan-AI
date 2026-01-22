import asyncio
import aiosqlite
import ccxt.async_support as ccxt
import ssl
import socket
import logging
from datetime import datetime, timedelta

# Telegram Libraries
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.client.session.aiohttp import AiohttpSession  # <--- Important

# NETWORK LIBRARIES
from aiohttp import TCPConnector, ClientSession
from aiohttp.resolver import AsyncResolver

from config import CFG
from data_provider import MarketDataProvider
from ai_analyst import AI_Analyst
from analyzer import Analyzer
from streamer import MexcStreamer


# ==============================================================================
# 🛡️ DNS & SSL BYPASS SESSION
# ==============================================================================
class DNSFriendlySession(AiohttpSession):
    """
    The correct way to implement DNS and SSL fixes in aiogram 3.x.
    We pass nothing to __init__, but configure everything inside create_session.
    """

    async def create_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            # 1. DNS Setup (Cloudflare)
            resolver = AsyncResolver(nameservers=["1.1.1.1", "8.8.8.8"])

            # 2. SSL Setup (Ignore provider certificate replacement)
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            # 3. Create connector
            connector = TCPConnector(
                resolver=resolver,
                family=socket.AF_INET,
                ssl=ssl_ctx
            )

            # 4. Create the session itself
            self._session = ClientSession(connector=connector, trust_env=True)

        return self._session


# ==============================================================================
# ⚙️ BOT ENGINE
# ==============================================================================
class BotEngine:
    def __init__(self):
        print("\n" + "=" * 50)
        print("⚙️ LOADING TITAN AI CORE (STABLE EDITION)...")
        print("=" * 50)

        # 0. Safety stubs (so shutdown doesn't fail)
        self.session = None
        self.ex = None

        # 1. Initialize DNS-friendly session for Telegram
        self.tg_session = DNSFriendlySession()

        self.bot = Bot(
            token=CFG.TG_TOKEN,
            session=self.tg_session,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()

        # 2. Configure DNS and SSL for Exchange (MEXC)
        resolver = AsyncResolver(nameservers=["1.1.1.1", "8.8.8.8"])
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        ex_connector = TCPConnector(resolver=resolver, family=socket.AF_INET, ssl=ssl_ctx)

        # 🔥 Here is the CORRECT NAME - self.session
        self.session = ClientSession(connector=ex_connector, trust_env=True)

        # 3. Configure CCXT parameters
        mexc_params = {
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'}
        }

        if CFG.REAL_TRADING:
            mexc_params.update({'apiKey': CFG.API_KEY, 'secret': CFG.MEXC_SECRET})
            mexc_params['options']['adjustForTimeDifference'] = True

        # Create exchange object
        self.ex = ccxt.mexc(mexc_params)

        # 🔥 Direct session injection inside the library
        self.ex.session = self.session

        # 4. Connect your modules (now they see the ready session)
        self.data = MarketDataProvider(self.ex)
        self.ai = AI_Analyst()
        self.streamer = MexcStreamer(CFG.SYMBOLS)

        self.db = "titan_ai_new.db"
        self.running = True
        self.lock = asyncio.Lock()
        self.last_list_update = datetime.now() - timedelta(minutes=20)
        self.cooldowns = {}
        self.data = MarketDataProvider(self.ex)
        self.ai = AI_Analyst()
        self.streamer = MexcStreamer(CFG.SYMBOLS)
        print("✅ Core initialized successfully.")

    # --------------------------------------------------------------------------
    # 🗄️ DATABASE SYSTEM
    # --------------------------------------------------------------------------
    async def init_db(self):
        """
        Full database initialization.
        Configures the journal, creates tables, and performs migrations.
        """
        print("🗄️ Database Diagnostics...")
        try:
            async with aiosqlite.connect(self.db) as db:
                # Enable Write-Ahead Logging (WAL)
                # Allows reading and writing simultaneously without locking.
                # Speed increases by 10x.
                await db.execute("PRAGMA journal_mode=WAL;")
                await db.execute("PRAGMA synchronous=NORMAL;")

                # --- TABLE 1: POSITIONS (Active Trades) ---
                # Stores info about what we hold right now.
                await db.execute("""
                                 CREATE TABLE IF NOT EXISTS positions
                                 (
                                     symbol
                                     TEXT
                                     PRIMARY
                                     KEY,
                                     amount
                                     REAL,
                                     avg_price
                                     REAL,
                                     max_price
                                     REAL,
                                     strat
                                     TEXT,
                                     adds
                                     INTEGER,
                                     entry_time
                                     TEXT,
                                     sl_price
                                     REAL,
                                     tp_price
                                     REAL
                                 )
                                 """)

                # --- TABLE 2: HISTORY (Archive) ---
                # Stores history of all closed trades for statistics.
                await db.execute("""
                                 CREATE TABLE IF NOT EXISTS history
                                 (
                                     id
                                     INTEGER
                                     PRIMARY
                                     KEY
                                     AUTOINCREMENT,
                                     symbol
                                     TEXT,
                                     pnl
                                     REAL,
                                     reason
                                     TEXT,
                                     date
                                     TEXT
                                 )
                                 """)

                # --- MIGRATIONS (Update old DB structure) ---
                # Add Stop Loss column if missing
                try:
                    await db.execute("ALTER TABLE positions ADD COLUMN sl_price REAL")
                except:
                    pass

                # Add Take Profit column if missing
                try:
                    await db.execute("ALTER TABLE positions ADD COLUMN tp_price REAL")
                except:
                    pass

                # Add Max Price column if missing
                try:
                    await db.execute("ALTER TABLE positions ADD COLUMN max_price REAL")
                except:
                    pass

                # Create INDEX for instant history search
                await db.execute("CREATE INDEX IF NOT EXISTS idx_history_symbol ON history (symbol);")

                await db.commit()
                print("✅ Database is healthy (WAL Mode active).")

        except Exception as e:
            print(f"❌ CRITICAL DATABASE ERROR: {e}")
            raise e  # If DB fails, stop the bot

    # --------------------------------------------------------------------------
    # 📨 NOTIFICATION SYSTEM
    # --------------------------------------------------------------------------
    async def notify(self, text):
        """
        Sends formatted message to Telegram.
        Automatically adds buttons to MEXC charts.
        """
        try:
            # Add timestamp
            if "📅" not in text:
                now = datetime.now().strftime("%H:%M:%S")
                text = f"📅 <b>{now}</b>\n{text}"

            # Parse text for tickers (e.g. BTC/USDT)
            # If found, create a link
            words = text.split()
            for w in words:
                if "/USDT" in w:
                    # Clean ticker from formatting symbols
                    clean_sym = w.replace("/USDT", "").replace("*", "").replace("<b>", "").replace("</b>", "")
                    link = f'<a href="https://www.mexc.com/exchange/{clean_sym}_USDT">📊 Open Chart ({clean_sym})</a>'
                    text += f"\n\n{link}"
                    break

            # Send via API
            await self.bot.send_message(
                CFG.TG_ADMIN_ID,
                text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
        except Exception as e:
            print(f"⚠️ Notification Error: {e}")

    # --------------------------------------------------------------------------
    # 🌍 MARKET SCANNER: FILTERING & DISCOVERY
        # --------------------------------------------------------------------------
        # 🌍 MARKET SCANNER: FILTERING & DISCOVERY (Updated with Blacklist)
        # --------------------------------------------------------------------------
    async def update_dynamic_list(self):
        print("🌍 SCANNING MARKET (TOP-200 + BLACKLIST)...")
        try:
            # 1. Fetch all tickers from the exchange
            tickers = await self.ex.fetch_tickers()
            candidates = []

            for symbol, data in tickers.items():
                # Filter for USDT pairs only
                if "/USDT" not in symbol:
                    continue

                # Extract base asset (e.g., "BTC" from "BTC/USDT")
                base_asset = symbol.split('/')[0]

                # --- FILTERING PIPELINE ---

                # 1. Validate against the BLACKLIST defined in config.py
                if base_asset in CFG.BLACKLIST:
                    continue

                # 2. Exclude leveraged "junk" tokens (e.g., 3L, 3S, LONG, SHORT)
                if any(x in symbol for x in ["3L", "3S", "LONG", "SHORT"]):
                    continue

                # 3. Filter by Quote Volume (USDT)
                vol = data.get('quoteVolume', 0)
                if vol and vol > 1000000:  # Only assets with > $1,000,000 daily volume
                    candidates.append({'symbol': symbol, 'vol': vol})

            # Sort by volume: most liquid (trending) coins at the top
            candidates.sort(key=lambda x: x['vol'], reverse=True)

            # Update the global symbol list (selecting top 200 candidates)
            CFG.SYMBOLS = [c['symbol'] for c in candidates[:200]]

            print(f"✅ Discovery: {len(CFG.SYMBOLS)} dynamic candidates updated (Blacklist applied).")
            return True
        except Exception as e:
            print(f"⚠️ Scanner Exception: {e}")
            return False
    # --------------------------------------------------------------------------
    # 💰 SMART SIZING: CAPITAL MANAGEMENT
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # 💰 SMART SIZING v2 (With ATR)
    # --------------------------------------------------------------------------
    async def get_smart_position_size(self, confidence=None, atr_pct=None):
        if not CFG.USE_SMART_SIZE:
            return CFG.BASE_ORDER_SIZE

        base = CFG.BASE_ORDER_SIZE
        history_mult = 1.0
        ai_mult = 1.0
        vol_mult = 1.0  # <--- New multiplier

        try:
            # 1. Balance
            if CFG.REAL_TRADING:
                balance = await self.ex.fetch_balance()
                free_usdt = float(balance.get('USDT', {}).get('free', 0.0))
            else:
                free_usdt = 200.0

            if free_usdt < 1.0: return 0.0

            # 2. History (Losers risk less)
            async with aiosqlite.connect(self.db) as db:
                rows = await (await db.execute("SELECT pnl FROM history ORDER BY id DESC LIMIT 5")).fetchall()
            if rows and rows[0][0] < 0:
                history_mult = 0.8

            # 3. AI Confidence
            if confidence:
                if int(confidence) >= 9:
                    ai_mult = 1.25
                elif int(confidence) < 7:
                    ai_mult = 0.8

            # 4. 🔥 VOLATILITY FILTER (ATR)
            # If coin moves 3% per candle — it's dangerous, cut size
            if atr_pct and atr_pct > 3.0:
                vol_mult = 0.7
                # If coin is flat (<1%) — take a bit more
            elif atr_pct and atr_pct < 1.0:
                vol_mult = 1.1

            final_size = round(base * history_mult * ai_mult * vol_mult, 2)

            # Limits
            if final_size > free_usdt * 0.95: final_size = free_usdt * 0.95
            if final_size < 5.0: final_size = 5.0

            return final_size
        except Exception as e:
            print(f"⚠️ Sizing Error: {e}")
            return base

    # --------------------------------------------------------------------------
    # 🛡️ REPUTATION SYSTEM: COIN KARMA CHECK
    # --------------------------------------------------------------------------
    async def check_coin_reputation(self, symbol):
        """
        Checks coin history. If it caused many losses — block it.
        """
        try:
            async with self.lock:
                async with aiosqlite.connect(self.db) as db:
                    row = await (await db.execute("SELECT SUM(pnl) FROM history WHERE symbol=?", (symbol,))).fetchone()

            if row and row[0]:
                total_loss = row[0]
                # If total loss for coin is > 15$, blacklist it
                if total_loss < -15.0:
                    # print(f"🚫 {symbol} blocked (Loss: {total_loss}$)")
                    return False

            return True
        except:
            return True

    # --------------------------------------------------------------------------
    # 🛒 EXECUTION: BUY (SMART ENTRY)
    # --------------------------------------------------------------------------
    async def execute_buy(self, symbol, current_price, score, debug_data):
        """
        Executes buy signal with smart order routing.
        """
        strategy = debug_data.get('strat', 'UNKNOWN')
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        async with self.lock:
            async with aiosqlite.connect(self.db) as db:
                # Check: do we already have a position?
                row = await (await db.execute("SELECT amount, adds, avg_price FROM positions WHERE symbol=?",
                                              (symbol,))).fetchone()

                if row:
                    # --- DCA LOGIC ---
                    cur_c, adds, old_avg = row
                    if adds >= CFG.MAX_ADDS:
                        return  # Limit reached

                    usd_amt = CFG.BASE_ORDER_SIZE
                    new_adds = adds + 1
                    action = "DCA"
                else:
                    # --- FIRST ENTRY LOGIC ---
                    conf = debug_data.get('confidence', 6)
                    # Extract ATR from data sent by Analyzer
                    atr = debug_data.get('atr_pct', 0)
                    usd_amt = await self.get_smart_position_size(conf, atr)

                    if usd_amt <= 1.0:
                        return

                    new_adds = 0
                    cur_c = 0
                    action = "BUY"
                    old_avg = 0

                # Calculate coin amount
                # Place limit slightly above market (+0.5%) to act as Market but with protection
                limit_price = current_price * 1.005
                raw_coins = usd_amt / limit_price

                # Round to exchange precision
                try:
                    amount = float(self.ex.amount_to_precision(symbol, raw_coins))
                except:
                    amount = round(raw_coins, 4)

                if amount <= 0: return

                real_price = current_price  # Temporary assumption, update after execution
                order_type_str = "SIMULATION"

                # --- SEND ORDER TO EXCHANGE (SMART EXECUTION) ---
                if CFG.REAL_TRADING:
                    try:
                        # ATTEMPT 1: Maker (Post-Only) - Save fees
                        # Place price slightly BELOW ask to enter book
                        maker_price = current_price * 0.9995

                        try:
                            await self.ex.create_order(
                                symbol, 'limit', 'buy', amount, maker_price,
                                params={'timeInForce': 'PO'}  # Post-Only flag
                            )
                            print(f"✅ REAL BUY (MAKER): {symbol} | {amount} @ {maker_price}")
                            real_price = maker_price
                            order_type_str = "MAKER (Limit)"

                        except Exception as e_maker:
                            # If failed (price moved or exchange rejected PO) -> Hit Market
                            # print(f"⚠️ Maker Fail ({e_maker}), switching to Taker...")

                            # ATTEMPT 2: Taker (Market) - Guaranteed entry
                            # Use quoteOrderQty (buy amount in USDT), it's safer
                            await self.ex.create_market_buy_order(symbol, amount, params={'quoteOrderQty': usd_amt})

                            print(f"🚀 REAL BUY (TAKER): {symbol} | {usd_amt}$")
                            real_price = current_price * 1.001  # Approx price with slippage
                            order_type_str = "TAKER (Market)"

                    except Exception as e:
                        print(f"❌ CRITICAL BUY ERROR {symbol}: {e}")
                        return
                else:
                    print(f"🔵 SIM BUY: {symbol} | {usd_amt:.2f}$ | Score: {score}")

                # Calculate Safety Levels (SL/TP)
                sl = real_price * CFG.DEFAULT_SL
                tp = real_price * getattr(CFG, 'DEFAULT_TP', 1.05)

                # Write to DB
                if row:
                    # If averaging - recalculate average price
                    total_cost = (cur_c * old_avg) + (amount * real_price)
                    total_c = cur_c + amount
                    new_avg = total_cost / total_c
                    new_max = max(real_price, new_avg)
                    await db.execute(
                        "UPDATE positions SET amount=?, avg_price=?, adds=?, strat=?, max_price=? WHERE symbol=?",
                        (total_c, new_avg, new_adds, strategy, new_max, symbol))
                else:
                    # If new - insert
                    await db.execute(
                        "INSERT INTO positions (symbol, amount, avg_price, adds, strat, entry_time, max_price, sl_price, tp_price) VALUES (?,?,?,?,?,?,?,?,?)",
                        (symbol, amount, real_price, 0, strategy, now_str, real_price, sl, tp))

                await db.commit()

        ai_reason = debug_data.get('ai_reason', 'N/A')

        # Beautiful notification
        msg = (
            f"🚀 <b>{action} {symbol}</b>\n"
            f"Type: {order_type_str}\n"
            f"💵 Sum: {usd_amt:.2f}$ | Price: {real_price:.4f}\n"
            f"Strategy: {strategy}\n"
            f"🧠 AI: {ai_reason}"
        )
        await self.notify(msg)

    # --------------------------------------------------------------------------
    # 📉 EXECUTION: SELL
    # --------------------------------------------------------------------------
    async def execute_sell(self, symbol, current_price, amount_db, pnl_usd, pnl_pct, reason):
        """
        Executes sale (full position closure).
        """
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        sold = False

        if CFG.REAL_TRADING:
            try:
                # 1. Cancel hanging limits (just in case)
                try:
                    await self.ex.cancel_all_orders(symbol)
                except:
                    pass

                # 2. Check balance (avoid "Insufficient Funds")
                bal = await self.ex.fetch_balance()
                cur = symbol.split('/')[0]
                wallet = bal.get(cur, {}).get('free', 0.0)

                # 3. Calculate sell amount
                final = float(self.ex.amount_to_precision(symbol, wallet)) if wallet > 0 else amount_db

                if final > 0:
                    # 4. Sell Market
                    await self.ex.create_market_sell_order(symbol, final)
                    print(f"📉 REAL SOLD {symbol} | PnL: {pnl_usd:.2f}$")
                    sold = True
            except Exception as e:
                print(f"⚠️ SELL ERROR {symbol}: {e}")
        else:
            print(f"🔵 SIM SOLD {symbol} | PnL: {pnl_usd:.2f}$")
            sold = True

        if sold:
            async with self.lock:
                async with aiosqlite.connect(self.db) as db:
                    # Save history
                    await db.execute("INSERT INTO history (symbol, pnl, reason, date) VALUES (?,?,?,?)",
                                     (symbol, pnl_usd, reason, date))
                    # Remove from active
                    await db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
                    await db.commit()

            # Set Cooldown for 5 minutes for this coin
            self.cooldowns[symbol] = datetime.now() + timedelta(minutes=5)

            icon = "🟢" if pnl_usd > 0 else "🔴"
            await self.notify(
                f"{icon} <b>SELL {symbol}</b>\nReason: {reason}\n💰 PnL: {pnl_usd:+.2f}$ ({pnl_pct * 100:+.2f}%)")

    # --------------------------------------------------------------------------
    # 👀 WATCHDOG v2: SMART TRAILING
    # --------------------------------------------------------------------------
    async def watchdog_task(self):
        print("👀 Watchdog: ON (Active Protection v2)")
        while self.running:
            try:
                async with self.lock:
                    async with aiosqlite.connect(self.db) as db:
                        rows = await (await db.execute(
                            "SELECT symbol, amount, avg_price, max_price, entry_time FROM positions")).fetchall()

                if not rows:
                    await asyncio.sleep(3)
                    continue

                for row in rows:
                    sym, amt, avg, max_p, entry = row

                    try:
                        ticker = await self.ex.fetch_ticker(sym)
                        cur = ticker['last']
                    except:
                        continue

                    # Update High
                    if max_p is None or cur > max_p:
                        max_p = cur
                        async with self.lock:
                            async with aiosqlite.connect(self.db) as db:
                                await db.execute("UPDATE positions SET max_price=? WHERE symbol=?", (max_p, sym))
                                await db.commit()

                    pnl_usd = (cur - avg) * amt
                    pnl_pct = (cur - avg) / avg
                    max_roi = (max_p - avg) / avg  # Max profit reached

                    should_sell = False
                    reason = "UNKNOWN"

                    # 1. ⛔ HARD STOP (Classic Stop Loss)
                    if cur < avg * CFG.DEFAULT_SL:
                        should_sell = True;
                        reason = "⛔ HARD STOP"

                    # 2. 🛡 BREAKEVEN
                    # If price went up 1.5%, move stop to 0 (+0.2%)
                    elif max_roi > 0.015 and cur < avg * 1.002:
                        should_sell = True;
                        reason = "🛡 BREAKEVEN (Saved)"

                    # 3. 💸 SMART TRAILING (Profit taking)
                    # If profit was 3% to 7% -> Stop at 1% below high
                    elif 0.03 < max_roi <= 0.07:
                        stop_price = max_p * 0.99
                        if cur < stop_price:
                            should_sell = True;
                            reason = "💰 TRAILING (Tight)"

                    # If profit flew > 7% -> Give 2.5% pullback (catch the rocket)
                    elif max_roi > 0.07:
                        stop_price = max_p * 0.975
                        if cur < stop_price:
                            should_sell = True;
                            reason = "🚀 TRAILING (Wide)"

                    # 4. 🆘 TIME STOP (If stuck)
                    elif (datetime.now() - datetime.strptime(entry, "%Y-%m-%d %H:%M:%S")).total_seconds() > 14400:
                        if pnl_pct > -0.01:  # Exit only if loss is small
                            should_sell = True;
                            reason = "⏰ TIME STOP"

                    if should_sell:
                        await self.execute_sell(sym, cur, amt, pnl_usd, pnl_pct, reason)

            except Exception as e:
                pass

            await asyncio.sleep(3)

    # --------------------------------------------------------------------------
    # 📅 REPORT SYSTEM
    # --------------------------------------------------------------------------
    async def daily_report_task(self):
        """
        Background task for daily reports (stub for now).
        """
        while self.running:
            await asyncio.sleep(3600)

    # ==========================================================================
    # 🔄 MAIN LOOP: INFINITE WORK CYCLE
    # ==========================================================================
    async def loop(self):
        """
        Main Loop: Scanner -> C++ Math (Telescope) -> Microstructure -> AI -> Trade
        """
        await self.init_db()
        try:
            await self.ex.load_markets()
            await self.update_dynamic_list()
            self.last_list_update = datetime.now()
        except Exception as e:
            print(f"⚠️ Failed to load markets: {e}")

        print(f"✅ TITAN AI v6.0 STARTED (MODE: TITANIUM).")
        await self.notify("🚀 <b>TITAN AI STARTED!</b>")

        # Start background tasks
        if hasattr(self, 'streamer'):
            asyncio.create_task(self.streamer.run())

        asyncio.create_task(self.watchdog_task())
        asyncio.create_task(self.daily_report_task())

        while self.running:
            try:
                if CFG.PAUSED:
                    await asyncio.sleep(5)
                    continue
                # We update the top-200 list only once every 15 minutes to avoid API rate limits
                time_since_update = datetime.now() - self.last_list_update
                if time_since_update.total_seconds() > 900:  # 900 seconds = 15 min
                    await self.update_dynamic_list()
                    self.last_list_update = datetime.now()


                # 1. Check Position Limit
                async with self.lock:
                    async with aiosqlite.connect(self.db) as db:
                        count = (await (await db.execute("SELECT count(*) FROM positions")).fetchone())[0]

                if count >= CFG.MAX_OPEN_POSITIONS:
                    await asyncio.sleep(30)
                    continue

                # 2. Determine Market Regime (BTC)
                btc_df = await self.data.get_candles(CFG.BTC_SYMBOL, CFG.TIMEFRAME)
                if btc_df.empty:
                    await asyncio.sleep(10)
                    continue

                # Live-patch BTC (Update BTC with live price for regime accuracy)
                if hasattr(self, 'streamer'):
                    ws_btc = self.streamer.get_price(CFG.BTC_SYMBOL)
                    if ws_btc:
                        # Gently update last close price
                        btc_df.iloc[-1, btc_df.columns.get_loc('close')] = ws_btc

                instr = Analyzer.get_market_instruction(btc_df)
                regime = instr['regime']

                if regime == "CRASH":
                    print("📉 MARKET CRASH DETECTED. Waiting 60 sec...")
                    await asyncio.sleep(60)
                    continue

                # 3. Symbol Scanning
                cands = []
                c_map = {}

                for sym in CFG.SYMBOLS:
                    # Scan slightly faster without overloading CPU
                    await asyncio.sleep(0.05)
                    try:
                        # Filters (Cooldown)
                        if sym in self.cooldowns and datetime.now() < self.cooldowns[sym]: continue

                        # Check reputation and existing position (Lock DB for min time)
                        async with self.lock:
                            async with aiosqlite.connect(self.db) as db:
                                has_pos = (await (
                                    await db.execute("SELECT 1 FROM positions WHERE symbol=?", (sym,))).fetchone())
                        if has_pos: continue

                        # Get Candles
                        df = await self.data.get_candles(sym, CFG.TIMEFRAME, limit=100)
                        if df.empty: continue

                        # 🔥 TELESCOPE: Get Live Price
                        ws_price = None
                        if hasattr(self, 'streamer'):
                            ws_price = self.streamer.get_price(sym)

                        # If no price in streamer, take from candle
                        current_check_price = ws_price if ws_price else df['close'].iloc[-1]

                        # --- ANALYSIS 1: C++ CORE (Passing live price!) ---
                        obi_ratio = 0.5

                        # MAJOR CHANGE HERE: Passing current_price
                        sc, dbg, st = Analyzer.calculate_signal(
                            df, obi_ratio, regime, btc_df,
                            current_price=ws_price
                        )

                        # If strategy not allowed - skip
                        if st not in instr['strategy_allowed']:
                            continue

                        # Check minimum threshold
                        if sc >= CFG.MIN_SCORE:
                            print(f"🔥 Potential signal: {sym} | Score: {sc} | Strat: {st}")

                            # --- ANALYSIS 2: MICROSTRUCTURE (Jane Street Logic) ---
                            try:
                                ob = await self.ex.fetch_order_book(sym, limit=20)
                                micro = Analyzer.get_microstructure_signal(ob)

                                # If breakout but sell wall exists -> Reject
                                if st in ["SNIPE", "VOL_BREAKOUT"] and micro['imbalance'] < -0.2:
                                    print(f"🛡 {sym} rejected: Strong sell pressure ({micro['imbalance']:.2f})")
                                    continue

                                # If strong buy pressure -> Score Bonus
                                if micro['imbalance'] > 0.3:
                                    sc += 10
                                    dbg['reasons'].append(f"ORDERFLOW_PUMP_+10")

                                # WHALES (Tape reading)
                                trades = await self.ex.fetch_trades(sym, limit=50)
                                whale_bonus = Analyzer.check_whale_activity(trades)
                                if whale_bonus > 0:
                                    sc += (whale_bonus * 5)
                                    dbg['reasons'].append(f"WHALES_DETECTED")

                                pkg = {
                                    'symbol': sym,
                                    'price': current_check_price,  # Use verified price
                                    'score': sc,
                                    'strat': st,
                                    'debug': dbg,
                                    'micro_imbalance': micro['imbalance']
                                }
                                cands.append(pkg)
                                c_map[sym] = pkg

                            except Exception as e_micro:
                                print(f"⚠️ Microstructure Error {sym}: {e_micro}")
                                # If micro fails, take by tech analysis anyway
                                pkg = {'symbol': sym, 'price': current_check_price, 'score': sc, 'strat': st,
                                       'debug': dbg}
                                cands.append(pkg)
                                c_map[sym] = pkg

                    except Exception as e_sym:
                        # print(f"Err {sym}: {e_sym}") # Uncomment for debug
                        pass

                # 4. AI Filtering
                if cands:
                    # Sort: best on top
                    cands.sort(key=lambda x: x['score'], reverse=True)
                    top_candidates = cands[:3]  # Take top-3 for AI to save limits

                    print(f"🧠 Sending {len(top_candidates)} candidates to AI...")

                    # Send to AI
                    decisions = await self.ai.analyze_batch(top_candidates, regime)

                    for d in decisions:
                        # Check position limit again before buy
                        if count >= CFG.MAX_OPEN_POSITIONS: break

                        if d.get('decision') == "BUY":
                            # Check AI confidence
                            if d.get('confidence', 0) < instr['ai_confidence']:
                                print(f"🤔 AI unsure about {d.get('symbol')} ({d.get('confidence')}/10)")
                                continue

                            s_name = d.get('symbol')
                            # Normalize to BTC/USDT format (just in case)
                            if "/USDT" not in s_name:
                                s_name = s_name.replace("USDT", "/USDT")

                            if s_name in c_map:
                                data_pkg = c_map[s_name]
                                data_pkg['ai_reason'] = d.get('reason')
                                data_pkg['confidence'] = d.get('confidence')

                                await self.execute_buy(s_name, data_pkg['price'], data_pkg['score'], data_pkg)
                                count += 1  # Increment local counter

            except Exception as e:
                print(f"🆘 Main Loop Error: {e}")
                await asyncio.sleep(5)

            # Pause between market scans
            # 10 seconds is optimal. Streamer works in background, so data is fresh.
            await asyncio.sleep(10)

    async def shutdown(self):
        print("🛑 SHUTTING DOWN BOT...")
        self.running = False
        # Close exchange session
        if hasattr(self, 'session') and self.session:
            await self.session.close()
        # Close exchange connection
        if hasattr(self, 'ex') and self.ex:
            await self.ex.close()
        print("👋 Bot stopped successfully.")
