import sys
import asyncio

# ==========================================================
# 🚑 WINDOWS ERROR FIX (DNS / SSL)
# (Must be the very first line of code)
# ==========================================================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import aiosqlite
from aiogram import types
from aiogram.filters import Command
from datetime import datetime, timezone
import orjson
# Imports of your modules
from config import CFG
from engine import BotEngine
from analyzer import Analyzer

# ==========================================================
# 🔇 TRASH LOG SILENCER
# Keeps the console clean and beautiful
# ==========================================================
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Main logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot_crash_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==========================================================
# ==========================================================
# 🕒 ADAPTIVE SESSION MONITOR (CHAMELEON + SMART HANDOVER)
# ==========================================================
# ==========================================================

async def market_mood_monitor(bot_engine):
    print("🦎 Session Monitor: STARTED")
    await asyncio.sleep(5)

    # Variable to store the previous session (to catch the switch moment)
    last_session = "INIT"

    while True:
        try:
            if CFG.PAUSED:
                await asyncio.sleep(60)
                continue

            now_hour = datetime.now(timezone.utc).hour

            # Default variables
            session_name = "UNKNOWN"
            new_min_score = CFG.MIN_SCORE
            new_sl = 0.98
            new_tp = 1.04  # Standard TP

            # --- 🌏 ASIA (00:00 - 07:00 UTC) ---
            if 0 <= now_hour < 7:
                session_name = "🌏 ASIA (Scalping)"
                new_min_score = 50
                new_sl = 0.985  # SL -1.5%
                new_tp = 1.025  # TP +2.5% (Fast exit)

            # --- 🇪🇺 LONDON (07:00 - 13:00 UTC) ---
            elif 7 <= now_hour < 13:
                session_name = "🇪🇺 LONDON (Trend)"
                new_min_score = 60
                new_sl = 0.98  # SL -2.0%
                new_tp = 1.04  # TP +4.0% (Standard)

            # --- 🇺🇸 NEW YORK (13:00 - 21:00 UTC) ---
            elif 13 <= now_hour < 21:
                session_name = "🇺🇸 NEW YORK (Volatility)"
                new_min_score = 75  # Strict selection
                new_sl = 0.96  # SL -4.0% (Hold the hit)
                new_tp = 1.07  # TP +7.0% (Catch the rocket!)

            # --- 🌑 LATE NIGHT (21:00 - 00:00 UTC) ---
            else:
                session_name = "🌑 LATE NIGHT"
                new_min_score = 70
                new_sl = 0.98
                new_tp = 1.04

            # Save session name to config so AI can see it
            CFG.CURRENT_SESSION = session_name

            # 🔥 SMART HANDOVER LOGIC (Smart cleanup on session switch) 🔥
            if session_name != last_session and last_session != "INIT":
                print(f"🔄 SESSION SWITCH: {last_session} -> {session_name}")

                # If switching TO AMERICA (most dangerous time) -> Clean weak positions
                if "NEW YORK" in session_name:
                    print("🇺🇸 PREPARING FOR NY VOLATILITY: Cleaning weak positions...")
                    try:
                        async with aiosqlite.connect(bot_engine.db) as db:
                            positions = await (
                                await db.execute("SELECT symbol, amount, avg_price FROM positions")).fetchall()

                        for row in positions:
                            sym, amt, avg = row
                            # Get actual price
                            ticker = await bot_engine.ex.fetch_ticker(sym)
                            current_price = ticker['last']

                            # Calculate PnL %
                            pnl_val = (current_price - avg) * amt
                            pnl_pct = (current_price - avg) / avg

                            # CLEANUP CONDITION: If loss is greater than 0.2% (weak position)
                            if pnl_pct < -0.002:
                                print(f"🔪 Closing WEAK position {sym} ({pnl_pct * 100:.2f}%) before NY.")
                                await bot_engine.execute_sell(sym, current_price, amt, pnl_val, pnl_pct,
                                                              "SESSION_CLEANUP")
                                await asyncio.sleep(1)  # Small pause between sells

                    except Exception as clean_err:
                        print(f"⚠️ Cleanup Error: {clean_err}")

            # Update session memory
            last_session = session_name

            # Check Bitcoin (Global Trend)
            try:
                btc_df = await bot_engine.data.get_candles(CFG.BTC_SYMBOL, "4h")
                if not btc_df.empty:
                    ai_score, reason = await bot_engine.ai.get_market_sentiment(btc_df)
                    # If BTC is falling, tighten the screws (minimum 80 score)
                    if ai_score < 50:
                        new_min_score = max(new_min_score, 80)
            except:
                pass

            # Apply settings (If something changed)
            if CFG.MIN_SCORE != new_min_score or CFG.DEFAULT_SL != new_sl or CFG.DEFAULT_TP != new_tp:
                CFG.MIN_SCORE = new_min_score
                CFG.DEFAULT_SL = new_sl
                CFG.DEFAULT_TP = new_tp

                msg = (
                    f"🦎 <b>CHAMELEON MODE ACTIVATE</b>\n"
                    f"⏰ Session: <b>{session_name}</b>\n"
                    f"📊 Score > <b>{new_min_score}</b>\n"
                    f"🛡 SL: {(1 - new_sl) * 100:.1f}% | 🎯 TP: {(new_tp - 1) * 100:.1f}%"
                )
                try:
                    await bot_engine.bot.send_message(CFG.TG_ADMIN_ID, msg)
                except:
                    pass
                print(f"✅ Config Updated: {session_name}")

        except Exception as e:
            print(f"⚠️ Session Monitor Error: {e}")

        await asyncio.sleep(900)


async def main():
    # 1. Create engine
    bot_engine = BotEngine()

    # ==========================================================
    # 🎮 TELEGRAM HANDLERS (ALL COMMANDS)
    # ==========================================================

    @bot_engine.dp.message(Command("start"))
    async def c_start(m: types.Message):
        await m.answer(
            f"🤖 <b>Titan AI v6.0 (Full Power)</b>\n\n"
            f"🎮 <b>CONTROLS:</b>\n"
            f"/run - ▶️ Start trading\n"
            f"/stop - ⏸ Pause (sells only)\n"
            f"/liquidate - ☢️ SELL ALL (Emergency)\n"
            f"/sell [BTC] - Sell specific coin\n\n"
            f"🧠 <b>INTELLIGENCE:</b>\n"
            f"/ai [BTC] - 👁 Gemini (Text + Walls + Vision)\n"
            f"/scan - 🔍 Market Scan (C++ Math)\n\n"
            f"📊 <b>INFO:</b>\n"
            f"/balance - 💼 Portfolio\n"
            f"/stats - 📈 Statistics\n"
            f"/history - 📜 History\n"
            f"/config - ⚙️ Config\n"
            f"/sync - 🔄 Sync DB"
        )

    @bot_engine.dp.message(Command("run"))
    async def c_run(m: types.Message):
        CFG.PAUSED = False
        bot_engine.cooldowns.clear()
        await m.answer(f"▶️ <b>Trading STARTED</b>\nEntry Threshold: {CFG.MIN_SCORE}")

    @bot_engine.dp.message(Command("stop"))
    async def c_stop(m: types.Message):
        CFG.PAUSED = True
        await m.answer("⏸ <b>PAUSED</b>\nBuys stopped. Sells are active.")

    @bot_engine.dp.message(Command("stats"))
    async def c_stats(m: types.Message):
        try:
            async with aiosqlite.connect(bot_engine.db) as db:
                row_total = await (await db.execute("SELECT SUM(pnl), COUNT(*) FROM history")).fetchone()
                total_pnl = row_total[0] if row_total and row_total[0] else 0.0
                total_trades = row_total[1] if row_total else 0

                row_wins = await (await db.execute("SELECT COUNT(*) FROM history WHERE pnl > 0")).fetchone()
                wins = row_wins[0] if row_wins else 0

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            icon = "🤑" if total_pnl > 0 else "📉"
            text = (f"<b>📊 TITAN STATS:</b>\n{icon} PnL: {total_pnl:+.2f}$\n"
                    f"✅ Wins: {wins}/{total_trades} ({win_rate:.1f}%)")
            await m.answer(text)
        except Exception as e:
            await m.answer(f"Stats error: {e}")

    @bot_engine.dp.message(Command("balance"))
    async def c_bal(m: types.Message):
        msg = await m.answer("⏳ Calculating portfolio...")
        try:
            async with aiosqlite.connect(bot_engine.db) as db:
                rows = await (await db.execute("SELECT symbol, amount, avg_price, strat FROM positions")).fetchall()

            if not rows:
                await msg.edit_text("📭 Portfolio is empty.")
                return

            text = "<b>💼 ACTIVE POSITIONS:</b>\n"
            total_unrealized = 0.0

            for r in rows:
                sym, amt, avg, strat = r
                try:
                    ticker = await bot_engine.ex.fetch_ticker(sym)
                    cur = ticker['last']
                except:
                    cur = avg

                val_now = amt * cur
                val_buy = amt * avg
                pnl = val_now - val_buy
                pct = (pnl / val_buy * 100) if val_buy != 0 else 0

                total_unrealized += pnl
                text += f"🔹 <b>{sym}</b>: {pct:+.2f}% ({pnl:+.2f}$)\n"

            text += f"\n💵 Floating PnL: {total_unrealized:+.2f}$"
            await msg.edit_text(text)
        except Exception as e:
            await msg.edit_text(f"Err: {e}")

    @bot_engine.dp.message(Command("history"))
    async def c_hist(m: types.Message):
        try:
            # Connect to database
            async with aiosqlite.connect(bot_engine.db) as db:
                try:
                    cursor = await db.execute("SELECT symbol, pnl, reason FROM history ORDER BY id DESC LIMIT 10")
                    rows = await cursor.fetchall()
                    await cursor.close()
                except Exception as db_err:
                    await m.answer(f"⚠️ Database error (table might be empty): {db_err}")
                    return

            if rows:
                text = "📜 <b>LAST TRADES:</b>\n"
                for r in rows:
                    sym, pnl, reason = r
                    if pnl is None: pnl = 0.0
                    if reason is None: reason = "Unknown"
                    color = "🟢" if pnl >= 0 else "🔴"
                    text += f"{color} <b>{sym}</b>: {pnl:+.2f}$ <i>({reason})</i>\n"
            else:
                text = "📜 <b>History empty.</b>\nBot hasn't sold anything yet."

            await m.answer(text)
        except Exception as e:
            await m.answer(f"❌ History error: {e}")

    @bot_engine.dp.message(Command("sync"))
    async def c_sync(m: types.Message):
        res = await m.answer("⏳ Syncing DB with exchange...")
        try:
            await bot_engine.sync_db_with_exchange()
            await res.edit_text("✅ Database synced with exchange balance.")
        except Exception as e:
            await res.edit_text(f"Sync error: {e}")

    @bot_engine.dp.message(Command("liquidate"))
    async def c_liquidate(m: types.Message):
        """SELL ALL POSITIONS AT MARKET PRICE"""
        msg = await m.answer("☢️ <b>STARTING FULL LIQUIDATION...</b>")

        async with bot_engine.lock:
            async with aiosqlite.connect(bot_engine.db) as db:
                rows = await (await db.execute("SELECT symbol, amount, avg_price FROM positions")).fetchall()

        if not rows:
            await msg.edit_text("🤷‍♂️ Portfolio empty, nothing to sell.")
            return

        sold_count = 0
        log_text = ""

        for row in rows:
            sym, amount, avg = row
            try:
                try:
                    await bot_engine.ex.cancel_all_orders(sym)
                except:
                    pass

                ticker = await bot_engine.ex.fetch_ticker(sym)
                price = ticker['last']

                await bot_engine.ex.create_market_sell_order(sym, amount)

                async with aiosqlite.connect(bot_engine.db) as db:
                    await db.execute("DELETE FROM positions WHERE symbol=?", (sym,))
                    await db.commit()

                sold_count += 1
                log_text += f"✅ Sold {sym} @ {price}\n"

            except Exception as e:
                log_text += f"❌ Error {sym}: {e}\n"

        await msg.edit_text(f"<b>☢️ LIQUIDATION COMPLETE</b>\nSold: {sold_count}\n\n{log_text}")

    @bot_engine.dp.message(Command("sell"))
    async def c_force_sell(m: types.Message):
        args = m.text.split()
        if len(args) < 2: return await m.answer("⚠️ Example: /sell BTC")

        raw_symbol = args[1].upper()
        symbol = raw_symbol if "/USDT" in raw_symbol else f"{raw_symbol}/USDT"
        status = await m.answer(f"⏳ Selling {symbol}...")

        async with bot_engine.lock:
            async with aiosqlite.connect(bot_engine.db) as db:
                row = await (
                    await db.execute("SELECT amount, avg_price FROM positions WHERE symbol=?", (symbol,))).fetchone()

        if not row: return await status.edit_text("❌ No such position in DB.")

        try:
            ticker = await bot_engine.ex.fetch_ticker(symbol)
            cur_price = ticker['last']
            amount, avg = row

            pnl = (amount * cur_price * (1 - CFG.FEE_RATE)) - (amount * avg)
            cost = amount * avg
            pct = (pnl / cost) if cost != 0 else 0

            await bot_engine.execute_sell(symbol, cur_price, amount, pnl, pct, "FORCE_SELL")
            await status.edit_text(f"✅ <b>{symbol} SOLD!</b>\nPnL: {pnl:.2f}$ ({pct * 100:.2f}%)")
        except Exception as e:
            await status.edit_text(f"Sell error: {e}")

    @bot_engine.dp.message(Command("config"))
    async def c_conf(m: types.Message):
        """Show current config"""
        status = "🔴 PAUSED" if CFG.PAUSED else "🟢 RUNNING"
        text = (
            f"⚙️ <b>TITAN CONFIG</b>\n"
            f"Status: {status}\n"
            f"Mode: {'💵 REAL MONEY' if CFG.REAL_TRADING else '🔫 SIMULATION'}\n"
            f"Order Size: {CFG.BASE_ORDER_SIZE}$\n"
            f"Max Pos: {CFG.MAX_OPEN_POSITIONS}\n"
            f"AI Judge: {'✅ ON' if CFG.USE_AI else '❌ OFF'}\n"
            f"Min Score (Dynamic): <b>{CFG.MIN_SCORE}</b>\n"
            f"Current Session: <b>{getattr(CFG, 'CURRENT_SESSION', 'N/A')}</b>"
        )
        await m.answer(text)

    @bot_engine.dp.message(Command("scan"))
    async def c_scan(message: types.Message):
        ms = await message.answer("🔍 Scanning market (Math + C++)...")
        try:
            # 1. Global Regime
            btc_df = await bot_engine.data.get_candles(CFG.BTC_SYMBOL, CFG.TIMEFRAME)
            btc_instr = Analyzer.get_market_instruction(btc_df)
            regime = btc_instr['regime']

            text = f"🌍 BTC Regime: <b>{regime}</b>\n\n"

            # 2. Scan coins
            count = 0
            candidates = await bot_engine.data.get_top_candidates(limit=200)
            for s in candidates:
                df = await bot_engine.data.get_candles(s, CFG.TIMEFRAME)
                if df.empty: continue

                obi = await bot_engine.data.get_order_book_imbalance(s)
                # Call C++ analyzer
                sc, dbg, st = Analyzer.calculate_signal(df, obi, regime, btc_df)

                if sc > 40:
                    icon = "🟢" if sc >= CFG.MIN_SCORE else "⚪️"
                    text += f"{icon} <b>{s}</b>: Score {sc} | RSI {dbg.get('rsi', 0):.0f} | {st}\n"
                    count += 1

            if count == 0: text += "No interesting situations."
            text += f"\n⚙️ Aggression: <b>Min Score {CFG.MIN_SCORE}</b>"
            await ms.edit_text(text)

        except Exception as e:
            await ms.edit_text(f"Scan error: {e}")

    # --- 🔥 FULL COMMAND /AI WITH "EYES", "WALLS" AND "LIQUIDATIONS" 🔥 ---
    @bot_engine.dp.message(Command("ai"))
    async def c_manual_ai(m: types.Message):
        """
        Enhanced manual AI request with double ticker check and error protection.
        """
        args = m.text.split()
        if len(args) < 2:
            return await m.answer("📝 Example: <code>/ai DOGE</code>")

        # 1. Format user input
        user_input = args[1].upper().strip()
        clean_name = user_input.replace("/", "").replace("USDT", "")

        # Variants to check
        sym_variants = [f"{clean_name}/USDT", f"{clean_name}USDT"]

        status = await m.answer(f"🧠 <b>Titan AI analyzing {clean_name}...</b>")

        try:
            # 2. Find data (OHLCV)
            df = None
            final_sym = ""

            for variant in sym_variants:
                try:
                    temp_df = await bot_engine.data.get_candles(variant, CFG.TIMEFRAME)
                    if temp_df is not None and not temp_df.empty:
                        df = temp_df
                        final_sym = variant
                        break
                except:
                    continue

            if df is None or df.empty:
                return await status.edit_text(f"❌ Data not found for <b>{user_input}</b> on MEXC.\n"
                                              f"Check if this pair exists vs USDT.")

            # 3. Gather market context (BTC and Regime)
            btc_df = await bot_engine.data.get_candles(CFG.BTC_SYMBOL, CFG.TIMEFRAME)
            if btc_df.empty:
                return await status.edit_text("⚠️ Error: Failed to fetch BTC data for regime analysis.")

            market_info = Analyzer.get_market_instruction(btc_df)
            regime = market_info['regime']

            # 4. Math Analysis (C++ Core + Orderbook)
            obi = await bot_engine.data.get_order_book_imbalance(final_sym)
            score, debug, strategy = Analyzer.calculate_signal(df, obi, regime, btc_df)

            # Orderbook Walls
            orderbook = await bot_engine.ex.fetch_order_book(final_sym, limit=100)
            walls = Analyzer.detect_liquidity_walls(orderbook, df['close'].iloc[-1])

            # Liquidation Heatmap (Math Model)
            liq_heatmap = Analyzer.estimate_liquidation_heatmap(df)
            current_price = df['close'].iloc[-1]

            # Nearby liquidation clusters (within 5%)
            nearby_liqs = [
                f"{l['type']} ({l['leverage']}) @ {l['price']:.2f}"
                for l in liq_heatmap
                if abs(current_price - l['price']) / current_price < 0.05
            ]

            # 5. Prepare data for Gemini
            candles_summary = df.tail(5)[['high', 'low', 'close', 'volume']].to_string()
            walls_info = (
                f"ORDERBOOK: Support {walls['support'][:2]}, Resistance {walls['resistance'][:2]}\n"
                f"LIQUIDATIONS: {nearby_liqs if nearby_liqs else 'No major clusters nearby'}"
            )

            ai_payload = [{
                'symbol': final_sym,
                'price': current_price,
                'math_score': score,
                'recent_candles': candles_summary,
                'liquidity_walls': walls_info
            }]

            # 6. Text Analysis (Gemini Text)
            decisions = await bot_engine.ai.analyze_batch(ai_payload, regime, f"BTC is {regime}")
            if not decisions:
                return await status.edit_text("⚠️ AI could not form a verdict.")

            res = decisions[0]

            # 7. Visual Analysis (Gemini Vision)
            await status.edit_text(f"👁 <b>Gemini Vision scanning chart {final_sym}...</b>")

            # Generate chart image in memory
            chart_file = await asyncio.to_thread(bot_engine.ai._generate_chart_bytes, df, final_sym)
            vision_res = await bot_engine.ai.analyze_with_vision(df, final_sym, score)

            # 8. Final Report Formatting
            v_verdict = vision_res.get('visual_verdict', 'N/A') if vision_res else "N/A"
            v_pattern = vision_res.get('pattern', 'None') if vision_res else "None"

            report_caption = (
                f"📊 <b>REPORT: {final_sym}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🤖 <b>Verdict:</b> {res.get('decision', 'SKIP')}\n"
                f"🎯 <b>Confidence:</b> {res.get('confidence', 0)}/10\n"
                f"👁 <b>Vision:</b> {v_verdict} ({v_pattern})\n"
                f"📈 <b>Score:</b> {score} | <b>Regime:</b> {regime}\n\n"
                f"🧱 <b>Walls:</b> Sup {len(walls['support'])} | Res {len(walls['resistance'])}\n"
                f"🩸 <b>Liq Clusters:</b> {len(nearby_liqs)}\n\n"
                f"🛡 <b>SL:</b> <code>{res.get('stop_loss', 'N/A')}</code>\n"
                f"🎯 <b>TP:</b> <code>{res.get('take_profit', 'N/A')}</code>\n\n"
                f"📝 <b>Analysis:</b> <i>{res.get('reason', '-')}</i>"
            )

            # 9. Send Result with Photo
            photo = types.BufferedInputFile(chart_file.getvalue(), filename=f"{final_sym}.png")
            await m.answer_photo(photo=photo, caption=report_caption)
            await status.delete()

        except Exception as e:
            await status.edit_text(f"❌ Error analyzing {user_input}: {str(e)}")

    # ==========================================================
    # APP STARTUP
    # ==========================================================
    try:
        await bot_engine.bot.delete_webhook(drop_pending_updates=True)
        print("Checking Database...")
        await bot_engine.init_db()

        print("Titan AI Starting...")

        # Run concurrently: Trade Loop + Telegram + Mood Monitor
        await asyncio.gather(
            bot_engine.loop(),
            bot_engine.dp.start_polling(bot_engine.bot),
            market_mood_monitor(bot_engine)
        )

    except Exception as e:
        error_msg = f"<b>CRITICAL ERROR!</b>\n<code>{str(e)}</code>"
        print(f"MAIN LOOP CRASH: {e}")
        try:
            await bot_engine.bot.send_message(CFG.TG_ADMIN_ID, error_msg)
        except:
            pass
        raise e

    finally:
        await bot_engine.shutdown()


# ==========================================================
# 🚑 UNIVERSAL LAUNCHER (WINDOWS + LINUX)
# Insert this at the very bottom of main.py
# ==========================================================
if __name__ == "__main__":
    import time
    import sys
    import asyncio  # Just in case

    # 1. LINUX TURBO (Server only)
    if sys.platform != 'win32':
        try:
            import uvloop

            uvloop.install()
            print("TURBO MODE: Active (uvloop installed)")
        except ImportError:
            print("uvloop not found. Running on standard engine.")

    # Infinite Reanimation Loop
    while True:
        try:
            # 2. WINDOWS FIX (Asyncio fix for Win)
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            # BOT START
            print("Bot starting...")
            # Ensure main() is defined above
            asyncio.run(main())

            print("Bot finished normally.")
            break

        except KeyboardInterrupt:
            print("Stopped by user (Ctrl+C).")
            break

        except Exception as e:
            print(f"CRASH: {e}")
            print("Restarting in 5 seconds...")
            time.sleep(5)
