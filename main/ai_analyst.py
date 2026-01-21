import os
import asyncio
import io
import mplfinance as mpf
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import CFG
from PIL import Image
import orjson  # 🔥 Fast JSON Serialization

load_dotenv()

class AI_Analyst:
    def __init__(self):
        """Initializes the AI Analyst with Gemini 2.0 Flash."""
        self.api_key = os.getenv("GEMINI_KEY")
        if not self.api_key:
            print("❌ ERROR: No Gemini API Key found in .env!")
            CFG.USE_AI = False
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            # 🔥 FLAGSHIP 2.0 MODEL
            self.model_name = "gemini-2.0-flash"
            print(f"✅ Gemini 2.0 Flash Connected (Orjson Powered)")
        except Exception as e:
            print(f"❌ Gemini Connection Error: {e}")
            CFG.USE_AI = False

    # =========================================================================
    # 👁 VISION MODULE (EYES)
    # =========================================================================

    def _prepare_df_for_plot(self, df, limit=60):
        """Prepares the DataFrame for chart plotting."""
        plot_df = df.tail(limit).copy()
        if 'timestamp' in plot_df.columns:
            plot_df.index = pd.to_datetime(plot_df['timestamp'], unit='ms')
        else:
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index, unit='ms')
        return plot_df

    def _generate_chart_image(self, df, symbol):
        """Generates a chart image for AI analysis (in-memory)."""
        buf = io.BytesIO()
        plot_df = self._prepare_df_for_plot(df)
        mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
        try:
            mpf.plot(plot_df, type='candle', volume=True, style=s,
                     title=f"{symbol} AI Analysis", savefig=buf, figsize=(10, 6))
        except Exception as e:
            print(f"Chart Generation Error: {e}")
            return None
        buf.seek(0)
        return Image.open(buf)

    def _generate_chart_bytes(self, df, symbol):
        """Generates chart image bytes for Telegram delivery."""
        buf = io.BytesIO()
        plot_df = self._prepare_df_for_plot(df)
        mc = mpf.make_marketcolors(up='green', down='red', edge='i', wick='i', volume='in', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
        try:
            mpf.plot(plot_df, type='candle', volume=True, style=s,
                     title=f"{symbol} Setup", savefig=buf, figsize=(12, 8))
        except:
            return None
        buf.seek(0)
        return buf

    async def analyze_with_vision(self, df, symbol, score):
        """Performs visual analysis of the chart via Gemini 2.0 Vision."""
        if not CFG.USE_AI: return None
        print(f"👁 AI Vision check for {symbol}...")
        try:
            chart_img = await asyncio.to_thread(self._generate_chart_image, df, symbol)
            if not chart_img: return None

            prompt = f"""
            Act as a Senior Technical Analyst.
            Look at the chart for {symbol}. Algo Score: {score}/100.

            CHECKLIST:
            1. Trend Structure (Higher Highs?).
            2. Candlestick Patterns (Hammer, Engulfing, Flags).
            3. Volume Confirmation.

            OUTPUT JSON:
            {{
                "visual_verdict": "BUY" or "SKIP",
                "pattern": "Bull Flag / None",
                "confidence": 8,
                "reason": "Clear breakout retest"
            }}
            """

            # 🔥 NATIVE JSON OUTPUT + SAFETY FILTERS OFF
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[prompt, chart_img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    ]
                )
            )
            return self._parse_json(response.text)
        except Exception as e:
            print(f"Vision Error: {e}")
            return None

    # =========================================================================
    # 🧠 LOGIC MODULE (BRAIN)
    # =========================================================================

    async def get_market_sentiment(self, btc_df):
        """Analyzes overall market sentiment using the BTC chart."""
        if not CFG.USE_AI: return 60, "AI Disabled"
        try:
            plot_df = self._prepare_df_for_plot(btc_df, limit=100)
            buf = io.BytesIO()
            mpf.plot(plot_df, type='candle', volume=True, style='yahoo', savefig=buf)
            buf.seek(0)
            chart_img = Image.open(buf)

            prompt = """
            Analyze BTC Chart. Determine RISK LEVEL.
            - Strong Uptrend -> Low Barrier (Score 45).
            - Choppy/Ranging -> Normal (Score 65).
            - Downtrend/Crash -> High Barrier (Score 80).
            OUTPUT JSON: {"min_score": 65, "reason": "..."}
            """

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[prompt, chart_img],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            res = self._parse_json(response.text)
            return res.get('min_score', 65), res.get('reason', 'Normal')
        except:
            return 65, "Error"

    async def analyze_batch(self, candidates_data, market_regime, global_context="Neutral"):
        """Main batch analysis for multiple coin candidates."""
        if not CFG.USE_AI or not candidates_data: return []

        # Get the current trading session
        current_session = getattr(CFG, 'CURRENT_SESSION', 'Unknown Session')

        # Detailed trading session instructions
        session_instructions = ""

        if "ASIA" in current_session:
            session_instructions = """
            🕒 SESSION: ASIA (Low Volume / Flat).
            - WARNING: Breakouts are often FAKE.
            - PREFER: Mean Reversion (buying near support, selling near resistance).
            - TARGETS: Keep Take Profit TIGHT (+1.5% to +2.5%). Don't be greedy.
            """
        elif "LONDON" in current_session:
            session_instructions = """
            🕒 SESSION: EUROPE/LONDON (Trend Start).
            - BEST TIME for Trend Following.
            - LOOK FOR: Breakouts of Asian Range.
            - TARGETS: Standard (+3% to +5%).
            """
        elif "NEW YORK" in current_session:
            session_instructions = """
            🕒 SESSION: NEW YORK (High Volatility / Wicks).
            - WARNING: Expect deep stop-hunts and long wicks (-2% drops are noise).
            - PREFER: Deep value entries. Do NOT buy the very top of a pump.
            - TARGETS: Aim HIGH (+6% to +8%) as volatility allows big moves.
            - RISK: Ensure Stop Loss is protected by a solid Wall.
            """
        else:
            session_instructions = "🕒 SESSION: Low Liquidity. Proceed with caution."

        # 🔥 FAST DATA PREPARATION (ORJSON)
        # .decode('utf-8') is mandatory for the API
        candidates_json = orjson.dumps(
            candidates_data,
            option=orjson.OPT_SERIALIZE_NUMPY
        ).decode('utf-8')

        final_prompt = f"""
        ROLE: Senior Crypto Hedge Fund Manager.
        CONTEXT: {market_regime}. 

        {session_instructions}

        INPUT DATA:
        {candidates_json}

        🔥 CRITICAL INSTRUCTION: Analyze 'liquidity_walls' & 'Heatmap':

        1. "REAL ORDERBOOK WALLS" (Hard Barriers):
           - RESISTANCE WALL above? -> Price likely REJECTS.
           - SUPPORT WALL below? -> Price likely BOUNCES. Use this for SAFE STOP LOSS.

        2. "ESTIMATED LIQUIDATION MAP" (Magnets):
           - "SHORT_LIQ" above price? -> BULLISH MAGNET (Price wants to hunt shorters).
           - "LONG_LIQ" below price? -> BEARISH MAGNET (Price wants to hunt longers).

        TASK:
        1. **PRIMARY FILTER:** Check Price Action vs Session Logic. 
           (If ASIA: Is it ranging? If NEW YORK: Is it volatile but trending?)

        2. **SECONDARY FILTER (WALLS):**
           - Do NOT buy if price is directly under a Resistance Wall.
           - BUY if price is bouncing off a Support Wall + "SHORT_LIQ" magnet is located above.

        3. **DECISION:**
           - Output BUY only if risk/reward is favorable for the CURRENT SESSION.
           - Calculate SL below the Support Wall.
           - Calculate TP based on Session targets (Tight for Asia, Wide for NY).

        OUTPUT JSON ARRAY:
        [
          {{
            "symbol": "BTC/USDT", 
            "decision": "BUY", 
            "confidence": 8, 
            "stop_loss": 94000.0, 
            "take_profit": 98000.0,
            "reason": "New York volatility breakout targeting SHORT_LIQ at 98k"
          }}
        ]
        """
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=final_prompt,
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
            )
            return self._parse_json(response.text)
        except Exception as e:
            print(f"⚠️ Text AI Error: {e}")
            return []

    async def ask_sell_advice(self, symbol, pnl_pct, reason, indicators, market_regime):
        """Asks the AI for risk management advice on whether to sell or hold."""
        if not CFG.USE_AI: return {"decision": "SELL", "reason": "AI Disabled"}
        prompt = f"Risk Manager. Trade: {symbol}. PnL: {pnl_pct * 100:.2f}%. Rules: Cut losses early, let winners run. OUTPUT JSON: {{\"decision\": \"SELL\" or \"HOLD\", \"reason\": \"...\"}}"
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return self._parse_json(response.text)
        except:
            return {"decision": "SELL", "reason": "Error"}

    # 🔥 FAST PARSING (ORJSON)
    def _parse_json(self, text):
        """Cleans and parses the AI's JSON response."""
        try:
            text = text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return orjson.loads(text)
        except:
            return {}
