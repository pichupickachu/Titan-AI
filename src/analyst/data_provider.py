import pandas as pd
import aiohttp
import logging


class MarketDataProvider:
    def __init__(self, exchange):
        self.ex = exchange
        self.markets_loaded = False

    async def ensure_markets(self):
        if not self.markets_loaded:
            try:
                await self.ex.load_markets()
                self.markets_loaded = True
            except Exception as e:
                logging.error(f"Error loading markets: {e}")

    async def get_candles(self, symbol, timeframe='15m', limit=100):
        try:
            # Use fetch_ohlcv directly. It is PUBLIC and does not require a signature.
            # We assume CCXT handles public endpoints correctly without auth params.
            ohlcv = await self.ex.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            # If the specific capital/config error still appears - ignore it in logs
            if "capital/config" not in str(e):
                print(f"❌ Data Error {symbol}: {e}")
            return pd.DataFrame()

    async def get_order_book_imbalance(self, symbol: str) -> float:
        try:
            ob = await self.ex.fetch_order_book(symbol, limit=20)
            bids = sum([x[1] for x in ob['bids']])
            asks = sum([x[1] for x in ob['asks']])
            return bids / asks if asks > 0 else 1.0
        except:
            return 1.0

    async def get_token_health(self, symbol):
        # Additional check via DexScreener (if needed)
        try:
            token_name = symbol.split('/')[0]
            url = f"https://api.dexscreener.com/latest/dex/search?q={token_name}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('pairs'):
                            return data['pairs'][0]
        except:
            pass
        return None
