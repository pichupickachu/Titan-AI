import asyncio
import aiohttp
import ssl
import json
import socket
import logging
import websockets
from aiohttp.resolver import AsyncResolver


class MexcStreamer:
    def __init__(self, symbols):
        # Format symbols for different APIs
        self.symbols_raw = [s.replace('/', '') for s in symbols]
        self.symbols_formatted = [s.replace("USDT", "/USDT") if "USDT" in s else s for s in self.symbols_raw]

        self.latest_prices = {}
        self.running = True
        self.tick_counter = 0

        # REST Configuration (Backup)
        self.api_url = "https://api.mexc.com/api/v3/ticker/price"

        # WebSocket Configuration (Main Stream)
        self.ws_url = "wss://wbs.mexc.com/ws"

    async def run_ws(self):
        """Main stream via WebSockets (Fast Data)"""
        print("🚀 WebSocket DRIVER STARTED (Direct Quote Stream)...")
        while self.running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    # Subscribe to deals (trades)
                    subscribe_msg = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.deals.v3.api@{s}" for s in self.symbols_raw]
                    }
                    await ws.send(json.dumps(subscribe_msg))

                    async for message in ws:
                        data = json.loads(message)

                        # Update price from deals stream
                        if "d" in data and "deals" in data["d"]:
                            symbol = data["s"].replace("USDT", "/USDT")
                            # Get the price of the last deal
                            last_price = float(data["d"]["deals"][0]["p"])
                            self.latest_prices[symbol] = last_price

            except Exception as e:
                print(f"📡 WS Connection interrupted: {e}. Reconnecting...")
                await asyncio.sleep(5)

    async def run_rest(self):
        """Backup stream via REST (Every second)"""
        print("🚜 REST DRIVER ACTIVE (Log every 30 sec)...")

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        resolver = AsyncResolver(nameservers=["8.8.8.8", "1.1.1.1"])

        connector = aiohttp.TCPConnector(resolver=resolver, family=socket.AF_INET, ssl=ssl_ctx)

        async with aiohttp.ClientSession(connector=connector) as session:
            while self.running:
                try:
                    async with session.get(self.api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            for item in data:
                                symbol = item['symbol']
                                if symbol in self.symbols_raw:
                                    formatted_sym = symbol.replace("USDT", "/USDT")
                                    # REST updates price if socket is silent
                                    self.latest_prices[formatted_sym] = float(item['price'])

                            # HEARTBEAT LOGIC (Keep the ping)
                            self.tick_counter += 1
                            if self.tick_counter % 30 == 0:
                                btc = self.get_price('BTC/USDT')
                                print(f"💓 [Ping] Bot is running. BTC: {btc}")
                        else:
                            print(f"⚠️ REST API Error: {response.status}")

                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"❌ REST Error: {e}")
                    await asyncio.sleep(2)

    async def run(self):
        """Launch both drivers simultaneously"""
        # Bot will use both Sockets and REST for maximum reliability
        await asyncio.gather(
            self.run_ws(),
            self.run_rest()
        )

    def get_price(self, symbol):
        return self.latest_prices.get(symbol)
