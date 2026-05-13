"""
execution/price_stream.py — Real-time OANDA price streaming for trade management.

Runs in a daemon thread alongside the H1 APScheduler loop.
Only handles trail SL updates and SL hit detection — no entry evaluation.
Entry evaluation and exit model remain on the H1 candle close scheduler.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import requests

log = logging.getLogger(__name__)


class PriceStream:
    """
    Streams real-time bid/ask prices from OANDA for active instruments.
    Calls on_tick(instrument, bid, ask, now) for every price update.
    Reconnects automatically on stream drop.
    """

    def __init__(
        self,
        api_key: str,
        account_id: str,
        instruments: list[str],
        on_tick: Callable[[str, float, float, datetime], None],
        environment: str = "practice",
    ) -> None:
        self.api_key     = api_key
        self.account_id  = account_id
        self.instruments = instruments
        self.on_tick     = on_tick
        self.environment = environment
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        base = (
            "https://stream-fxpractice.oanda.com"
            if environment == "practice"
            else "https://stream-fxtrade.oanda.com"
        )
        self._url = (
            f"{base}/v3/accounts/{account_id}/pricing/stream"
            f"?instruments={'%2C'.join(instruments)}"
        )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept-Encoding": "gzip, deflate",
        }

    def start(self) -> None:
        """Start the streaming thread as a daemon."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="PriceStream", daemon=True
        )
        self._thread.start()
        log.info("PriceStream started | instruments=%d", len(self.instruments))

    def stop(self) -> None:
        """Signal the streaming thread to stop."""
        self._stop_event.set()
        log.info("PriceStream stop requested")

    def _run(self) -> None:
        """Main streaming loop with automatic reconnection."""
        backoff = 2.0
        while not self._stop_event.is_set():
            try:
                log.info("PriceStream connecting...")
                with requests.get(
                    self._url,
                    headers=self._headers,
                    stream=True,
                    timeout=30,
                ) as resp:
                    resp.raise_for_status()
                    backoff = 2.0  # reset backoff on successful connect
                    log.info("PriceStream connected")
                    for line in resp.iter_lines():
                        if self._stop_event.is_set():
                            break
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            if msg.get("type") != "PRICE":
                                continue
                            instrument = msg["instrument"]
                            bid = float(msg["bids"][0]["price"])
                            ask = float(msg["asks"][0]["price"])
                            now = datetime.now(timezone.utc)
                            self.on_tick(instrument, bid, ask, now)
                        except Exception as exc:
                            log.debug("PriceStream parse error: %s", exc)
                            continue

            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.warning(
                    "PriceStream disconnected: %s — reconnecting in %.0fs",
                    exc, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        log.info("PriceStream stopped")
