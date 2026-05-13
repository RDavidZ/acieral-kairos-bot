"""
deploy/seed_dashboard.py — Seed Supabase dashboard with backtest history.

Run once after VPS deployment to populate the dashboard with backtest
trade history and equity curve from engine_with_exit results.

Usage:
    python -m deploy.seed_dashboard
"""

import time as _time
import pandas as pd
from pathlib import Path

import config

BOT_ID      = "acieral_kairos_v1"
RESULTS_DIR = Path("backtest/results_exit")


def _clear_bot(bot_id: str) -> None:
    """Delete all trades and equity snapshots for bot_id before re-seeding."""
    from dashboard.db import _connection  # type: ignore[import]
    with _connection() as conn:
        if conn is None:
            return
        with conn.cursor() as cur:
            cur.execute("DELETE FROM equity_snapshots WHERE bot_id = %s", (bot_id,))
            cur.execute("DELETE FROM trades WHERE bot_id = %s", (bot_id,))
        conn.commit()
    print(f"Cleared existing data for {bot_id}")


def main() -> None:
    from dashboard.db import init_db, register_bot, bulk_write_trades, bulk_write_equity_snapshots  # type: ignore[import]

    t0 = _time.time()

    print("=== Seeding dashboard ===")
    print(f"Bot ID : {BOT_ID}")
    print(f"Source : {RESULTS_DIR}")
    print()

    init_db()
    _clear_bot(BOT_ID)

    active = config.DISCOVERED_PARAMS["ACTIVE_INSTRUMENTS"]

    register_bot(
        BOT_ID, "Acieral Kairos Bot v1",
        active, 10000.0,
        product="acieral", is_live=False,
        data_source="practice", lifecycle_stage="practice",
    )
    print(f"Bot registered | {BOT_ID}")
    print()

    total_trades = 0
    equity_seeded = False

    for instrument in active:
        trades_path = RESULTS_DIR / f"{instrument}_trades.csv"
        if not trades_path.exists():
            print(f"  {instrument}: no trades file — skipping")
            continue

        df = pd.read_csv(trades_path)
        df = df[df["exit_reason"].notna()].copy()

        if df.empty:
            print(f"  {instrument}: 0 closed trades")
            continue

        trades = []
        for _, row in df.iterrows():
            def safe(key: str, cast=float):
                v = row.get(key)
                return cast(v) if pd.notna(v) else None

            direction = "LONG" if safe("direction", int) == 1 else "SHORT"

            trades.append({
                "bot_id":       BOT_ID,
                "pair":         instrument,
                "direction":    direction,
                "entry_time":   str(row["entry_time"])[:19],
                "exit_time":    str(row["exit_time"])[:19] if pd.notna(row.get("exit_time")) else None,
                "entry_price":  safe("entry_price"),
                "exit_price":   safe("close_price"),
                "sl_price":     safe("sl_price"),
                "units":        safe("units"),
                "pnl_gbp":      safe("pnl_gbp"),
                "exit_reason":  str(row["exit_reason"]) if pd.notna(row.get("exit_reason")) else None,
                "bars_held":    safe("bars_held", int),
                "mfe_price":    safe("mfe_price"),
                "mae_price":    safe("mae_price"),
                "confidence":   safe("confidence"),
                "atr_at_entry": safe("atr_at_entry"),
                "status":       "closed",
                "trade_type":   "backtest",
            })

        inserted = bulk_write_trades(trades)
        total_trades += inserted
        print(f"  {instrument}: {inserted} trades seeded")

        if not equity_seeded:
            eq_path = RESULTS_DIR / f"{instrument}_equity.csv"
            if eq_path.exists():
                eq_df = pd.read_csv(eq_path)
                snapshots = [{"bot_id": BOT_ID, "equity": float(r["equity"]), "source": "backtest"} for _, r in eq_df.iterrows()]
                eq_count = bulk_write_equity_snapshots(snapshots)
                print(f"  Equity curve seeded from {instrument} ({eq_count} snapshots)")
                equity_seeded = True

    elapsed = _time.time() - t0
    print()
    print(f"Total trades seeded : {total_trades}")
    print(f"Equity curve seeded : {equity_seeded}")
    print(f"Elapsed             : {elapsed:.1f}s")
    print()
    print(f"Dashboard: https://acieral.aureyn.io/bots/{BOT_ID}")


if __name__ == "__main__":
    main()
