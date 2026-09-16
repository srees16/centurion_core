"""
Monitoring layer — latency, health, alerts and audit trail.

The other five "institutional layer" facades (market_data, alpha_research,
risk_engine, execution_engine, portfolio) were never wired to anything and
were removed; the live implementations live in nse_engine/, services/ and
kite_connect/.  Only ``monitoring`` has a consumer (api/routers/health.py).
"""
