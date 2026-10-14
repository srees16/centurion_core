"""US equities on the NSE engine's rules (tracker docs/US_MEMORY_TRACKER.txt).

``us_engine.data`` loads today's S&P 500 members from yfinance as a
``MarketData`` panel; ``us_engine.advisory`` runs the advisory book (US-A1).
The India book is untouched: these modules only call ``nse_engine``.
"""
