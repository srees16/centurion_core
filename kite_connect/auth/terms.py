"""
The terms an account holder accepts before Centurion manages their Zerodha
account (tracker MU1).

A DRAFT for a securities lawyer to review before any connected account
trades: plain language, versioned, and accepted by the holder themselves, on
the page Zerodha's login returns them to, so the record
(``accounts.record_consent``: version, time, Zerodha user id) shows it was
them.  Changing the text means a new ``TERMS_VERSION``: every holder accepts
again at their next login, and until then Centurion does not trade their
account.
"""

TERMS_VERSION = "2026-10-08-draft2"

TERMS = (
    ("Centurion places buy and sell orders for Indian stocks in my Zerodha account automatically, using its "
     "configuration and strategies and the capital the operator sets; it can be stopped at any time, and with no "
     "Kite login on a day it places no orders that day."),
    ("Trading carries risk: I can lose money, up to all the capital allocated. Backtested or past results do not "
     "guarantee future returns."),
    ("Centurion uses my Kite Connect app's key and secret (stored encrypted) and my daily Kite login; it never asks "
     "for or stores my Zerodha password or TOTP."),
    ("Centurion records my account's orders, holdings and results, which its operator can see; I can ask for them "
     "to be deleted when the account is disconnected."),
    ("Centurion does not yet hold an exchange algo-provider empanelment or a SEBI registration. I use it at my own "
     "risk, and its operator may stop automatic trading in my account at any time, including when the law or "
     "Zerodha requires it."),
    "My taxes, my Zerodha charges and the funds in my account remain my responsibility.",
)
