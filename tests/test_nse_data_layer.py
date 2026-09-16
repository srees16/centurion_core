"""Tests for the NSE data layer: archive, store, reference, panel (no network)."""

import io
import json
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nse_engine.data import archive, panel, reference, store

LEGACY_HEADER = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,"

LEGACY_SAMPLE = LEGACY_HEADER + """
GOLDBEES,EQ,30.2,36.3,30.1,33.55,33.6,3359.6,1664422,55981467.15,19-DEC-2019,11932,INF732E01102,
RELIANCE,EQ,1573.7,1614.9,1571.8,1609.95,1613,1575.85,9375484,14960269698.6,19-DEC-2019,194045,INE002A01018,
"""

UDIFF_SAMPLE = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-09-10,2025-09-10,CM,NSE,STK,14428,INF204KB17I5,GOLDBEES,EQ,,,,,NIP IND ETF GOLD BEES,91.80,92.40,90.51,90.90,91.13,91.17,,90.90,,,29134766,2644382494.85,80775,F1,1,,,,,
2025-09-10,2025-09-10,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE INDUSTRIES LTD,1383.90,1388.50,1374.10,1377.00,1377.50,1376.20,,1377.00,,,7790815,10752266849.20,222655,F1,1,,,,,
"""

FULL_SAMPLE = """SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
GOLDBEES, EQ, 19-Dec-2019, 3359.60, 30.20, 36.30, 30.10, 33.60, 33.55, 33.63, 1664422, 559.81, 11932, 775250, 46.58
SGBX, GB, 19-Dec-2019, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10, 0.01, 1, -, -
"""

MTO_SAMPLE = """Security Wise Delivery Position - Compulsory Rolling Settlement
10,MTO,15012013,280793289,0001480
Trade Date <15-JAN-2013>,Settlement Type <N>,Settlement No <2013011>,Settlement Date <17-JAN-2013>
Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity(gross across client level),% of Deliverable Quantity to Traded Quantity
20,1,20MICRONS,EQ,599108,164405,27.44
20,2,3MINDIA,EQ,1661,1592,95.85
"""

INDEX_2013 = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
S&P CNX Nifty,15-01-2013,6037.85,6068.5,6018.6,6056.6,32.55,.54,138364003,6978.26,19.12,3.21,1.36
S&P CNX 500,15-01-2013,4700,4710,4690,4705,1,.1,1,1,1,1,1
India VIX,15-01-2013,-,-,-,14.5,-,-,-,-,-,-,-
"""

BC_LEGACY = """SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE
EQ,HCLTECH,HCL Technologies Ltd.,07/12/2019, , ,05/12/2019, , ,BONUS 1:1
EQ,GOLDBEES,Reliance ETF Gold BeES,20/12/2019, , ,19/12/2019, , ,FVSPLT FRM RS100 TO RS 1
EQ,BOROSIL,Borosil Glass Works Ltd, ,20/12/2019,26/12/2019,18/12/2019, , ,AGM/DIV- RS 0.65 PER SH
NC,SOMEBOND,Bond,2025-10-17,,,2025-10-17,,,INTEREST PAYMENT
"""

BC_ISO = """SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE
EQ,TATAMOTORS,Tata Motors Limited,2025-10-14,,,2025-10-14,,,DEMERGER
"""


def _zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


class ParserTests(unittest.TestCase):
    def test_legacy_bhavcopy_from_zip(self):
        text = store.read_raw_text(_zip_bytes("cm19DEC2019bhav.csv", LEGACY_SAMPLE))
        df = store.parse_legacy_bhavcopy(text)
        self.assertEqual(list(df.columns), store.EQUITY_COLUMNS)
        row = df.set_index("symbol").loc["GOLDBEES"]
        self.assertEqual(row["date"], pd.Timestamp("2019-12-19"))
        self.assertAlmostEqual(row["prev_close"], 3359.6)
        self.assertAlmostEqual(row["value_inr"], 55981467.15)
        self.assertEqual(row["isin"], "INF732E01102")

    def test_udiff_bhavcopy(self):
        df = store.parse_udiff_bhavcopy(UDIFF_SAMPLE)
        row = df.set_index("symbol").loc["RELIANCE"]
        self.assertEqual(row["date"], pd.Timestamp("2025-09-10"))
        self.assertAlmostEqual(row["close"], 1377.0)
        self.assertAlmostEqual(row["prev_close"], 1376.2)
        self.assertEqual(row["volume"], 7790815)
        self.assertEqual(row["trades"], 222655)
        self.assertEqual(df["volume"].dtype, np.float64)

    def test_full_bhavcopy_turnover_in_lakhs_and_delivery(self):
        df = store.parse_full_bhavcopy(FULL_SAMPLE).set_index("symbol")
        self.assertAlmostEqual(df.loc["GOLDBEES", "value_inr"], 559.81e5, places=0)
        self.assertAlmostEqual(df.loc["GOLDBEES", "deliv_pct"], 46.58)
        self.assertTrue(np.isnan(df.loc["SGBX", "deliv_qty"]))

    def test_mto(self):
        df = store.parse_mto(MTO_SAMPLE)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["date"] == pd.Timestamp("2013-01-15")).all())
        self.assertEqual(df.iloc[0]["deliv_qty"], 164405)
        self.assertAlmostEqual(df.iloc[1]["deliv_pct"], 95.85)

    def test_index_close_and_names(self):
        df = store.parse_index_close(INDEX_2013).set_index("index_name")
        self.assertAlmostEqual(df.loc["NIFTY50", "close"], 6056.6)
        self.assertIn("NIFTY500", df.index)
        self.assertAlmostEqual(df.loc["INDIAVIX", "close"], 14.5)
        self.assertTrue(np.isnan(df.loc["INDIAVIX", "open"]))
        for raw, want in [("CNX Nifty", "NIFTY50"), ("Nifty 50", "NIFTY50"), ("CNX 500", "NIFTY500"),
                          ("Nifty 200", "NIFTY200"), ("India VIX", "INDIAVIX"), ("CNX Nifty Junior", "NIFTYNEXT50"),
                          ("CNX Bank", "NIFTYBANK")]:
            self.assertEqual(store.normalise_index_name(raw), want, raw)

    def test_corporate_actions_both_date_formats(self):
        legacy = reference.parse_corporate_actions(BC_LEGACY, pd.Timestamp("2019-12-05"))
        iso = reference.parse_corporate_actions(BC_ISO, pd.Timestamp("2025-10-14"))
        self.assertEqual(legacy.set_index("symbol").loc["HCLTECH", "ex_date"], pd.Timestamp("2019-12-05"))
        self.assertEqual(iso.iloc[0]["ex_date"], pd.Timestamp("2025-10-14"))
        events = reference.corporate_action_events(pd.concat([legacy, iso]))
        ev = events.set_index("symbol")
        self.assertNotIn("BOROSIL", ev.index)  # dividends ignored
        self.assertAlmostEqual(ev.loc["GOLDBEES", "factor"], 0.01)
        self.assertEqual(ev.loc["TATAMOTORS", "price_based"], 1.0)

    def test_revised_purpose_keeps_latest_file(self):
        # GRUH 2012: split announced RS10->RS5, then revised to RS10->RS2 for the same ex-date
        raw = pd.DataFrame({
            "symbol": ["GRUH", "GRUH", "GRUH"], "series": ["EQ", "EQ", "EQ"],
            "ex_date": pd.to_datetime(["2012-07-24"] * 2 + ["2012-06-06"]),
            "purpose": ["FV SPLIT FRM RS10 TO RS5", "FV SPLIT FRM RS10 TO RS2", "AGM/DIV RS 11.50PER SHARE"],
            "file_date": pd.to_datetime(["2012-06-26", "2012-06-27", "2012-05-08"]),
        })
        ev = reference.corporate_action_events(raw)
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev.iloc[0]["factor"], 0.2)

    def test_classify_purpose(self):
        c = reference.classify_purpose
        self.assertAlmostEqual(c("BONUS 1:2")["factor"], 2 / 3)
        self.assertAlmostEqual(c("BONUS - 3:1")["factor"], 0.25)
        self.assertAlmostEqual(c("FV SPLIT RS.10 TO RS.2")["factor"], 0.2)
        self.assertAlmostEqual(c("FV SPLIT- RS 10 TO RE 1")["factor"], 0.1)
        self.assertAlmostEqual(c("BON 1:2/SPLIT RS.10TORS.2")["factor"], 2 / 15)
        self.assertAlmostEqual(c("CONSOLIDATION RE1 TO RS10")["factor"], 10.0)
        r = c("RHTS 5:12@PRM RS.35 PR SH")
        self.assertEqual((r["rights_new"], r["rights_held"], r["rights_premium"]), (5.0, 12.0, 35.0))
        self.assertEqual(c("AGM/DIV- RS 0.65 PER SH"), {})
        self.assertEqual(c("CAPITAL REDUCTION"), {"price_based": 1.0})

    def test_classify_split_spellings(self):
        c = reference.classify_purpose
        cases = {
            "FVSPLT FRM RS 10 TO RS 5": 0.5, "FV SPL FRM RS 10 TO RE 1": 0.1, "FV SPLIT RS.2/- TO RE.1/-": 0.5,
            "BONUS1:1/FV SPL-RS10TORS5": 0.25, "SUB DIV FRM RS 10 TO RS 2": 0.2, "ADD ISSUANCE 9:1": 0.1,
            "BONUS 2:1/FVS RS.5TORE.1": 1 / 15, "FV SPLT2503.61TO250.361": 0.1, "FVSPLTFRM10TO1": 0.1,
            "DIV 1.35+FV SPL RS10TORS2": 0.2, "BON-1:25/SPLIT RS10TORS.5": 25 / 52, "CNSLDATN RE 1 TO RS 10": 10.0,
            "FV SPLT FRM RS 10 TO 2": 0.2, "BONUS ISSUE 1 : 1": 0.5, "AGM/DIV-2.50/BONUS 1:10": 10 / 11,
            "AGM/DIVRS10/STK SPT10 TO1": 0.1,
        }
        for purpose, want in cases.items():
            self.assertAlmostEqual(c(purpose).get("factor", np.nan), want, msg=purpose)
        # truncated purposes: a split/bonus without its ratio -> measured from prices
        for purpose in ("FV SPLIT FROM RS10 TO RS", "FV SPLT FRM RS 177.27 TO", "AGM/DIV-RS 29.50/BONUS"):
            self.assertEqual(c(purpose), {"hint": reference.HINT_SPLIT}, purpose)
        self.assertEqual(c("CAP REDN/CONSOLIDATION")["hint"], reference.HINT_CONSOLIDATION)
        # bonus debentures / preference shares are value distributions, not share-count changes
        for purpose in ("SCH AGMT-BONUS NCRPS 4:1", "AGM/DIV-RS3/BON DEB 1:1", "BON 1 DVR : 4 EQ SHARES"):
            self.assertEqual(c(purpose), {"price_based": 1.0}, purpose)
        self.assertEqual(c("SPL INT DIV-RS.13.5 PR SH"), {})
        self.assertEqual(c("RHTS 7:5 PRM@55/DIV RE.1")["rights_premium"], 55.0)

    def test_dividend_amount_spellings(self):
        p = reference.parse_dividend_amount
        cases = {
            "DIV - RS 3 PER SH": 3, "INTDIV - RS 4.80 PER SH": 4.8, "AGM/DIV - RS 2.50 PER SHARE": 2.5,
            "INT DIV RS 2.5": 2.5, "SPLDIV - RS 10 PER SH": 10, "DIVIDEND RE 0.50": 0.5, "AGM/DIV RE 0.75/-": 0.75,
            "AGM/DIV-RE.0.20 PER SHARE": 0.2, "DIVIDEND-RS.10/- PR SHARE": 10, "SPL INT DIV-RS.13.5 PR SH": 13.5,
            "AGM/DIV-FINRS 22+SPLRS 10": 32, "DIV/SPDIV - RS 3 & RS 3": 6, "AGM/SPDV/DIV- RS 8 & 20": 28,
            "DIV- 6.75 SPLDV- 2.75": 9.5, "DIV/SPLDIV-RS 35/5 PR SH": 40, "INTDIV-09/SPLDIV-13 PRSH": 22,
            "BONUS 1:1/DIV-RS 30 PR SH": 30, "AGM/DIV-RS 29.50/BONUS": 29.5, "DIV 1.35+FV SPL RS10TORS2": 1.35,
            "DIV-RS6/SPLIT RS 10TORE 1": 6, "DIV RE 1 + RIGHTS 5:6": 1, "RHTS 7:5 PRM@55/DIV RE.1": 1,
            "2ND INT DIV-RS.3/- PR SHR": 3, "2D INT DIV RS 3 PER SHARE": 3, "DIV - RS 2,50 PER SH": 2.5,
            "INTDVSPDV- RS 5 & RS 3": 8, "FIN RS 18+SPL RS 25": 43, "AGM/DI-RS 5 PER SHARE": 5,
            "DIV:INTRM 2.2+SPL 0.30 PS": 2.5, "INTDIV-RS6SPLINTDIV-RS10": 16, "SPL DIV RS 1,000 PER SH": 1000,
        }
        for purpose, want in cases.items():
            self.assertAlmostEqual(p(purpose), want, msg=purpose)
        for purpose in ("INTERIM DIVIDEND", "ANNUAL GENERAL MEETING", "BONUS 1:1", "FV SPLIT RS.10 TO RS.2",
                        "INTEREST PAYMENT", "SUB DIV FRM RS 10 TO RS 2", "DIV 25%", "BUYBACK", "RIGHTS 1:2 @ PREM RS 5"):
            self.assertIsNone(p(purpose), purpose)

    def test_dividend_events_dedupe(self):
        rows = []

        def add(sym, ex, purpose, files, series=("EQ", "BE")):
            for f in files:
                for ser in series:
                    rows.append((sym, ser, pd.Timestamp(ex), purpose, pd.Timestamp(f)))

        add("AAA", "2020-03-10", "INTERIM DIVIDEND", ["2020-02-20"])  # no amount
        add("AAA", "2020-03-10", "INT DIV-RS 5 PER SHARE", ["2020-03-02", "2020-03-05", "2020-03-09"])
        add("BBB", "2020-06-10", "DIV - RS 2 PER SH", ["2020-05-20", "2020-05-25"])  # ex-date revised ...
        add("BBB", "2020-06-12", "DIV - RS 2 PER SH", ["2020-05-27", "2020-06-11"])  # ... to the 12th
        add("CCC", "2020-07-01", "AGM/DIV-RS 29.50 PER SH", ["2020-06-01"])  # replaced by the combined listing
        add("CCC", "2020-07-01", "AGM/DIV-RS 29.50/BONUS", ["2020-06-20", "2020-06-30"])
        add("DDD", "2020-08-03", "INT DIV RS 4", ["2020-07-28", "2020-07-31"])  # two dividends, separate rows
        add("DDD", "2020-08-03", "SPL DIV RS 3", ["2020-07-28", "2020-07-31"])
        add("EEE", "2020-09-01", "DIV RS 2", ["2020-08-20"], series=("N1",))  # not an equity series
        raw = pd.DataFrame(rows, columns=reference.CA_COLUMNS)
        out = reference.dividend_events(raw).set_index("symbol")
        self.assertEqual(sorted(out.index), ["AAA", "BBB", "CCC", "DDD"])
        self.assertAlmostEqual(out.loc["AAA", "dividend"], 5.0)
        self.assertEqual(out.loc["BBB", "ex_date"], pd.Timestamp("2020-06-12"))
        self.assertAlmostEqual(out.loc["BBB", "dividend"], 2.0)
        self.assertAlmostEqual(out.loc["CCC", "dividend"], 29.5)
        self.assertAlmostEqual(out.loc["DDD", "dividend"], 7.0)

    def test_index_dates_checked_against_file_date(self):
        text = ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value\n"
                "Nifty 50,04-11-2023,1,1,1,17722.3\nNifty 500,05-12-2023,1,1,1,9000\n")
        df = store.parse_index_close(text, pd.Timestamp("2023-04-11"))
        self.assertEqual(list(df["index_name"]), ["NIFTY50"])  # swapped month/day fixed, other date dropped
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2023-04-11"))
        slash = "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value\n" \
                "CNX Nifty,09/06/2014,7621.65,7673.7,7580.25,7654.6\n"
        df = store.parse_index_close(slash, pd.Timestamp("2014-06-09"))
        self.assertEqual(df.iloc[0]["date"], pd.Timestamp("2014-06-09"))
        self.assertAlmostEqual(df.iloc[0]["close"], 7654.6)

    def test_symbol_changes_and_etf_list(self):
        text = (" NIPPON INDIA MF - Plan, E - GO,RDAXEDG,NDAXEDG,30-OCT-2019\n"
                "ETERNAL LIMITED,ZOMATO,ETERNAL,09-APR-2025\n")
        ch = reference.parse_symbol_changes(text)
        self.assertEqual(list(ch["old"]), ["RDAXEDG", "ZOMATO"])
        self.assertEqual(ch.iloc[1]["date"], pd.Timestamp("2025-04-09"))
        etfs = reference.parse_etf_list("Symbol,Underlying Asset\nGOLDBEES,Gold\nSILVERBEES ,Silver\n")
        self.assertEqual(etfs, frozenset({"GOLDBEES", "SILVERBEES"}))


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(404, b"<html>not found</html>"))

    def close(self):
        pass


class ArchiveTests(unittest.TestCase):
    def test_format_selection_around_boundary(self):
        root = Path("/tmp/x")
        c = archive.candidate_files(root, "equity", date(2024, 7, 5))
        self.assertEqual([x.fmt for x in c], ["legacy", "udiff"])
        c = archive.candidate_files(root, "equity", date(2024, 7, 8))
        self.assertEqual([x.fmt for x in c], ["udiff", "legacy"])
        self.assertEqual(c[0].path, root / "equity" / "2024" / "udiff20240708.csv.zip")
        c = archive.candidate_files(root, "equity", date(2013, 1, 15))
        self.assertEqual([x.fmt for x in c], ["legacy"])
        self.assertTrue(c[0].url.endswith("/2013/JAN/cm15JAN2013bhav.csv.zip"))
        self.assertEqual(c[0].path, root / "equity" / "2013" / "cm20130115.csv.zip")
        self.assertTrue(archive.candidate_files(root, "corpact", date(2019, 12, 5))[0].url.endswith("PR051219.zip"))

    def test_session_dates_skip_weekends(self):
        d = archive.session_dates(date(2020, 1, 31), date(2020, 2, 3))
        self.assertEqual(d, [date(2020, 1, 31), date(2020, 2, 1), date(2020, 2, 3)])  # budget Saturday

    def test_sync_resumable_holidays_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            d_ok, d_hol = date(2024, 7, 5), date(2024, 7, 4)
            legacy_ok = archive.candidate_files(Path(tmp), "equity", d_ok)[0]
            udiff_fb = archive.candidate_files(Path(tmp), "equity", date(2024, 7, 3))[1]
            routes = {legacy_ok.url: FakeResponse(200, _zip_bytes("a.csv", LEGACY_SAMPLE)),
                      udiff_fb.url: FakeResponse(200, _zip_bytes("b.csv", UDIFF_SAMPLE))}
            fake = FakeSession(routes)
            arc = archive.BhavcopyArchive(tmp, requests_per_second=0, session_factory=lambda: fake)
            counts = arc.sync(date(2024, 7, 3), d_ok, kinds=("equity",))
            self.assertEqual(counts["equity"], {"downloaded": 2, "holiday": 1})
            self.assertTrue(legacy_ok.path.exists())
            self.assertTrue(udiff_fb.path.exists())  # legacy 404 near boundary -> UDiFF
            manifest = json.loads((Path(tmp) / "manifest" / "missing.json").read_text())
            self.assertEqual(manifest["equity"], [d_hol.isoformat()])
            n_calls = len(fake.calls)
            arc2 = archive.BhavcopyArchive(tmp, requests_per_second=0, session_factory=lambda: fake)
            self.assertEqual(arc2.sync(date(2024, 7, 3), d_ok, kinds=("equity",))["equity"], {"skipped": 3})
            self.assertEqual(len(fake.calls), n_calls)  # nothing refetched

    def test_html_error_page_is_not_saved(self):
        p = Path("x.csv.zip")
        self.assertFalse(archive.looks_valid(b"<!DOCTYPE html><html>", p))
        self.assertTrue(archive.looks_valid(_zip_bytes("a.csv", "x"), p))
        cand = archive.RemoteFile("corpact", "pr", "u", Path("bc.csv"))
        pr = io.BytesIO()
        with zipfile.ZipFile(pr, "w") as zf:
            zf.writestr("Pd051219.csv", "x")
            zf.writestr("Bc051219.csv", BC_LEGACY)
        self.assertEqual(archive.transform_content(cand, pr.getvalue()).decode(), BC_LEGACY)


class SymbolLinkTests(unittest.TestCase):
    def test_rename_chain_is_date_aware(self):
        changes = pd.DataFrame({"old": ["TELCO", "TATAMOTORS"], "new": ["TATAMOTORS", "TMPV"],
                                "date": pd.to_datetime(["2003-12-26", "2025-10-24"])})
        syms = pd.Series(["TELCO", "TATAMOTORS", "TMPV", "TATAMOTORS"])
        dates = pd.Series(pd.to_datetime(["2003-01-01", "2019-01-01", "2025-11-01", "2025-11-01"]))
        out = reference.resolve_symbols(syms, dates, changes)
        # A later TATAMOTORS row (after the rename) is a different listing and keeps its name.
        self.assertEqual(list(out), ["TMPV", "TMPV", "TMPV", "TATAMOTORS"])

    def test_isin_continuity_rename_and_reused_symbol_guard(self):
        cal = pd.bdate_range("2020-01-01", periods=40)
        rows = pd.DataFrame({
            "symbol": ["OLDCO"] * 5 + ["NEWCO"] * 10 + ["OLDCO"] * 3 + ["OLDCO"] * 5 + ["X2"] * 3,
            "isin": ["INE1"] * 5 + ["INE1"] * 10 + ["INE9"] * 3 + ["INE7"] * 5 + ["INE7"] * 3,
            "date": list(cal[0:5]) + list(cal[5:15]) + list(cal[20:23]) + list(cal[30:35]) + list(cal[35:38]),
        })
        spans = reference.compute_spans(rows)
        table = reference.build_change_table(pd.DataFrame(columns=["old", "new", "date"]), spans, cal)
        pairs = set(zip(table["old"], table["new"]))
        self.assertIn(("OLDCO", "NEWCO"), pairs)
        self.assertIn(("OLDCO", "X2"), pairs)
        out = reference.resolve_symbols(rows["symbol"], rows["date"], table)
        self.assertTrue((out[:5] == "NEWCO").all())
        # the unrelated OLDCO listing (INE9, after a gap) is not merged into NEWCO or X2
        self.assertTrue((out[15:18] == "OLDCO").all())
        self.assertTrue((out[18:23] == "X2").all())

    def test_sector_map_includes_old_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "reference"
            ref.mkdir()
            (ref / "ind_nifty500list.csv").write_text(
                "Company Name,Industry,Symbol,Series,ISIN Code\nEternal Ltd.,Consumer Services,ETERNAL,EQ,INE758T01015\n")
            (ref / "symbolchange.csv").write_text("ETERNAL LIMITED,ZOMATO,ETERNAL,09-APR-2025\n")
            out = Path(tmp) / "map.json"
            sectors = reference.build_sector_map(tmp, out)
            self.assertEqual(sectors, {"ETERNAL": "Consumer Services", "ZOMATO": "Consumer Services"})
            self.assertEqual(json.loads(out.read_text()), sectors)


class AdjustmentTests(unittest.TestCase):
    def _frames(self, closes, prev):
        idx = pd.bdate_range("2020-01-01", periods=len(closes))
        return pd.DataFrame({"A": closes}, index=idx, dtype=float), pd.DataFrame({"A": prev}, index=idx, dtype=float)

    def test_prev_close_factor(self):
        # 1:2 split on day 3 where the exchange-adjusted prev close is published
        close, prev = self._frames([100, 102, 51.5, 52], [99, 100, 51, 51.5])
        mult, factors = panel.adjustment_multipliers(close, prev, infer_gaps=False)
        adj = close * mult
        self.assertAlmostEqual(factors["A"].iloc[2], 0.5)
        np.testing.assert_allclose(adj["A"].to_numpy(), [50, 51, 51.5, 52])

    def test_bad_prev_close_rejected(self):
        # prev_close claims a halving but prices did not move
        close, prev = self._frames([100, 101, 102, 103], [100, 100, 50.5, 102])
        mult, _ = panel.adjustment_multipliers(close, prev, infer_gaps=False)
        np.testing.assert_allclose(mult["A"].to_numpy(), 1.0)

    def test_missing_session_prev_close_mismatch_ignored(self):
        # every symbol's prev_close differs on day 2 (an un-archived session in between)
        idx = pd.bdate_range("2020-01-01", periods=3)
        rng = np.random.default_rng(0)
        base = rng.uniform(50, 500, size=40)
        close = pd.DataFrame([base, base * 1.01, base * 1.02], index=idx)
        prev = pd.DataFrame([base, base * 0.99, base * 1.01], index=idx)
        mult, factors = panel.adjustment_multipliers(close, prev, infer_gaps=False)
        np.testing.assert_allclose(mult.to_numpy(), 1.0)
        self.assertEqual(factors.shape[1], 0)

    def _events(self, idx, factor=np.nan, hint=np.nan, day=2):
        return pd.DataFrame({"symbol": ["A"], "canonical": ["A"], "ex_date": [idx[day]], "purpose": ["x"],
                             "factor": [factor], "rights_new": [np.nan], "rights_held": [np.nan],
                             "rights_premium": [np.nan], "price_based": [np.nan], "hint": [hint]})

    def test_isin_change_snaps_inferred_split(self):
        # SHRIRAMFIN 2025-01-10: FV Rs10 -> Rs2 with no NSE record; ISIN changed, close ratio 0.1893
        closes = [2958.4, 2898.75, 2809.85, 532.0, 521.1, 544.1, 540.0, 538.0]
        opens = [2977.05, 2970.0, 2903.75, 566.0, 530.05, 526.95, 541.0, 539.0]
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        open_ = pd.DataFrame({"A": opens}, index=close.index, dtype=float)
        isin = np.zeros(close.shape, dtype=bool)
        isin[3, 0] = True
        mult, factors = panel.adjustment_multipliers(close, prev, open_, isin_change=isin)
        self.assertEqual(factors["A"].iloc[3], 0.2)
        self.assertEqual(factors.attrs["source_counts"]["inferred_snapped"], 1)
        # without the open, the close ratio (8.7% from 0.2) snaps only on an ISIN-change date
        closes2 = [1000.0, 1000.0, 1000.0, 184.0, 185.0, 184.0, 186.0, 185.0]
        close2, prev2 = self._frames(closes2, [np.nan] + closes2[:-1])
        _, f_isin = panel.adjustment_multipliers(close2, prev2, close2, isin_change=isin)
        self.assertEqual(f_isin["A"].iloc[3], 0.2)
        _, f_plain = panel.adjustment_multipliers(close2, prev2, close2)
        self.assertAlmostEqual(f_plain["A"].iloc[3], 0.184)  # unexplained: raw ratio kept
        self.assertEqual(f_plain.attrs["source_counts"]["inferred_unexplained"], 1)

    def test_unexplained_gap_not_snapped_and_penny_ticks_ignored(self):
        closes = [100.0, 100.0, 100.0, 160.0, 161.0, 160.0, 159.0, 160.0]  # +60% gap: no common ratio
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        _, factors = panel.adjustment_multipliers(close, prev, close)
        self.assertAlmostEqual(factors["A"].iloc[3], 1.6)
        self.assertEqual(factors.attrs["source_counts"]["inferred_unexplained"], 1)
        ticks = [0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]  # Rs 0.05 tick noise
        close, prev = self._frames(ticks, [np.nan] + ticks[:-1])
        mult, _ = panel.adjustment_multipliers(close, prev, close)
        np.testing.assert_allclose(mult["A"].to_numpy(), 1.0)

    def test_ratio_event_moved_to_adjacent_session(self):
        # NSE lists the 10:1 split a session before the price actually changes
        closes = [1000.0, 1002.0, 1004.0, 100.5, 101.0, 102.0]
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        _, factors = panel.adjustment_multipliers(close, prev, close, self._events(close.index, 0.1, day=2),
                                                  infer_gaps=False)
        self.assertTrue(np.isnan(factors["A"].iloc[2]))
        self.assertAlmostEqual(factors["A"].iloc[3], 0.1)

    def test_split_without_ratio_measured_from_prices(self):
        closes = [500.0, 505.0, 102.0, 103.0, 104.0]  # "FV SPLIT FROM RS10 TO RS" (truncated)
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        _, factors = panel.adjustment_multipliers(close, prev, close,
                                                  self._events(close.index, hint=reference.HINT_SPLIT),
                                                  infer_gaps=False)
        self.assertEqual(factors["A"].iloc[2], 0.2)

    def test_dividend_total_return_factors(self):
        close, _ = self._frames([100, 102, 96, 97], [np.nan] * 4)
        amounts = np.full(close.shape, np.nan)
        amounts[2, 0] = 5.1  # ex-date day 2: 5% of the prior close
        mult, yields = panel.dividend_multipliers(close, amounts)
        self.assertAlmostEqual(yields["A"].iloc[2], 0.05)
        adj = (close * mult)["A"].to_numpy()
        np.testing.assert_allclose(adj, [95.0, 96.9, 96, 97])
        # ex-date return = P_t / (P_{t-1} - D): the dividend reinvested at the prior close
        self.assertAlmostEqual(adj[2] / adj[1] - 1, 96 / (102 - 5.1) - 1)
        amounts[2, 0] = 60.0  # >= 50% of price: ignored
        mult, _ = panel.dividend_multipliers(close, amounts)
        np.testing.assert_allclose(mult.to_numpy(), 1.0)

    def test_split_and_dividend_ordering(self):
        # Rs 2 dividend on day 1; 1:1 bonus with a Rs 1 dividend on the same record date (day 2): the
        # dividend is per pre-bonus share; Rs 0.5 dividend on day 4, after the bonus
        closes = [200.0, 196.0, 97.0, 98.0, 97.5, 98.0]
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        split_mult, factors = panel.adjustment_multipliers(close, prev, close,
                                                           self._events(close.index, 0.5, day=2), infer_gaps=False)
        amounts = np.full(close.shape, np.nan)
        amounts[1, 0], amounts[2, 0], amounts[4, 0] = 2.0, 1.0, 0.5
        f = factors.reindex(columns=close.columns).to_numpy()
        div_mult, yields = panel.dividend_multipliers(close, amounts, f)
        self.assertAlmostEqual(yields["A"].iloc[1], 2.0 / 200)
        self.assertAlmostEqual(yields["A"].iloc[2], 1.0 / 196)  # pre-bonus units
        self.assertAlmostEqual(yields["A"].iloc[4], 0.5 / 98)
        adj = (close * split_mult * div_mult)["A"].to_numpy()
        tr = adj[1:] / adj[:-1] - 1
        want = [196 / (200 - 2) - 1, 97 / ((196 - 1) * 0.5) - 1, 98 / 97 - 1, 97.5 / (98 - 0.5) - 1, 98 / 97.5 - 1]
        np.testing.assert_allclose(tr, want, rtol=1e-12)
        np.testing.assert_allclose(split_mult["A"].to_numpy(), [0.5, 0.5, 1, 1, 1, 1])
        _, post = panel.dividend_multipliers(close, amounts, f, same_day_units="post")
        self.assertAlmostEqual(post["A"].iloc[2], 1.0 / 98)

    def test_large_dividend_kept_only_when_prices_confirm(self):
        # MAJESCO 2020-12-23: Rs 974 interim dividend on a ~Rs 986 share, opened ~Rs 12
        closes = [985.0, 986.0, 12.2, 12.4, 12.3, 12.5, 12.4]
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        amounts = np.full(close.shape, np.nan)
        amounts[2, 0] = 974.0
        mult, factors = panel.adjustment_multipliers(close, prev, close, dividend_amounts=amounts)
        self.assertEqual(factors.shape[1], 0)  # the drop is the dividend, not a split
        div_mult, yields = panel.dividend_multipliers(close, amounts, open_=close)
        self.assertAlmostEqual(yields["A"].iloc[2], 974 / 986)
        adj = (close * div_mult)["A"]
        self.assertLess(adj.pct_change().abs().max(), 0.05)

    def test_ex_date_on_non_trading_session_not_double_counted(self):
        # ETF split listed for a session it did not trade; the next session opens at the new scale
        closes = [270.0, 272.8, np.nan, 27.4, 27.7, 27.2, 27.4, 27.9]
        close, prev = self._frames(closes, [np.nan] + closes[:-1])
        _, factors = panel.adjustment_multipliers(close, prev, close, self._events(close.index, 0.1, day=2))
        self.assertEqual(factors.attrs["source_counts"]["ca_ratio"], 1)
        self.assertEqual(int(factors.notna().to_numpy().sum()), 1)

    def test_total_return_index_from_dividend_points(self):
        idx = pd.bdate_range("2020-03-25", periods=6)
        price = pd.Series([100.0, 101, 102, 102, 103, 104], index=idx)
        points = pd.Series([10.0, 11.0, 0.0, 0.5, 0.4, 1.5], index=idx)  # FY reset on day 2, noise dip on day 4
        tri = panel.total_return_index(price, points)
        want = [100.0]
        for p0, p1, d in zip(price[:-1], price[1:], [1.0, 0.0, 0.5, 0.0, 1.0]):
            want.append(want[-1] * (p1 + d) / p0)
        np.testing.assert_allclose(tri.to_numpy(), want)

    def test_corporate_action_event_used_when_prev_close_unadjusted(self):
        close, prev = self._frames([3350, 3359.6, 33.55, 33.65], [3340, 3350, 3359.6, 33.55])
        events = pd.DataFrame({"symbol": ["A"], "canonical": ["A"], "ex_date": [close.index[2]],
                               "purpose": ["FVSPLT FRM RS100 TO RS 1"], "factor": [0.01],
                               "rights_new": [np.nan], "rights_held": [np.nan], "rights_premium": [np.nan],
                               "price_based": [np.nan]})
        mult, _ = panel.adjustment_multipliers(close, prev, close, events, infer_gaps=False)
        adj = (close * mult)["A"]
        self.assertLess(adj.pct_change().abs().max(), 0.01)
        self.assertAlmostEqual(adj.iloc[0], 33.5)


def _legacy_file(day: pd.Timestamp, rows) -> bytes:
    lines = [LEGACY_HEADER]
    stamp = day.strftime("%d-%b-%Y").upper()
    for sym, series, close, prev, qty, isin in rows:
        lines.append(f"{sym},{series},{close},{close},{close},{close},{close},{prev},{qty},{close * qty},{stamp},10,{isin},")
    return _zip_bytes("cm.csv", "\n".join(lines) + "\n")


class PanelIntegrationTests(unittest.TestCase):
    """Raw archive files -> build_store -> load_market_data, fully offline."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.arch, cls.store = root / "archive", root / "store"
        days = pd.bdate_range("2020-01-06", periods=6)
        cls.days = days
        for i, day in enumerate(days):
            rows = []
            # BONUS: 1:1 bonus ex day 3, prev_close NOT adjusted (as in real bhavcopies)
            c = [200, 202, 204, 103, 104, 105][i]
            rows.append(("BONUS", "EQ", c, [199, 200, 202, 204, 103, 104][i], 1_000_000, "INE000B01011"))
            # OLDNAME renamed to NEWNAME effective day 4 (symbolchange.csv)
            sym = "OLDNAME" if i < 4 else "NEWNAME"
            rows.append((sym, "EQ", 50 + i, 50 + i - 1, 500_000, "INE000R01011"))
            # BOTH trades in BE and EQ on day 2: EQ preferred
            if i == 2:
                rows.append(("BOTH", "BE", 999, 30, 100, "INE000X01011"))
            rows.append(("BOTH", "EQ", 30 + i, 30 + i - 1, 300_000, "INE000X01011"))
            rows.append(("GOLDBEES", "EQ", 40 + i, 40 + i - 1, 10, "INF204KB17I5"))  # illiquid but included
            rows.append(("TINY", "EQ", 5, 5, 10, "INE000T01011"))  # illiquid -> dropped
            # DIVCO: Rs 4 dividend ex day 3 (listed in several daily Bc files, EQ and BE)
            rows.append(("DIVCO", "EQ", [100, 101, 102, 98, 99, 100][i], [100, 100, 101, 102, 98, 99][i],
                         1_000_000, "INE000D01011"))
            path = arch_path = cls.arch / "equity" / "2020" / f"cm{day:%Y%m%d}.csv.zip"
            arch_path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_legacy_file(day, rows))
            mto = cls.arch / "delivery" / "2020" / f"mto{day:%Y%m%d}.dat"
            mto.parent.mkdir(parents=True, exist_ok=True)
            mto.write_text(f"Security Wise Delivery Position\n10,MTO,{day:%d%m%Y},1,1\n20,1,BONUS,EQ,1000000,400000,40.00\n")
            ind = cls.arch / "indices" / "2020" / f"ind{day:%Y%m%d}.csv"
            ind.parent.mkdir(parents=True, exist_ok=True)
            ind.write_text("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value\n"
                           f"Nifty 50,{day:%d-%m-%Y},1,1,1,{12000 + i}\nNifty 500,{day:%d-%m-%Y},1,1,1,{9000 + i}\n"
                           + (f"India VIX,{day:%d-%m-%Y},1,1,1,{15 + i}\n" if i > 0 else "")
                           + f"Nifty50 Dividend Points,{day:%d-%m-%Y},1,1,1,{[5, 5, 6, 6, 6.5, 6.5][i]}\n")
            bc = cls.arch / "corpact" / "2020" / f"bc{day:%Y%m%d}.csv"
            bc.parent.mkdir(parents=True, exist_ok=True)
            bc.write_text("SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
                          f"EQ,BONUS,Bonus Ltd,,,,{days[3]:%d/%m/%Y},,,BONUS 1:1\n"
                          + (f"EQ,DIVCO,Div Co,,,,{days[3]:%d/%m/%Y},,,INTDIV - RS 4 PER SH\n"
                             f"BE,DIVCO,Div Co,,,,{days[3]:%d/%m/%Y},,,INTDIV - RS 4 PER SH\n" if i < 3 else ""))
        ref = cls.arch / "reference"
        ref.mkdir(parents=True)
        (ref / "symbolchange.csv").write_text(f"Rename Ltd,OLDNAME,NEWNAME,{days[4]:%d-%b-%Y}\n".upper())
        (ref / "eq_etfseclist.csv").write_text("Symbol,Underlying Asset\nGOLDBEES,Gold\n")
        cls.sector_map = root / "sectors.json"
        cls.sector_map.write_text(json.dumps({"BONUS": "Capital Goods", "OLDNAME": "Chemicals", "NEWNAME": "Chemicals"}))
        cls.summary = store.build_store(cls.arch, cls.store, workers=1)
        cls.data = panel.load_market_data(cls.store, "2020-01-01", "2020-01-31", min_median_value_inr=1e6,
                                          vix_fallback=False, sector_map_path=cls.sector_map)
        cls.price_only = panel.load_market_data(cls.store, "2020-01-01", "2020-01-31", min_median_value_inr=1e6,
                                                vix_fallback=False, sector_map_path=cls.sector_map,
                                                adjust_dividends=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_store_idempotent(self):
        again = store.build_store(self.arch, self.store, workers=1)
        self.assertEqual(again["rebuilt_years"], [])
        self.assertEqual(self.summary["rebuilt_years"], [2020])

    def test_alignment_and_validate(self):
        d = self.data
        d.validate()
        self.assertTrue(d.dates.equals(pd.DatetimeIndex(self.days)))
        self.assertEqual(d.close.dtypes.unique().tolist(), [np.dtype("float32")])
        self.assertEqual(d.data_hash, d.compute_hash())
        self.assertEqual(list(d.index_close.columns), ["NIFTY50", "NIFTY500", "INDIAVIX", "NIFTY50_TRI"])
        tri = d.index_close["NIFTY50_TRI"].to_numpy()
        self.assertAlmostEqual(tri[0], 12000.0)
        self.assertAlmostEqual(tri[2], 12000.0 * (12001 / 12000) * ((12002 + 1) / 12001), places=2)
        self.assertTrue(np.isnan(d.index_close["INDIAVIX"].iloc[0]))
        self.assertAlmostEqual(float(d.delivery_pct["BONUS"].iloc[0]), 40.0)

    def test_liquidity_filter_keeps_include_symbols(self):
        self.assertIn("GOLDBEES", self.data.symbols)
        self.assertNotIn("TINY", self.data.symbols)
        self.assertEqual(self.data.etfs, frozenset({"GOLDBEES"}))

    def test_rename_linked_into_canonical_symbol(self):
        self.assertNotIn("OLDNAME", self.data.symbols)
        np.testing.assert_allclose(self.data.close["NEWNAME"].to_numpy(), [50, 51, 52, 53, 54, 55])
        self.assertEqual(self.data.sectors.get("NEWNAME"), "Chemicals")

    def test_bonus_back_adjusted(self):
        adj = self.data.close["BONUS"].to_numpy()
        np.testing.assert_allclose(adj, [100, 101, 102, 103, 104, 105], rtol=1e-6)
        vol = self.data.volume["BONUS"].to_numpy()
        np.testing.assert_allclose(vol[:3], 2_000_000, rtol=1e-6)
        self.assertAlmostEqual(float(self.data.value["BONUS"].iloc[0]), 200 * 1_000_000, delta=64)

    def test_dividend_adjustment_and_price_only_mode(self):
        self.assertTrue(pd.read_parquet(self.store / "dividends.parquet")["symbol"].eq("DIVCO").any())
        tr = self.data.close["DIVCO"].to_numpy(dtype="float64")
        y = 4.0 / 102.0
        np.testing.assert_allclose(tr, [100 * (1 - y), 101 * (1 - y), 102 * (1 - y), 98, 99, 100], rtol=1e-6)
        np.testing.assert_allclose(self.price_only.close["DIVCO"].to_numpy(), [100, 101, 102, 98, 99, 100], rtol=1e-6)
        # splits and volumes are identical in both modes
        pd.testing.assert_frame_equal(self.data.volume, self.price_only.volume)
        np.testing.assert_allclose(self.price_only.close["BONUS"].to_numpy(), [100, 101, 102, 103, 104, 105], rtol=1e-6)
        self.assertNotEqual(self.data.data_hash, self.price_only.data_hash)

    def test_eq_preferred_over_be(self):
        self.assertAlmostEqual(float(self.data.close["BOTH"].iloc[2]), 32.0)


class LiquidityTests(unittest.TestCase):
    def test_rolling_median_threshold(self):
        idx = pd.bdate_range("2020-01-01", periods=12)
        value = pd.DataFrame({"LIQ": 5e6, "NEW": 5e6, "SPARSE": np.nan, "ILLIQ": 1e5, "GOLDBEES": 1.0}, index=idx)
        value.iloc[:8, 1] = np.nan  # listed late: qualifies after min_sessions of history
        value.iloc[::4, 2] = 9e6  # trades one day in four: zero-value days dominate the median
        keep = panel.liquid_symbols(value, 2.5e6, include=("GOLDBEES",), window=6, min_sessions=4)
        self.assertEqual(keep, ["LIQ", "NEW", "GOLDBEES"])


if __name__ == "__main__":
    unittest.main()
