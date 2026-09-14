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
                           + (f"India VIX,{day:%d-%m-%Y},1,1,1,{15 + i}\n" if i > 0 else ""))
            bc = cls.arch / "corpact" / "2020" / f"bc{day:%Y%m%d}.csv"
            bc.parent.mkdir(parents=True, exist_ok=True)
            bc.write_text("SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
                          f"EQ,BONUS,Bonus Ltd,,,,{days[3]:%d/%m/%Y},,,BONUS 1:1\n")
        ref = cls.arch / "reference"
        ref.mkdir(parents=True)
        (ref / "symbolchange.csv").write_text(f"Rename Ltd,OLDNAME,NEWNAME,{days[4]:%d-%b-%Y}\n".upper())
        (ref / "eq_etfseclist.csv").write_text("Symbol,Underlying Asset\nGOLDBEES,Gold\n")
        cls.sector_map = root / "sectors.json"
        cls.sector_map.write_text(json.dumps({"BONUS": "Capital Goods", "OLDNAME": "Chemicals", "NEWNAME": "Chemicals"}))
        cls.summary = store.build_store(cls.arch, cls.store, workers=1)
        cls.data = panel.load_market_data(cls.store, "2020-01-01", "2020-01-31", min_median_value_inr=1e6,
                                          vix_fallback=False, sector_map_path=cls.sector_map)

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
        self.assertEqual(list(d.index_close.columns), ["NIFTY50", "NIFTY500", "INDIAVIX"])
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
