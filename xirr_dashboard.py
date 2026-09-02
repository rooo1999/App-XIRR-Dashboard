"""
Portfolio XIRR Dashboard with Nifty 50 Benchmarking
=====================================================
Upload a transaction report (SN, Transaction ID, Goal Name/Basket Name, Fund,
Date & Time, Type, Amount, <signed cash flow>) and this app will:

1. Clean & parse the transactions
2. Calculate the portfolio's money-weighted return (XIRR)
3. Simulate the same cash flows invested in a Nifty 50 benchmark to get a
   like-for-like comparison XIRR. Tries, in order: the raw Nifty 50 index
   (^NSEI via Yahoo Finance) -> UTI Nifty 50 Index Fund - Direct Plan NAV
   (via the free mfapi.in API, as a real, investable proxy) -> a manually
   uploaded CSV as a last resort.
4. Let you download a full Excel workbook (summary + transactions +
   benchmark detail) that anyone can open, no login/tooling required

Run with:  streamlit run xirr_dashboard.py
Requires:  pip install streamlit pandas numpy scipy yfinance requests xlsxwriter openpyxl
"""

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import brentq

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

st.set_page_config(page_title="Portfolio XIRR vs Nifty 50", layout="wide")

BENCHMARK_TICKER = "^NSEI"  # Nifty 50 on Yahoo Finance


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------

def clean_num(val):
    """Turn '7,500.00' / '-7,500.00' / '6.23%' style strings into floats."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("%", "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def classify_type(type_val) -> str:
    """Buckets a transaction Type into invest / withdraw / switch / unknown.
    Handles label variants across different report exports."""
    if pd.isna(type_val):
        return "unknown"
    t = str(type_val).strip().lower()
    if "switch" in t or "stp" in t:
        return "switch"
    if "withdraw" in t or "redeem" in t:
        return "withdraw"
    if "sip" in t or "lump" in t or "purchase" in t:
        return "invest"
    return "unknown"


def is_switch_type(type_val) -> bool:
    """A 'Switch' (or STP) moves money between two funds inside the same
    portfolio — it is not a real external cash flow (no money enters/leaves
    the investor's pocket), so it must be excluded from XIRR and from
    Invested/Withdrawn totals."""
    return classify_type(type_val) == "switch"


def derive_cashflow(type_val, amount) -> float:
    """Used when the report has no signed cash-flow column (only an
    unsigned Amount): assigns the sign from the Type. Switch rows are left
    as NaN because their direction (in vs out of a given fund) can't be
    determined from Amount alone — they're excluded from totals anyway."""
    if pd.isna(amount):
        return np.nan
    cat = classify_type(type_val)
    if cat == "invest":
        return -abs(amount)
    if cat == "withdraw":
        return abs(amount)
    if cat == "switch":
        return np.nan
    return -abs(amount)  # unrecognized type -> conservatively treat as money invested


def parse_date(val):
    """Handles both 'dd-mm-yyyy H.MM' (transactions) and 'dd-mm-yyyy'
    (the summary 'current value as of' row)."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for fmt in ("%d-%m-%Y %H.%M", "%d-%m-%Y", "%d-%m-%Y %H:%M",
                "%d %b %Y %I:%M %p", "%d %B %Y %I:%M %p", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return parsed if not pd.isna(parsed) else pd.NaT


@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes):
    """Parses the uploaded CSV into (transactions_df, current_value,
    as_of_date, platform_reported_xirr_pct). Handles two report variants:
    (a) one with a trailing signed cash-flow column and summary rows for
    current value / platform XIRR, and (b) one with only an unsigned
    Amount column and no summary rows, where the sign is derived from
    Type and the current value must be entered manually in the app."""
    # index_col=False guards against a trailing comma in some exports being
    # mistaken by pandas for an index column, which silently shifts every
    # other column one slot to the left.
    raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, index_col=False)
    raw.columns = [c.strip() for c in raw.columns]

    standard_cols = {"SN", "Transaction ID", "Goal Name/Basket Name", "Fund",
                      "Date & Time", "Type", "Amount"}
    extra_cols = [c for c in raw.columns if c not in standard_cols]
    has_signed_col = len(extra_cols) > 0

    if has_signed_col:
        raw = raw.rename(columns={extra_cols[-1]: "CashFlow"})
        raw["CashFlow_num"] = raw["CashFlow"].apply(clean_num)
    else:
        raw["CashFlow"] = np.nan
        raw["CashFlow_num"] = np.nan

    raw["Amount_num"] = raw["Amount"].apply(clean_num)
    raw["Date_parsed"] = raw["Date & Time"].apply(parse_date)
    raw["Type_Category"] = raw["Type"].apply(classify_type)
    raw["Is_Switch"] = raw["Type_Category"] == "switch"

    # fill in CashFlow_num from Type+Amount wherever a signed value wasn't
    # supplied (covers reports with no signed column at all, as well as any
    # individual blank cells in a report that does have one)
    missing = raw["CashFlow_num"].isna()
    raw.loc[missing, "CashFlow_num"] = raw.loc[missing].apply(
        lambda r: derive_cashflow(r["Type"], r["Amount_num"]), axis=1
    )

    # trailing summary rows (only present in some report exports) have no
    # SN / Transaction ID
    summary_mask = raw["SN"].isna() | raw["Transaction ID"].isna()
    summary = raw[summary_mask]
    txns = raw[~summary_mask].copy()

    txns = txns.dropna(subset=["Date_parsed"])
    txns = txns.sort_values("Date_parsed").reset_index(drop=True)

    current_value, as_of_date, reported_xirr = None, None, None
    for _, row in summary.iterrows():
        cf_raw = str(row["CashFlow"])
        if "%" in cf_raw:
            reported_xirr = clean_num(cf_raw)
        elif not pd.isna(row["CashFlow_num"]):
            current_value = row["CashFlow_num"]
            as_of_date = parse_date(row["Date & Time"])

    return txns, current_value, as_of_date, reported_xirr


# ----------------------------------------------------------------------------
# XIRR
# ----------------------------------------------------------------------------

def xirr(cashflows):
    """cashflows: list of (datetime, amount). Returns annualised rate."""
    dates = [d for d, _ in cashflows]
    d0 = min(dates)

    def npv(rate):
        return sum(a / (1 + rate) ** ((d - d0).days / 365.0) for d, a in cashflows)

    try:
        return brentq(npv, -0.999999, 10)
    except ValueError:
        # widen the search if brentq can't bracket a root
        for lo, hi in [(-0.9999999, 100), (-0.99999999, 1000)]:
            try:
                return brentq(npv, lo, hi)
            except ValueError:
                continue
        return np.nan


# ----------------------------------------------------------------------------
# Nifty 50 benchmark simulation
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_benchmark_series(start: datetime, end: datetime):
    """Daily Nifty 50 close, forward-filled across non-trading days."""
    data = yf.download(
        BENCHMARK_TICKER,
        start=start - timedelta(days=7),
        end=end + timedelta(days=7),
        progress=False,
        auto_adjust=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for {BENCHMARK_TICKER} between "
            f"{start:%d-%b-%Y} and {end:%d-%b-%Y}. This is often Yahoo "
            "rate-limiting or blocking requests from a cloud server's IP — "
            "try again in a bit, or upload a Nifty 50 CSV manually below."
        )
    if "Close" not in data.columns:
        raise ValueError(f"Unexpected response shape from yfinance (columns: {list(data.columns)}).")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns multi-col
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    full_range = pd.date_range(close.index.min(), close.index.max(), freq="D")
    series = close.reindex(full_range).ffill().bfill()
    return series


def parse_manual_benchmark_csv(file_bytes: bytes):
    """Fallback path: user-supplied CSV with Date + Close (or similar) columns,
    e.g. exported from niftyindices.com or NSE, for when both automatic
    sources are unreachable."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    price_col = next((c for c in df.columns if c in ("close", "closing price", "price", "nav")), None)
    if date_col is None or price_col is None:
        raise ValueError(
            f"Couldn't find Date/Close columns in that CSV (found: {list(df.columns)}). "
            "Expected a column with 'date' in the name and one named Close/Price/NAV."
        )
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df[price_col] = df[price_col].apply(clean_num)
    series = df.set_index(date_col)[price_col]
    full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_range).ffill().bfill()


# ----------------------------------------------------------------------------
# mfapi.in fallback — UTI Nifty 50 Index Fund (Direct Plan) NAV as a proxy.
# This is a real, investable fund that tracks the Nifty 50, so its NAV
# already nets out the small tracking error and expense ratio a real
# investor would have paid — arguably a *more* realistic benchmark than the
# raw index, and mfapi.in is a free India-specific API that isn't subject
# to the Yahoo Finance cloud-IP blocking that affects yfinance.
# ----------------------------------------------------------------------------

MFAPI_BASE = "https://api.mfapi.in"
UTI_NIFTY50_INCLUDE_HINTS = ["uti", "nifty 50", "direct", "growth"]
UTI_NIFTY50_EXCLUDE_HINTS = ["next 50", "etf"]


@st.cache_data(show_spinner=False)
def find_uti_nifty50_scheme_code():
    """Looks up the scheme code for 'UTI Nifty 50 Index Fund - Direct Plan -
    Growth' via mfapi.in's search endpoint, rather than hard-coding a code
    that could change or vary by source."""
    resp = requests.get(f"{MFAPI_BASE}/mf/search", params={"q": "UTI Nifty 50 Index Fund"}, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    for r in results:
        name = r.get("schemeName", "").lower()
        if all(h in name for h in UTI_NIFTY50_INCLUDE_HINTS) and not any(h in name for h in UTI_NIFTY50_EXCLUDE_HINTS):
            return r["schemeCode"], r["schemeName"]
    raise ValueError("Could not find 'UTI Nifty 50 Index Fund - Direct Plan - Growth' via mfapi.in search.")


@st.cache_data(show_spinner=False)
def fetch_mfapi_nav_series(scheme_code: int):
    """Daily NAV history for a mutual fund scheme from mfapi.in, forward-
    filled across non-NAV days (weekends/holidays)."""
    resp = requests.get(f"{MFAPI_BASE}/mf/{scheme_code}", timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "SUCCESS" or not payload.get("data"):
        raise ValueError(f"mfapi.in returned no NAV data for scheme {scheme_code}.")
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df = df.sort_values("date")
    series = df.set_index("date")["nav"]
    full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_range).ffill().bfill()


def price_on(series: pd.Series, date: datetime):
    d = pd.Timestamp(date.date())
    if d in series.index:
        return float(series.loc[d])
    # nearest available (handles dates outside fetched range)
    idx = series.index.get_indexer([d], method="nearest")[0]
    return float(series.iloc[idx])


def simulate_benchmark(txns: pd.DataFrame, current_value: float, as_of_date: datetime,
                        series: pd.Series):
    """Replays every real cash flow as if it happened in the Nifty 50 index
    instead, and returns (benchmark_xirr, benchmark_terminal_value, detail_df)."""
    units = 0.0
    rows = []
    for _, r in txns.iterrows():
        px = price_on(series, r["Date_parsed"])
        cf = r["CashFlow_num"]
        if cf < 0:  # investment -> buy units
            bought = (-cf) / px
            units += bought
        else:  # withdrawal -> sell units for the same rupee amount
            bought = -(cf / px)
            units += bought
        rows.append({
            "Date": r["Date_parsed"], "Fund": r["Fund"], "CashFlow": cf,
            "Nifty50_Price": px, "Units_Delta": bought, "Cumulative_Units": units,
        })

    final_px = price_on(series, as_of_date)
    benchmark_value = units * final_px

    detail_df = pd.DataFrame(rows)
    bench_cashflows = list(zip(txns["Date_parsed"], txns["CashFlow_num"]))
    bench_cashflows.append((as_of_date, benchmark_value))
    b_xirr = xirr(bench_cashflows)
    return b_xirr, benchmark_value, detail_df


# ----------------------------------------------------------------------------
# Excel export
# ----------------------------------------------------------------------------

def build_excel(summary: dict, txns: pd.DataFrame, breakdown_df: pd.DataFrame,
                 breakdown_label: str, bench_detail: pd.DataFrame | None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        bold = wb.add_format({"bold": True})
        pct_fmt = wb.add_format({"num_format": "0.00%"})
        rupee_fmt = wb.add_format({"num_format": "#,##0.00"})
        date_fmt = wb.add_format({"num_format": "dd-mmm-yyyy"})
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})

        # ---- Summary sheet ----
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        ws.set_column("A:A", 32)
        ws.set_column("B:B", 20)
        ws.write(0, 0, "Portfolio Performance Summary", bold)
        rows = [
            ("As-of Date", summary["as_of_date"], date_fmt),
            ("Total Invested (Rs.)", summary["total_invested"], rupee_fmt),
            ("Total Withdrawn (Rs.)", summary["total_withdrawn"], rupee_fmt),
            ("Current Portfolio Value (Rs.)", summary["current_value"], rupee_fmt),
            ("Absolute Gain (Rs.)", summary["absolute_gain"], rupee_fmt),
            ("Portfolio XIRR", summary["portfolio_xirr"], pct_fmt),
        ]
        if summary.get("platform_xirr") is not None:
            rows.append(("Platform-Reported XIRR (for reference)", summary["platform_xirr"] / 100, pct_fmt))
        if summary.get("benchmark_xirr") is not None:
            rows += [
                ("Nifty 50 Benchmark Value if same cash flows (Rs.)", summary["benchmark_value"], rupee_fmt),
                ("Nifty 50 Benchmark XIRR", summary["benchmark_xirr"], pct_fmt),
                ("Alpha vs Nifty 50 (XIRR pts)", summary["portfolio_xirr"] - summary["benchmark_xirr"], pct_fmt),
            ]
            if summary.get("benchmark_source"):
                rows.append(("Benchmark Source", summary["benchmark_source"], None))
        r = 2
        for label, val, fmt in rows:
            ws.write(r, 0, label, bold)
            ws.write(r, 1, val, fmt)
            r += 1

        # ---- Transactions sheet ----
        txns_out = txns[["Date_parsed", "Goal Name/Basket Name", "Fund", "Type",
                          "CashFlow_num", "Is_Switch"]].rename(columns={
            "Date_parsed": "Date", "CashFlow_num": "CashFlow (Rs.)",
            "Is_Switch": "Excluded (Switch)",
        })
        txns_out.to_excel(writer, sheet_name="Transactions", index=False, startrow=0)
        ws_t = writer.sheets["Transactions"]
        for c, name in enumerate(txns_out.columns):
            ws_t.write(0, c, name, header_fmt)
        ws_t.set_column("A:A", 14)
        ws_t.set_column("B:C", 38)
        ws_t.set_column("D:D", 12)
        ws_t.set_column("E:E", 16, rupee_fmt)
        ws_t.set_column("F:F", 16)

        # ---- Breakdown sheet (grouped by whichever category was picked in-app) ----
        sheet_name = f"{breakdown_label} Breakdown"[:31]  # Excel sheet-name limit
        breakdown_df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws_f = writer.sheets[sheet_name]
        for c, name in enumerate(breakdown_df.columns):
            ws_f.write(0, c, name, header_fmt)
        ws_f.set_column("A:A", 55)
        ws_f.set_column("B:E", 18, rupee_fmt)

        # ---- Benchmark detail sheet ----
        if bench_detail is not None and not bench_detail.empty:
            bd = bench_detail.rename(columns={"CashFlow": "CashFlow (Rs.)"})
            bd.to_excel(writer, sheet_name="Nifty50 Benchmark Detail", index=False)
            ws_b = writer.sheets["Nifty50 Benchmark Detail"]
            for c, name in enumerate(bd.columns):
                ws_b.write(0, c, name, header_fmt)
            ws_b.set_column("A:A", 14)
            ws_b.set_column("B:B", 45)
            ws_b.set_column("C:F", 16)

    return output.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("📊 Portfolio XIRR Dashboard — vs Nifty 50")
st.caption(
    "Upload your transaction report to calculate money-weighted returns (XIRR) "
    "and see how the same cash flows would have performed if invested in the "
    "Nifty 50 index instead."
)

uploaded = st.file_uploader("Upload transaction report (.csv)", type=["csv"])

if not uploaded:
    st.info("Waiting for a file. Expected columns: SN, Transaction ID, Goal Name/Basket "
            "Name, Fund, Date & Time, Type, Amount, and a final signed cash-flow column.")
    st.stop()

txns, current_value, as_of_date, platform_xirr = load_report(uploaded.getvalue())

if txns.empty:
    st.error("No usable transaction rows were found in this file.")
    st.stop()

st.success(f"Parsed {len(txns):,} transactions from "
           f"{txns['Date_parsed'].min():%d-%b-%Y} to {txns['Date_parsed'].max():%d-%b-%Y}.")

with st.expander("Preview parsed transactions"):
    st.dataframe(
        txns[["Date_parsed", "Goal Name/Basket Name", "Fund", "Type", "CashFlow_num"]]
        .rename(columns={"Date_parsed": "Date", "CashFlow_num": "Cash Flow (Rs.)"}),
        use_container_width=True,
    )

# --- Current value / as-of date, with manual override if not found in file ---
st.subheader("Current portfolio value")
if current_value is None:
    st.info("This report doesn't include a current portfolio value or as-of date — "
             "enter both below (e.g. from your latest CAS or the platform's dashboard) "
             "so XIRR can be calculated.")
col1, col2 = st.columns(2)
with col1:
    current_value = st.number_input(
        "Current portfolio value (Rs.)",
        min_value=0.0,
        value=float(current_value) if current_value else 0.0,
        step=1000.0,
        help="Auto-detected from the report's summary row if present; edit if needed.",
    )
with col2:
    default_date = as_of_date.date() if as_of_date else datetime.today().date()
    as_of_date_input = st.date_input("As-of date for the current value", value=default_date)
    as_of_date = datetime.combine(as_of_date_input, datetime.min.time())

if current_value <= 0:
    st.warning("Enter the current portfolio value above to compute XIRR.")
    st.stop()

# --- Portfolio XIRR ---
# Switch rows move money between funds inside the portfolio and are not
# real external cash flows, so they're excluded from XIRR and from the
# Invested / Withdrawn totals below.
external_txns = txns[~txns["Is_Switch"]]
switch_txns = txns[txns["Is_Switch"]]

port_cashflows = list(zip(external_txns["Date_parsed"], external_txns["CashFlow_num"]))
port_cashflows.append((as_of_date, current_value))
portfolio_xirr = xirr(port_cashflows)

total_invested = -external_txns.loc[external_txns["CashFlow_num"] < 0, "CashFlow_num"].sum()
total_withdrawn = external_txns.loc[external_txns["CashFlow_num"] > 0, "CashFlow_num"].sum()
absolute_gain = current_value + total_withdrawn - total_invested

total_switched = switch_txns["Amount_num"].sum()
if not switch_txns.empty:
    st.caption(f"ℹ️ {len(switch_txns):,} 'Switch' transactions totalling ₹{total_switched:,.0f} "
               "(gross, both legs) were found and excluded from Invested/Withdrawn and XIRR, "
               "since that money moved between funds rather than in or out of your pocket.")

unknown_txns = external_txns[external_txns["Type_Category"] == "unknown"]
if not unknown_txns.empty:
    unknown_types = ", ".join(sorted(unknown_txns["Type"].dropna().unique()))
    st.warning(f"{len(unknown_txns):,} transactions have an unrecognized Type "
               f"({unknown_types}) and were conservatively treated as money invested. "
               "Double-check these in the transaction preview above.")

# --- Benchmark ---
st.subheader("Nifty 50 benchmark")
run_benchmark = st.checkbox("Compare against Nifty 50", value=True)

benchmark_xirr = benchmark_value = benchmark_source = None
bench_detail = None

if run_benchmark:
    series = None

    if YF_AVAILABLE:
        with st.spinner("Fetching Nifty 50 historical prices (Yahoo Finance)..."):
            try:
                series = fetch_benchmark_series(external_txns["Date_parsed"].min(), as_of_date)
                benchmark_source = "Nifty 50 Index (^NSEI, Yahoo Finance)"
            except Exception as e:
                st.warning(f"Yahoo Finance fetch failed ({e}). Trying UTI Nifty 50 Index "
                           "Fund NAV via mfapi.in as a proxy instead...")
    else:
        st.info("`yfinance` isn't installed — trying UTI Nifty 50 Index Fund NAV "
                "via mfapi.in as a proxy instead.")

    if series is None:
        with st.spinner("Fetching UTI Nifty 50 Index Fund NAV (mfapi.in)..."):
            try:
                scheme_code, scheme_name = find_uti_nifty50_scheme_code()
                series = fetch_mfapi_nav_series(scheme_code)
                benchmark_source = f"{scheme_name} NAV (mfapi.in, proxy for Nifty 50)"
                st.info(f"Using **{scheme_name}** NAV as a Nifty 50 proxy (via mfapi.in), "
                        "since it's a real, investable fund tracking the index.")
            except Exception as e:
                st.error(f"Could not fetch a Nifty 50 proxy via mfapi.in either: {e}")

    if series is None:
        manual_csv = st.file_uploader(
            "Or upload a Nifty 50 / index-fund historical CSV manually (columns: Date, Close/NAV)",
            type=["csv"], key="manual_benchmark_csv",
        )
        if manual_csv is not None:
            try:
                series = parse_manual_benchmark_csv(manual_csv.getvalue())
                benchmark_source = "Manually uploaded CSV"
                st.success("Loaded benchmark prices from your uploaded file.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    if series is not None:
        # Only real external cash flows get replayed into the benchmark —
        # a Switch never added or removed money from the portfolio, so it
        # shouldn't add or remove benchmark units either.
        benchmark_xirr, benchmark_value, bench_detail = simulate_benchmark(
            external_txns, current_value, as_of_date, series
        )

# --- Metrics ---
st.subheader("Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Invested", f"₹{total_invested:,.0f}")
m2.metric("Total Withdrawn", f"₹{total_withdrawn:,.0f}")
m3.metric("Current Value", f"₹{current_value:,.0f}")
m4.metric("Absolute Gain", f"₹{absolute_gain:,.0f}")

m5, m6, m7 = st.columns(3)
m5.metric("Portfolio XIRR", f"{portfolio_xirr*100:,.2f}%" if not np.isnan(portfolio_xirr) else "N/A")
if benchmark_xirr is not None:
    m6.metric("Nifty 50 Benchmark XIRR", f"{benchmark_xirr*100:,.2f}%",
              delta=f"{(portfolio_xirr-benchmark_xirr)*100:,.2f} pts vs benchmark")
    if benchmark_source:
        m6.caption(f"Source: {benchmark_source}")
else:
    m6.metric("Nifty 50 Benchmark XIRR", "—")
if platform_xirr is not None:
    m7.metric("Platform-Reported XIRR", f"{platform_xirr:,.2f}%",
              help="Read directly from the source report, shown here as a cross-check.")

# --- Breakdown table, grouped by Fund / Goal / Type (switchable) ---
def build_breakdown(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.assign(
            Invested=lambda d: d["CashFlow_num"].where(d["CashFlow_num"] < 0, 0).abs(),
            Withdrawn=lambda d: d["CashFlow_num"].where(d["CashFlow_num"] > 0, 0),
            Switched=lambda d: d["Amount_num"].where(d["Is_Switch"], 0.0),
        )
        .groupby(group_col, as_index=False)[["Invested", "Withdrawn", "Switched"]]
        .sum()
        .assign(Net=lambda d: d["Invested"] - d["Withdrawn"])
        .sort_values("Invested", ascending=False)
    )

st.subheader("Breakdown")
group_label = st.selectbox("Group breakdown by", ["Fund", "Goal Name/Basket Name", "Type"])
breakdown_df = build_breakdown(txns, group_label)

with st.expander(f"{group_label}-wise invested / withdrawn amounts"):
    st.caption("Per-group XIRR isn't shown because the report only gives one "
               "current-value figure for the whole portfolio, not per group.")
    if not switch_txns.empty:
        if group_label == "Type":
            st.caption("The 'Switch' row shows money moved between funds — it's informational "
                       "and already excluded from the portfolio Invested/Withdrawn totals above.")
        else:
            st.caption("Includes Switch legs (money moved between funds), so totals here can run "
                       "higher than the portfolio-level Invested/Withdrawn figures above.")
    st.dataframe(breakdown_df, use_container_width=True)

# --- Excel export ---
summary = {
    "as_of_date": as_of_date,
    "total_invested": total_invested,
    "total_withdrawn": total_withdrawn,
    "current_value": current_value,
    "absolute_gain": absolute_gain,
    "portfolio_xirr": portfolio_xirr,
    "platform_xirr": platform_xirr,
    "benchmark_xirr": benchmark_xirr,
    "benchmark_value": benchmark_value,
    "benchmark_source": benchmark_source,
}
excel_bytes = build_excel(summary, txns, breakdown_df, group_label, bench_detail)

st.subheader("Download")
st.download_button(
    "⬇️ Download Excel Report",
    data=excel_bytes,
    file_name=f"portfolio_xirr_report_{as_of_date:%Y%m%d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
