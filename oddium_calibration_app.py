# oddium_calibration_app.py
# Streamlit app om voorspelde kansen + odds te tracken, te evalueren op calibratie en ROI.
import math
import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Oddium Calibration & ROI", layout="wide")

DATA_PATH = Path("data.csv")
LEDGER_PATH = Path("ledger.csv")

# ------------------ Helpers ------------------
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        # Backward compatibility & dtypes
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

        # Normalize columns
        expected_cols = ["timestamp","event","odds","pred_prob","stake","outcome"]
        for c in expected_cols:
            if c not in df.columns:
                # try common aliases
                aliases = {
                    "odds":["odd","quote","quotering","decimal_odds"],
                    "pred_prob":["prob","predicted_prob","kans","percentage","pred%"],
                    "stake":["bet_size","amount","inzet"],
                    "outcome":["result","win","hit"],
                    "event":["match","wedstrijd","game"]
                }
                if c in aliases:
                    for a in aliases[c]:
                        if a in df.columns:
                            df[c] = df[a]
                            break
                if c not in df.columns:
                    # Fill sensible defaults
                    if c == "timestamp":
                        df[c] = pd.NaT
                    elif c == "stake":
                        df[c] = 1.0
                    else:
                        df[c] = np.nan

        # Ensure numeric types
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(1.0)

        # outcome can be {0,1} or text
        def to_bin(x):
            if pd.isna(x): return np.nan
            if isinstance(x, str):
                x = x.strip().lower()
                if x in ("1","win","w","true","yes","y","hit","goed"): return 1
                if x in ("0","lose","l","false","no","n","miss","fout"): return 0
                try:
                    return int(float(x))
                except:
                    return np.nan
            try:
                return int(float(x))
            except:
                return np.nan

        df["outcome"] = df["outcome"].apply(to_bin)

        # Normalize prob: accept 0–1 or 0–100
        if (df["pred_prob"] > 1.01).any():
            df["pred_prob"] = df["pred_prob"] / 100.0

        # Clip to [0,1]
        df["pred_prob"] = df["pred_prob"].clip(0,1)

        # Fill timestamps
        if df["timestamp"].isna().all():
            df["timestamp"] = pd.Timestamp.now()
        else:
            df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())

        # Clean events
        df["event"] = df["event"].fillna("").astype(str)

        # outcome mag NaN zijn (open bet); filter pas bij analyses
        return df.dropna(subset=["odds","pred_prob"])

    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp","event","odds","pred_prob","stake","outcome"])

def save_data(df):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def add_record(event, odds, pred_prob, stake, outcome):
    df = load_data()
    now = pd.Timestamp.now()
    if outcome is None:
        out = np.nan
    else:
        out = 1 if outcome else 0
    row = {
        "timestamp": now,
        "event": event,
        "odds": odds,
        "pred_prob": pred_prob if pred_prob <= 1 else pred_prob/100.0,
        "stake": stake,
        "outcome": out,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_data(df)
    return df

def calc_roi(df):
    # Profit per bet: outcome * (odds-1)*stake - (1 - outcome)*stake
    prof = df["outcome"] * (df["odds"] - 1.0) * df["stake"] - (1 - df["outcome"]) * df["stake"]
    return prof.sum() / df["stake"].sum() if df["stake"].sum() > 0 else 0.0

def brier(df):
    try:
        return brier_score_loss(df["outcome"], df["pred_prob"])
    except Exception:
        return np.nan

def logloss(df):
    try:
        p = np.clip(df["pred_prob"].values, 1e-12, 1-1e-12)
        return float(log_loss(df["outcome"].values, p))
    except Exception:
        return np.nan

def bin_by_prob(df, n_bins=10):
    try:
        df = df.copy()
        df["prob_bin"] = pd.qcut(df["pred_prob"], q=n_bins, labels=False, duplicates="drop")
        return df
    except ValueError:
        df = df.copy()
        df["prob_bin"] = 0
        return df

def bin_by_odds(df, edges=(1.01,1.5,2,3,5,10,1000)):
    df = df.copy()
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges)-1)]
    df["odds_bin"] = pd.cut(df["odds"], bins=edges, right=False, labels=labels, include_lowest=True)
    return df

def ev_per_bet(row):
    # EV per stake unit: p*(odds-1) - (1-p)
    p = row["pred_prob"]
    o = row["odds"]
    return p*(o-1) - (1-p)

def realized_per_bet(row):
    return row["outcome"]*(row["odds"]-1) - (1-row["outcome"])
    


# ------- Ledger helpers (startbalans & stortingen) -------
def load_ledger():
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=["timestamp","type","amount","note"])
    led = pd.read_csv(LEDGER_PATH)
    led["timestamp"] = pd.to_datetime(led.get("timestamp"), errors="coerce")
    led["type"] = led.get("type","").astype(str)
    led["amount"] = pd.to_numeric(led.get("amount"), errors="coerce").fillna(0.0)
    led["note"] = led.get("note","").astype(str)
    return led

def save_ledger(ledger_df):
    ledger_df = ledger_df.copy()
    ledger_df["timestamp"] = pd.to_datetime(ledger_df["timestamp"], errors="coerce").fillna(pd.Timestamp.now())
    ledger_df.to_csv(LEDGER_PATH, index=False)

def current_start_amount(ledger_df, default_start=10.0):
    starts = ledger_df[ledger_df["type"]=="start"]["amount"]
    if len(starts):
        return float(starts.iloc[-1])
    return float(default_start)

def sum_deposits(ledger_df):
    return float(ledger_df[ledger_df["type"]=="deposit"]["amount"].sum())

# ------------------ UI ------------------
st.title("📈 Oddium – Calibratie, ROI & Balans")

# === Data laden ===
df = load_data()

# === EV & realized per bet eerst berekenen (KeyError fix) ===
df["theoretical_ev_per_stake"] = df.apply(ev_per_bet, axis=1)  # per 1 unit stake
df["realized_per_stake"] = df.apply(lambda r: realized_per_bet(r) if pd.notna(r["outcome"]) else np.nan, axis=1)
# Alleen bets met uitkomst voor analyses/statistieken:
df_eval = df.dropna(subset=["outcome"]).copy()

# === Balans & stortingen (bovenaan) ===
# === Balans bovenaan (groot) + ingestorte secties ===

# Ledger helpers blijven hetzelfde (load_ledger/save_ledger/etc.)
def ensure_start_balance(ledger_df, default_start=10.0):
    """Zorg dat er altijd een startregel is. Voeg 'start' toe als die ontbreekt."""
    if (ledger_df["type"] == "start").any():
        return ledger_df
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "type": "start",
        "amount": float(default_start),
        "note": "auto start"
    }])
    return pd.concat([ledger_df, new_row], ignore_index=True)

# ── bereken eerst EV/realized en df_eval (laat jouw bestaande code zo):
# df = load_data()
# df["theoretical_ev_per_stake"] = df.apply(ev_per_bet, axis=1)
# df["realized_per_stake"] = df.apply(lambda r: realized_per_bet(r) if pd.notna(r["outcome"]) else np.nan, axis=1)
# df_eval = df.dropna(subset=["outcome"]).copy()

# Ledger laden + start afdwingen (default €10)
ledger = ensure_start_balance(load_ledger(), default_start=10.0)

# Hulpgetallen
start_amt = current_start_amount(ledger, default_start=10.0)
deposits_total = sum_deposits(ledger)
realized_profit_total = float((df_eval["realized_per_stake"] * df_eval["stake"]).sum()) if len(df_eval) else 0.0

# Nieuw: splits balans in beschikbaar vs. incl. open
open_stake_total = float(df[df["outcome"].isna()]["stake"].sum())
available_balance = start_amt + deposits_total + realized_profit_total          # = wat al gespeeld/afgerond is
gross_balance      = available_balance + open_stake_total                       # = incl. open inzetten

# ── BALANS-BANNER ──
st.markdown(
    f"""
<div style="padding:16px; background:#0f172a; color:#fff; border-radius:14px; margin-bottom:12px;">
  <div style="display:flex; gap:16px; justify-content:space-between; align-items:center; flex-wrap:wrap;">
    <div style="flex:1; min-width:240px; text-align:center;">
      <div style="font-size:14px; opacity:.85;">Beschikbaar saldo (gerealiseerd)</div>
      <div style="font-size:44px; font-weight:800; line-height:1; margin:6px 0 2px;">€ {available_balance:,.2f}</div>
    </div>
    <div style="flex:1; min-width:240px; text-align:center;">
      <div style="font-size:14px; opacity:.85;">Totaal (incl. open inzetten)</div>
      <div style="font-size:32px; font-weight:800; line-height:1; margin:6px 0 2px;">€ {gross_balance:,.2f}</div>
      <div style="font-size:12px; opacity:.85;">Open inzet: € {open_stake_total:,.2f}</div>
    </div>
  </div>
  <div style="font-size:12px; opacity:.85; text-align:center; margin-top:8px;">
    Start: €{start_amt:,.2f} • Gestort: €{deposits_total:,.2f} • Gerealiseerde winst: €{realized_profit_total:,.2f}
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ───────────────── Invoerformulier: Nieuwe bet toevoegen ─────────────────
st.divider()
st.subheader("➕ Nieuwe bet toevoegen")
with st.expander("Formulier", expanded=True):
    cols = st.columns(2)
    event = cols[0].text_input("Event / Wedstrijd", placeholder="E.g. PSV - Ajax (BTTS)")
    odds = cols[0].number_input("Quotering (decimal)", min_value=1.01, step=0.01, value=2.00)
    pred_prob = cols[0].number_input(
        "Voorspelde kans",
        min_value=0.0, step=0.1, value=55.0,
        help="Mag 0–1 (bv. 0.55) of 0–100 (bv. 55)."
    )
    stake = cols[1].number_input("Inzet (units/€)", min_value=0.01, step=0.5, value=1.0)
    outcome_str = cols[1].selectbox("Uitkomst", ["Nog onbekend", "Win (1)", "Lose (0)"], index=0)

    submitted = st.button("Toevoegen", type="primary", use_container_width=True)

    if submitted:
        if outcome_str.startswith("Nog"):
            outcome_val = None  # sla op als NaN
        else:
            outcome_val = 1 if "Win" in outcome_str else 0
        add_record(event, odds, pred_prob, stake, outcome_val)
        st.success("Bet opgeslagen!")

# Na toevoegen: data opnieuw laden en afgeleide kolommen opnieuw berekenen
df = load_data()
df["theoretical_ev_per_stake"] = df.apply(ev_per_bet, axis=1)
df["realized_per_stake"] = df.apply(lambda r: realized_per_bet(r) if pd.notna(r["outcome"]) else np.nan, axis=1)
df_eval = df.dropna(subset=["outcome"]).copy()

# ── Sectie: Startbalans & stortingen (INGEKLAPT, NIET standaard zichtbaar) ──
with st.expander("💼 Startbalans & stortingen (klik om te openen)", expanded=False):
    c1, c2 = st.columns(2)

    with c1:
        new_start = st.number_input("Startbalans (EUR)", min_value=0.0, step=1.0, value=float(start_amt))
        if st.button("Opslaan/Reset startbalans", use_container_width=True):
            # verwijder bestaande startregels en zet nieuwe
            led = load_ledger()
            led = led[led["type"] != "start"].copy()
            led = pd.concat([led, pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "type": "start",
                "amount": float(new_start),
                "note": "reset start"
            }])], ignore_index=True)
            save_ledger(led)
            st.success("Startbalans opgeslagen. Herlaad de pagina voor update.")

    with c2:
        dep_amount = st.number_input("Nieuwe storting (EUR)", min_value=0.0, step=1.0, value=0.0)
        dep_note = st.text_input("Opmerking (optioneel)", "")
        if st.button("➕ Storten", use_container_width=True, disabled=(dep_amount <= 0.0)):
            led = load_ledger()
            led = pd.concat([led, pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "type": "deposit",
                "amount": float(dep_amount),
                "note": dep_note
            }])], ignore_index=True)
            save_ledger(led)
            st.success("Storting toegevoegd. Herlaad de pagina voor update.")

# ── Sectie: Winst per maand & week (INGEKLAPT) ──
with st.expander("📅 Winst per maand & week (klik om te openen)", expanded=False):
    if len(df_eval):
        df_eval2 = df_eval.copy()
        df_eval2["profit"] = df_eval2["realized_per_stake"] * df_eval2["stake"]

        st.markdown("**Winst per maand**")
        monthly = (df_eval2
                   .assign(month=lambda x: x["timestamp"].dt.to_period("M").astype(str))
                   .groupby("month")["profit"].sum().reset_index()
                   .rename(columns={"month": "Maand", "profit": "Winst (€)"}))
        st.dataframe(monthly, use_container_width=True, height=180)

        st.markdown("**Winst per week**")
        weekly = (df_eval2
                  .assign(week=lambda x: x["timestamp"].dt.to_period("W").astype(str))
                  .groupby("week")["profit"].sum().reset_index()
                  .rename(columns={"week": "Week", "profit": "Winst (€)"}))
        st.dataframe(weekly, use_container_width=True, height=180)
    else:
        st.info("Nog geen gerealiseerde winst: vul eerst uitslagen in.")


# ------------------ Uitslagen bijwerken ------------------
with st.expander("✅ Uitslagen bijwerken", expanded=False):
    st.caption("Zet open bets op Win/Lose of corrigeer eerdere uitkomsten. Alleen 'Outcome' is bewerkbaar.")
    df_view = df.copy()

    def out_to_str(x):
        if pd.isna(x): return "Nog onbekend"
        return "Win (1)" if int(x) == 1 else "Lose (0)"

    df_view["Outcome"] = df_view["outcome"].apply(out_to_str)

    edited = st.data_editor(
        df_view[["timestamp","event","odds","pred_prob","stake","Outcome"]],
        use_container_width=True,
        height=360,
        column_config={
            "Outcome": st.column_config.SelectboxColumn(
                "Outcome",
                options=["Nog onbekend", "Win (1)", "Lose (0)"],
                help="Kies de uitkomst. Laat 'Nog onbekend' staan voor open bets."
            )
        },
        disabled=["timestamp","event","odds","pred_prob","stake"],
        key="editor_outcomes"
    )

    if st.button("💾 Wijzigingen opslaan", use_container_width=True):
        mapping = {"Nog onbekend": np.nan, "Win (1)": 1, "Lose (0)": 0}
        new_outcomes = edited["Outcome"].map(mapping)
        df["outcome"] = new_outcomes.values
        save_data(df)
        st.success("Uitslagen bijgewerkt en opgeslagen.")
        # refresh derived views
        df["realized_per_stake"] = df.apply(lambda r: realized_per_bet(r) if pd.notna(r["outcome"]) else np.nan, axis=1)

# ------------------ Kerncijfers ------------------
st.divider()
st.subheader("📏 Kerncijfers")
colA, colB, colC, colD = st.columns(4)

if len(df_eval):
    total_roi = calc_roi(df_eval)
    brier_sc  = brier(df_eval)
    logl      = logloss(df_eval)
    mean_odds = df_eval["odds"].mean()

    colA.metric("Totale ROI", f"{total_roi*100:.2f}%")
    colB.metric("Brier score (↓ beter)", f"{brier_sc:.4f}" if not math.isnan(brier_sc) else "—")
    colC.metric("Log-loss (↓ beter)", f"{logl:.4f}" if not math.isnan(logl) else "—")
    colD.metric("Gem. odds", f"{mean_odds:.2f}")
else:
    colA.metric("Totale ROI", "—")
    colB.metric("Brier score (↓ beter)", "—")
    colC.metric("Log-loss (↓ beter)", "—")
    colD.metric("Gem. odds", "—")

if not len(df_eval):
    st.info("Nog geen bets met uitkomst. Voeg uitslagen toe om analyses te zien.")
    st.stop()

# ------------------ Calibratie per kans-deciel ------------------
st.divider()
st.subheader("🎯 Calibratie per kans-bin")
df_prob = bin_by_prob(df_eval, n_bins=10)
cal = (df_prob
       .groupby("prob_bin")
       .agg(n=("outcome","count"),
            pred=("pred_prob","mean"),
            obs=("outcome","mean"))
       .reset_index()
      )
cal = cal.dropna(subset=["prob_bin"])
cal["pred_%"] = cal["pred"] * 100
cal["obs_%"] = cal["obs"] * 100

left, right = st.columns([1.1, 1])
with left:
    st.markdown("**Reliability Plot** (verwacht vs. gezien)")
    fig = plt.figure()
    x = cal["pred"]
    y = cal["obs"]
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(x, y, marker="o")
    plt.xlabel("Voorspelde kans")
    plt.ylabel("Geobserveerde hitrate")
    plt.grid(True, linestyle=":")
    st.pyplot(fig, clear_figure=True)

with right:
    st.markdown("**Tabel per deciel**")
    st.dataframe(
        cal.rename(columns={"n":"#","pred_%":"Pred %","obs_%":"Obs %"})
           [["#","Pred %","Obs %"]]
           .style.format({"Pred %":"{:.1f}","Obs %":"{:.1f}","#":"{:,.0f}"}),
        use_container_width=True, height=320
    )

# ------------------ Performance per odds-bucket ------------------
st.divider()
st.subheader("💸 Performance per odds-bucket")
edges_default = (1.01, 1.5, 2, 3, 5, 10, 1000)
edges_str = st.text_input(
    "Custom odds-bins (komma-gescheiden, min→max)",
    value=", ".join(str(x) for x in edges_default),
    help="Bijv: 1.01,1.8,2.2,3,5,1000"
)
try:
    edges = tuple(float(x.strip()) for x in edges_str.split(",") if x.strip())
    if len(edges) < 2: raise ValueError
except:
    edges = edges_default

def _calc_roi_group(g):
    prof = g["outcome"]*(g["odds"]-1.0)*g["stake"] - (1-g["outcome"])*g["stake"]
    return prof.sum()/g["stake"].sum() if g["stake"].sum() > 0 else 0.0

df_odds = bin_by_odds(df_eval, edges)
agg_odds = (df_odds.groupby("odds_bin")
            .apply(lambda g: pd.Series({
                "#": len(g),
                "Hitrate %": 100 * g["outcome"].mean() if len(g) else np.nan,
                "ROI %": 100 * _calc_roi_group(g),
                "Avg odds": g["odds"].mean(),
            }))
            .reset_index()
           )

st.dataframe(agg_odds, use_container_width=True)

st.markdown("**ROI per odds-bucket**")
fig2 = plt.figure()
plt.bar(agg_odds["odds_bin"].astype(str), agg_odds["ROI %"])
plt.axhline(0, linestyle="--")
plt.xticks(rotation=30, ha="right")
plt.ylabel("ROI %")
plt.grid(axis="y", linestyle=":")
st.pyplot(fig2, clear_figure=True)

# ------------------ EV vs Realized ------------------
st.divider()
st.subheader("⚖️ EV vs. Realized (per bet en totaal)")
tot_ev = (df["theoretical_ev_per_stake"] * df["stake"]).sum()
tot_real = (df_eval["realized_per_stake"] * df_eval["stake"]).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Theoretische EV (units/€)", f"{tot_ev:.2f}")
c2.metric("Gerealiseerd resultaat (units/€)", f"{tot_real:.2f}")
c3.metric("Verschil (units/€)", f"{(tot_real - tot_ev):.2f}")

st.markdown("**Scatter: voorspelde kans vs. realized/EV** (per bet)")
fig3 = plt.figure()
plt.scatter(df_eval["pred_prob"], df_eval["realized_per_stake"], label="Realized per stake", alpha=0.7)
plt.scatter(df["pred_prob"], df["theoretical_ev_per_stake"], label="EV per stake", alpha=0.7, marker="x")
plt.xlabel("Voorspelde kans")
plt.ylabel("Winst per stake (units/€)")
plt.legend()
plt.grid(True, linestyle=":")
st.pyplot(fig3, clear_figure=True)

# ------------------ Filters & Export ------------------
st.divider()
st.subheader("🔎 Filteren & export")
with st.expander("Filter op datum", expanded=False):
    min_date = df["timestamp"].min().date() if not df["timestamp"].isna().all() else dt.date.today()
    max_date = df["timestamp"].max().date() if not df["timestamp"].isna().all() else dt.date.today()
    start, end = st.date_input("Bereik", value=(min_date, max_date))
    if isinstance(start, tuple):  # streamlit older quirk
        start, end = start
    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    df_filtered = df[mask].copy()
    st.write(f"Geselecteerd: **{len(df_filtered)}** bets")
    if len(df_filtered):
        st.dataframe(df_filtered.sort_values("timestamp", ascending=False), use_container_width=True, height=240)
        st.download_button(
            "⬇️ Download gefilterde CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"data_{start}_{end}.csv",
            mime="text/csv"
        )

st.caption("Tip: `pred_prob` mag als percentage of fractie. Odds in decimaal. Outcome: Win=1, Lose=0 of leeg (Nog onbekend).")
