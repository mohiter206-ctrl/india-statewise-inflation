
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import itertools
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India CPI Research Intelligence", page_icon="📊", layout="wide")
DATA_PATH = Path(__file__).parent / "Statewise_CPI_28_States_Complete.csv"
MONTHS = {m:i for i,m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
MONTH_NAMES = {v:k for k,v in MONTHS.items()}
# Full candidate set restored for a controlled 12-model comparison.
# Keep benchmark, statistical, tree-ensemble and boosting methods together.
MODEL_NAMES = ["Naive", "Seasonal Naive", "Moving Average (3-month)", "Drift",
               "Holt Linear", "Theta", "ETS (Holt-Winters)", "SARIMA",
               "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost"]

st.markdown("""
<style>
.block-container{padding-top:1.1rem;max-width:1500px}
.hero{padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(115deg,#10243a,#245b78);color:#fff;margin-bottom:1rem}
.hero h1{font-size:2rem;margin:0 0 .35rem 0}.hero p{margin:0;color:#d9e7ef}
div[data-testid="stMetric"]{background:rgba(120,140,160,.08);padding:12px 14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("sector","year","name","month"): ren[c] = low.title()
    return df.rename(columns=ren)

def to_wide(df):
    """Accept the project's wide format, or a simple long format."""
    df = normalize_columns(df)
    if {"Sector","Year","Name"}.issubset(df.columns):
        wide = df.copy()
        wide["Year"] = pd.to_numeric(wide["Year"], errors="coerce")
        wide["Name"] = wide["Name"].astype(str).str.strip().str.title()
        wide["Date"] = pd.to_datetime(
            dict(year=wide["Year"], month=wide["Name"].map(MONTHS), day=1), errors="coerce")
        wide = wide.dropna(subset=["Date","Sector"])
        states = [c for c in wide.columns if c not in ["Sector","Year","Name","Date","Base Year","Base_Year"]]
        for c in states: wide[c] = pd.to_numeric(wide[c], errors="coerce")
        wide["Sector"] = wide["Sector"].astype(str).str.strip()
        return wide[["Date","Sector"]+states], states
    # Long format: Date, Sector, State/UT, CPI (or Year, Month, State/UT, Sector, CPI)
    lower = {c.lower():c for c in df.columns}
    value_col = next((lower[k] for k in ["cpi","index","value"] if k in lower), None)
    state_col = next((lower[k] for k in ["state/ut","state","state_ut"] if k in lower), None)
    sector_col = lower.get("sector")
    date_col = lower.get("date")
    if value_col and state_col and sector_col and date_col:
        temp = df[[date_col,sector_col,state_col,value_col]].copy()
        temp.columns = ["Date","Sector","State/UT","CPI"]
        temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        temp["CPI"] = pd.to_numeric(temp["CPI"], errors="coerce")
        wide = temp.pivot_table(index=["Date","Sector"],columns="State/UT",values="CPI",aggfunc="last").reset_index()
        states = [c for c in wide.columns if c not in ["Date","Sector"]]
        return wide, states
    raise ValueError("Unsupported layout. Use wide format (Sector, Year, Name, State columns) or long format (Date, Sector, State/UT, CPI).")

def canonical_panel(wide):
    d = wide.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.to_period("M").dt.to_timestamp()
    d["Sector"] = d["Sector"].astype(str).str.strip()
    states = [c for c in d.columns if c not in ["Date","Sector"]]
    for c in states: d[c] = pd.to_numeric(d[c], errors="coerce")
    # Duplicate Sector-Date rows are consolidated with last non-null value per state.
    d = d.sort_values("Date").groupby(["Sector","Date"], as_index=False).last()
    return d, states

@st.cache_data
def load_base():
    raw = pd.read_csv(DATA_PATH)
    wide, _ = to_wide(raw)
    return canonical_panel(wide)[0]

base = load_base()
st.markdown('<div class="hero"><h1>India CPI Research Intelligence</h1><p>State-wise inflation analysis · Statistical and machine-learning forecasting · Prediction uncertainty</p></div>', unsafe_allow_html=True)
st.caption("Monthly CPI analysis | 28 State/UTs | Rural, Urban and Rural+Urban sectors | January 2011–May 2023")

# Session state for upload and merge audit
if "combined_data" not in st.session_state:
    st.session_state.combined_data = base.copy()
if "active_data" not in st.session_state:
    st.session_state.active_data = None
if "active_dataset_label" not in st.session_state:
    st.session_state.active_dataset_label = "Original dataset"
if "merge_log" not in st.session_state:
    st.session_state.merge_log = pd.DataFrame()
if "upload_name" not in st.session_state:
    st.session_state.upload_name = ""

with st.sidebar:
    st.header("Dashboard controls")
    st.caption("All analysis uses the active dataset selected in Data Quality.")
    st.subheader("Add / extend CPI data")
    uploaded = st.file_uploader("Upload additional CSV or Excel", type=["csv","xlsx","xls"])
    base_year_confirm = st.selectbox("Uploaded data CPI base year", ["2012=100","2024=100","2010=100","Other / unknown"], index=0,
        help="Only combine index levels with the existing series if they use the same base and compatible definitions.")
    confirm_compatible = st.checkbox("I confirm the uploaded series is compatible with the existing 2012=100 series")
    merge_mode = st.selectbox("Merge rule for conflicting cells", ["Uploaded non-missing values take priority","Keep existing values"], index=0)
    if uploaded is not None and st.button("Validate and combine uploaded data", type="primary", use_container_width=True):
        try:
            if base_year_confirm != "2012=100" or not confirm_compatible:
                raise ValueError("Base-year compatibility is not confirmed. Rebase/link the series officially before merging index levels.")
            extra_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            extra_wide, extra_states = to_wide(extra_raw)
            extra, extra_states = canonical_panel(extra_wide)
            current = st.session_state.combined_data.copy()
            all_states = sorted(set([c for c in current.columns if c not in ["Date","Sector"]]) | set(extra_states))
            keys = ["Date","Sector"]
            for c in all_states:
                if c not in current: current[c] = np.nan
                if c not in extra: extra[c] = np.nan
            cur = current.set_index(keys)
            ext = extra.set_index(keys)
            union_idx = cur.index.union(ext.index).sort_values()
            cur = cur.reindex(union_idx); ext = ext.reindex(union_idx)
            log_rows = []
            merged = cur.copy()
            for c in all_states:
                oldv, newv = cur[c], ext[c]
                conflict = oldv.notna() & newv.notna() & (~np.isclose(oldv.fillna(0),newv.fillna(0),rtol=0,atol=1e-9))
                for idx in conflict[conflict].index:
                    log_rows.append({"Date":idx[0],"Sector":idx[1],"State/UT":c,
                                     "Existing value":oldv.loc[idx],"Uploaded value":newv.loc[idx],
                                     "Selected value":newv.loc[idx] if merge_mode.startswith("Uploaded") else oldv.loc[idx],
                                     "Action":merge_mode})
                if merge_mode.startswith("Uploaded"):
                    merged[c] = newv.combine_first(oldv)
                else:
                    merged[c] = oldv.combine_first(newv)
            merged = merged.reset_index()
            st.session_state.combined_data = merged.sort_values(["Sector","Date"]).reset_index(drop=True)
            st.session_state.active_data = None
            st.session_state.active_dataset_label = "Combined dataset (uploaded data merged)"
            st.session_state.merge_log = pd.DataFrame(log_rows)
            st.session_state.upload_name = uploaded.name
            st.success(f"Combined {len(extra):,} uploaded sector-month rows. {len(log_rows):,} conflicting cells logged.")
        except Exception as e:
            st.error(f"Upload not combined: {e}")

data = (st.session_state.active_data.copy() if st.session_state.active_data is not None else st.session_state.combined_data.copy())
states = [c for c in data.columns if c not in ["Date","Sector"]]
sectors = sorted(data["Sector"].dropna().unique(), key=lambda x: ["Rural","Urban","Rural+Urban","Rural+Urban"].index(x) if x in ["Rural","Urban","Rural+Urban"] else 99)
with st.sidebar:
    st.divider()
    sector = st.selectbox("Sector", sectors, index=sectors.index("Rural+Urban") if "Rural+Urban" in sectors else 0)
    state = st.selectbox("State / UT", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
    dmin, dmax = data["Date"].min().date(), data["Date"].max().date()
    dates = st.date_input("Analysis period", value=(dmin,dmax), min_value=dmin, max_value=dmax)
    if isinstance(dates,tuple) and len(dates)==2:
        start,end = pd.Timestamp(dates[0]),pd.Timestamp(dates[1])
    else: start,end = pd.Timestamp(dmin),pd.Timestamp(dmax)

long = data.melt(id_vars=["Date","Sector"],value_vars=states,var_name="State/UT",value_name="CPI")
long = long.sort_values(["State/UT","Sector","Date"])
long["Inflation_YoY"] = long.groupby(["State/UT","Sector"])["CPI"].transform(lambda s:(s/s.shift(12)-1)*100)
missing_total = int(data[states].isna().sum().sum())
expected_rows = len(sectors) * data["Date"].nunique()
available_cpi = int(data[states].notna().sum().sum())

tabs = st.tabs(["Overview","Data Quality","Explore Data (EDA)","Compare States","Forecasting Lab","Model Evaluation Guidance"])

with tabs[0]:
    st.subheader("Research overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("States / UTs",len(states))
    c2.metric("Latest month",data["Date"].max().strftime("%b %Y"))
    c3.metric("Missing CPI cells",f"{missing_total:,}",help="Individual blank State/UT × month × sector entries, not whole missing months.")
    c4.metric("CPI observations",f"{available_cpi:,}")
    sector_long = long[(long.Sector==sector)&long.Date.between(start,end)]
    monthly = sector_long.groupby("Date",as_index=False).agg(mean_cpi=("CPI","mean"),mean_yoy=("Inflation_YoY","mean"),states_available=("CPI","count"))
    f=px.line(monthly,x="Date",y="mean_cpi",title=f"Unweighted mean state CPI index · {sector}")
    f.update_layout(height=360,yaxis_title="Mean CPI index",xaxis_title="")
    st.plotly_chart(f,use_container_width=True)
    f=px.line(monthly,x="Date",y="mean_yoy",title=f"Unweighted mean state YoY inflation · {sector}")
    f.update_layout(height=340,yaxis_title="Inflation (%)",xaxis_title="")
    st.plotly_chart(f,use_container_width=True)
    st.info("These are unweighted averages of available State/UT observations, not the official all-India CPI inflation rate.")

with tabs[2]:
    st.subheader(f"Explore data · {state} · {sector}")
    s=long[(long["State/UT"]==state)&(long.Sector==sector)&long.Date.between(start,end)].sort_values("Date")
    v=s.dropna(subset=["CPI"])
    if v.empty: st.warning("No CPI values are available for this selection.")
    else:
        latest=v.iloc[-1]
        a,b,c,d=st.columns(4)
        a.metric("Latest CPI",f"{latest.CPI:.2f}")
        a_date=latest.Date.strftime("%b %Y")
        lastinfl=s.dropna(subset=["Inflation_YoY"])
        b.metric("Latest YoY inflation",f"{lastinfl.iloc[-1].Inflation_YoY:.2f}%" if len(lastinfl) else "—")
        c.metric("Available months",len(v))
        d.metric("Missing cells",len(s)-len(v))
        f=px.line(s,x="Date",y="CPI",markers=True,title="CPI index over time")
        f.update_layout(height=350,xaxis_title="",yaxis_title="CPI index")
        st.plotly_chart(f,use_container_width=True)
        f=px.line(s,x="Date",y="Inflation_YoY",title="Year-on-year inflation")
        f.add_hline(y=0,line_dash="dash")
        f.update_layout(height=330,xaxis_title="",yaxis_title="Inflation (%)")
        st.plotly_chart(f,use_container_width=True)
        st.dataframe(s[["Date","CPI","Inflation_YoY"]],use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Cross-state and sector comparison")
    chosen=st.multiselect("States / UTs",states,default=states[:min(6,len(states))])
    measure=st.radio("Measure",["CPI index","YoY inflation (%)"],horizontal=True)
    col="CPI" if measure=="CPI index" else "Inflation_YoY"
    cp=long[(long.Sector==sector)&long["State/UT"].isin(chosen)&long.Date.between(start,end)]
    f=px.line(cp,x="Date",y=col,color="State/UT",title=f"{measure} · {sector}")
    f.update_layout(height=450,xaxis_title="",yaxis_title=measure)
    st.plotly_chart(f,use_container_width=True)
    st.markdown("#### Compare Rural, Urban and Combined sectors")
    sector_compare=long[(long["State/UT"]==state)&long.Date.between(start,end)]
    f=px.line(sector_compare,x="Date",y="CPI",color="Sector",title=f"{state}: sector comparison")
    f.update_layout(height=380,xaxis_title="",yaxis_title="CPI index")
    st.plotly_chart(f,use_container_width=True)

with tabs[2]:
    st.subheader("Advanced exploratory data analysis")
    s=long[(long["State/UT"]==state)&(long.Sector==sector)].sort_values("Date").set_index("Date")["CPI"].asfreq("MS")
    st.markdown("#### Descriptive statistics")
    st.dataframe(s.describe().to_frame("CPI index").round(3),use_container_width=True)
    st.markdown("#### Monthly seasonality of YoY inflation")
    yoy=(s/s.shift(12)-1)*100
    season=pd.DataFrame({"Month":yoy.index.month,"YoY inflation":yoy.values}).dropna()
    if not season.empty:
        season["Month name"]=season["Month"].map(MONTH_NAMES)
        order=list(MONTHS.keys())
        f=px.box(season,x="Month name",y="YoY inflation",category_orders={"Month name":order},title="Distribution of YoY inflation by calendar month")
        f.update_layout(height=380,xaxis_title="",yaxis_title="Inflation (%)")
        st.plotly_chart(f,use_container_width=True)
    try:
        from statsmodels.tsa.seasonal import STL
        clean=s.interpolate(method="time",limit=2,limit_area="inside")
        if clean.notna().sum()>=36 and clean.isna().sum()==0:
            stl=STL(clean,period=12,robust=True).fit()
            comp=pd.DataFrame({"Observed":clean,"Trend":stl.trend,"Seasonal":stl.seasonal,"Remainder":stl.resid})
            f=px.line(comp,title="STL decomposition (robust, 12-month seasonality)")
            f.update_layout(height=520)
            st.plotly_chart(f,use_container_width=True)
            st.caption("STL is used as an exploratory decomposition. Interpolation is temporary and only supports this diagnostic; it does not modify the active dataset.")
        else: st.info("STL decomposition requires at least 36 monthly observations and no unresolved gaps after the limited diagnostic-only interpolation.")
    except Exception as e: st.info(f"STL decomposition unavailable for this series: {e}")

    st.markdown("#### Stationarity and dependence diagnostics")
    yoy_clean = yoy.dropna()
    if len(yoy_clean) >= 36:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
        from statsmodels.stats.stattools import jarque_bera
        import matplotlib.pyplot as plt
        c1,c2=st.columns(2)
        try:
            adf=adfuller(yoy_clean,autolag="AIC")
            c1.metric("ADF p-value",f"{adf[1]:.4f}")
            c1.caption("H₀: unit root / non-stationarity. Small p-value is evidence against H₀.")
        except Exception as e: c1.warning(f"ADF unavailable: {e}")
        try:
            kp=kpss(yoy_clean,regression="c",nlags="auto")
            c2.metric("KPSS p-value",f"{kp[1]:.4f}")
            c2.caption("H₀: level stationarity. Small p-value is evidence against H₀.")
        except Exception as e: c2.warning(f"KPSS unavailable: {e}")
        st.caption("Read ADF and KPSS jointly: their null hypotheses are opposite. Neither test alone proves stationarity.")
        try:
            lagmax=min(24,max(5,len(yoy_clean)//4))
            fig,ax=plt.subplots(figsize=(10,3.2)); plot_acf(yoy_clean,lags=lagmax,ax=ax); st.pyplot(fig); plt.close(fig)
            fig,ax=plt.subplots(figsize=(10,3.2)); plot_pacf(yoy_clean,lags=min(lagmax,len(yoy_clean)//2-1),method="ywm",ax=ax); st.pyplot(fig); plt.close(fig)
        except Exception as e: st.info(f"ACF/PACF unavailable: {e}")
        try:
            lb=acorr_ljungbox(yoy_clean,lags=[6,12,18,24],return_df=True)
            st.markdown("**Ljung–Box test** (H₀: no autocorrelation up to the tested lag)")
            st.dataframe(lb.round(4),use_container_width=True)
        except Exception as e: st.info(f"Ljung–Box unavailable: {e}")
        st.caption("The Ljung–Box test here describes dependence in the inflation series. Model residual diagnostics are reported separately in Model Evaluation.")
    else:
        st.info("At least 36 observed YoY values are recommended for the advanced time-series diagnostics.")

    st.markdown("#### Rolling trend and volatility")
    if len(yoy_clean)>=36:
        roll_mean=yoy_clean.rolling(12).mean(); roll_sd=yoy_clean.rolling(12).std()
        fig=go.Figure(); fig.add_trace(go.Scatter(x=yoy_clean.index,y=yoy_clean.values,name="YoY inflation"))
        fig.add_trace(go.Scatter(x=roll_mean.index,y=roll_mean,name="12-month rolling mean"))
        fig.add_trace(go.Scatter(x=roll_sd.index,y=roll_sd,name="12-month rolling SD",yaxis="y2"))
        fig.update_layout(title="Rolling location and volatility",height=400,yaxis_title="Inflation (%)",yaxis2=dict(title="Rolling SD",overlaying="y",side="right"))
        st.plotly_chart(fig,use_container_width=True)

def longest_contiguous(series):
    s=series.asfreq("MS")
    arr=s.isna().to_numpy()
    if arr.all(): return s.iloc[:0]
    last=np.where(arr)[0]
    start=int(last[-1]+1) if len(last) else 0
    return s.iloc[start:].dropna()

def make_features(y):
    z=pd.DataFrame({"y":y})
    for lag in [1,2,3,6,12,24]: z[f"lag{lag}"]=z.y.shift(lag)
    return z.dropna()

def one_step(train,model):
    train=pd.Series(train,dtype=float)
    if train.isna().any(): raise ValueError("Missing values in training window.")
    if model=="Naive": return float(train.iloc[-1])
    if model=="Seasonal Naive":
        if len(train)<12: raise ValueError("Needs 12+ months.")
        return float(train.iloc[-12])
    if model=="Moving Average (3-month)":
        if len(train)<3: raise ValueError("Needs 3+ months.")
        return float(train.iloc[-3:].mean())
    if model=="Drift":
        if len(train)<2: raise ValueError("Needs 2+ months.")
        return float(train.iloc[-1]+(train.iloc[-1]-train.iloc[0])/(len(train)-1))
    if model=="Holt Linear":
        from statsmodels.tsa.holtwinters import Holt
        fit=Holt(train,initialization_method="estimated").fit(optimized=True)
        return float(fit.forecast(1).iloc[0])
    if model=="Theta":
        from statsmodels.tsa.forecasting.theta import ThetaModel
        if len(train)<24: raise ValueError("Needs 24+ months.")
        return float(ThetaModel(train,period=12,deseasonalize=True).fit().forecast(1).iloc[0])
    if model=="ETS (Holt-Winters)":
        if len(train)<24: raise ValueError("Needs 24+ months.")
        fit=ExponentialSmoothing(train,trend="add",damped_trend=True,seasonal="add",seasonal_periods=12,initialization_method="estimated").fit(optimized=True)
        return float(fit.forecast(1).iloc[0])
    if model=="SARIMA":
        if len(train)<36: raise ValueError("Needs 36+ months.")
        fit=SARIMAX(train,order=(1,0,1),seasonal_order=(1,0,0,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        return float(fit.forecast(1).iloc[0])
    features=make_features(train)
    if len(features)<20: raise ValueError("Not enough lagged rows for ML model.")
    X,y=features.drop(columns="y"),features.y
    if model=="Random Forest":
        m=RandomForestRegressor(n_estimators=60,max_depth=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
    elif model=="Extra Trees":
        from sklearn.ensemble import ExtraTreesRegressor
        m=ExtraTreesRegressor(n_estimators=60,max_depth=6,min_samples_leaf=2,random_state=42,n_jobs=-1)
    elif model=="Gradient Boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    elif model=="XGBoost":
        from xgboost import XGBRegressor
        m=XGBRegressor(n_estimators=60,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",random_state=42,n_jobs=4)
    else: raise ValueError("Unknown model")
    m.fit(X,y)
    row={f"lag{k}":float(train.iloc[-k]) for k in [1,2,3,6,12,24]}
    return float(m.predict(pd.DataFrame([row]))[0])

def rolling_eval(y,model,origins=18,min_train=36):
    """One-step expanding-window evaluation; retain model errors for diagnosis."""
    y=y.asfreq("MS")
    start=max(min_train,len(y)-origins)
    rows=[]; errors=[]
    for pos in range(start,len(y)):
        if pd.isna(y.iloc[pos]): continue
        train=longest_contiguous(y.iloc[:pos])
        if len(train)<min_train: continue
        if train.index[-1] != y.index[pos]-pd.offsets.MonthBegin(1): continue
        try:
            pred=one_step(train,model)
            if not np.isfinite(pred): raise ValueError("Model returned a non-finite prediction.")
            rows.append({"Date":y.index[pos],"Actual":float(y.iloc[pos]),"Predicted":float(pred)})
        except Exception as exc:
            errors.append(f"{y.index[pos]:%Y-%m}: {type(exc).__name__}: {exc}")
    if len(rows)<5:
        detail="; ".join(errors[-3:]) if errors else "Not enough consecutive observed test months."
        raise ValueError(f"Only {len(rows)} valid origins (minimum 5). {detail}")
    out=pd.DataFrame(rows).set_index("Date")
    out.attrs["fit_errors"]=errors
    return out

def multi_step_forecast(train, model, horizon):
    """Fit once at each forecast origin and recursively predict the requested horizon."""
    train=pd.Series(train,dtype=float)
    if train.isna().any(): raise ValueError("Missing values in training window.")
    horizon=int(horizon)
    if model=="Naive":
        return np.repeat(float(train.iloc[-1]),horizon)
    if model=="Seasonal Naive":
        if len(train)<12: raise ValueError("Needs 12+ months.")
        return np.resize(train.iloc[-12:].to_numpy(dtype=float),horizon)
    if model=="Moving Average (3-month)":
        hist=train.copy(); preds=[]
        for _ in range(horizon):
            pred=float(hist.iloc[-3:].mean()); preds.append(pred)
            hist.loc[hist.index[-1]+pd.offsets.MonthBegin(1)]=pred
        return np.asarray(preds)
    if model=="Drift":
        if len(train)<2: raise ValueError("Needs 2+ months.")
        drift=float((train.iloc[-1]-train.iloc[0])/(len(train)-1))
        return np.asarray([float(train.iloc[-1]+step*drift) for step in range(1,horizon+1)])
    if model=="Holt Linear":
        from statsmodels.tsa.holtwinters import Holt
        fit=Holt(train,initialization_method="estimated").fit(optimized=True)
        return np.asarray(fit.forecast(horizon),dtype=float)
    if model=="Theta":
        from statsmodels.tsa.forecasting.theta import ThetaModel
        if len(train)<24: raise ValueError("Needs 24+ months.")
        fit=ThetaModel(train,period=12,deseasonalize=True).fit()
        return np.asarray(fit.forecast(horizon),dtype=float)
    if model=="ETS (Holt-Winters)":
        if len(train)<24: raise ValueError("Needs 24+ months.")
        fit=ExponentialSmoothing(train,trend="add",damped_trend=True,seasonal="add",seasonal_periods=12,initialization_method="estimated").fit(optimized=True)
        return np.asarray(fit.forecast(horizon),dtype=float)
    if model=="SARIMA":
        if len(train)<36: raise ValueError("Needs 36+ months.")
        fit=SARIMAX(train,order=(1,0,1),seasonal_order=(1,0,0,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        return np.asarray(fit.forecast(horizon),dtype=float)
    features=make_features(train)
    if len(features)<20: raise ValueError("Not enough lagged rows for ML model.")
    X,target=features.drop(columns="y"),features.y
    if model=="Random Forest":
        estimator=RandomForestRegressor(n_estimators=60,max_depth=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
    elif model=="Extra Trees":
        from sklearn.ensemble import ExtraTreesRegressor
        estimator=ExtraTreesRegressor(n_estimators=60,max_depth=6,min_samples_leaf=2,random_state=42,n_jobs=-1)
    elif model=="Gradient Boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        estimator=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    elif model=="XGBoost":
        from xgboost import XGBRegressor
        estimator=XGBRegressor(n_estimators=60,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",random_state=42,n_jobs=4)
    else: raise ValueError("Unknown model")
    estimator.fit(X,target)
    hist=train.copy(); preds=[]
    for _ in range(horizon):
        row={f"lag{k}":float(hist.iloc[-k]) for k in [1,2,3,6,12,24]}
        pred=float(estimator.predict(pd.DataFrame([row]))[0])
        if not np.isfinite(pred): raise ValueError("Model returned a non-finite prediction.")
        preds.append(pred)
        hist.loc[hist.index[-1]+pd.offsets.MonthBegin(1)]=pred
    return np.asarray(preds,dtype=float)

def multi_horizon_backtest(y, models, horizon=12, origins=4, min_train=60):
    """Expanding-window recursive h-step evaluation. Each origin is scored at horizons 1..H."""
    y=y.asfreq("MS")
    last_origin=len(y)-horizon
    first_origin=max(min_train,last_origin-origins+1)
    rows=[]; failures=[]
    for pos in range(first_origin,last_origin+1):
        train=longest_contiguous(y.iloc[:pos])
        actual=y.iloc[pos:pos+horizon]
        if len(train)<min_train or len(actual)!=horizon or actual.isna().any():
            continue
        if train.empty or train.index[-1] != y.index[pos-1]:
            continue
        origin_date=y.index[pos-1]
        for model in models:
            try:
                path=multi_step_forecast(train,model,horizon)
                for h,pred in enumerate(path,1):
                    rows.append({"Model":model,"Origin":origin_date,"Target date":actual.index[h-1],"Horizon":h,"Actual":float(actual.iloc[h-1]),"Forecast":float(pred),"Error":float(pred)-float(actual.iloc[h-1])})
            except Exception as e:
                failures.append({"Model":model,"Origin":origin_date,"Reason":str(e)})
    return pd.DataFrame(rows),pd.DataFrame(failures)

with tabs[4]:
    st.subheader("Forecast target")
    forecast_target = st.selectbox(
        "Choose the series to calculate and forecast",
        ["CPI Index", "YoY Inflation (%)"],
        index=1,
        key="forecast_target_series",
        help="CPI Index forecasts the index level. YoY Inflation forecasts the percentage change from the same month one year earlier."
    )
    y=(long[(long["State/UT"]==state)&(long.Sector==sector)].sort_values("Date").set_index("Date")["CPI"].asfreq("MS"))
    yoy=(y/y.shift(12)-1)*100
    forecast_series = y if forecast_target == "CPI Index" else yoy
    series_unit = "CPI index points" if forecast_target == "CPI Index" else "percentage points"
    series_axis = "CPI index" if forecast_target == "CPI Index" else "YoY inflation (%)"
    st.caption(f"Selected forecast target: **{forecast_target}**. All validation, model comparison and forecast workflows below use this series.")
    if forecast_target == "YoY Inflation (%)":
        st.info("YoY inflation is calculated as 100 × (CPI[t] / CPI[t−12] − 1). The first 12 months have no YoY value because a 12-month comparison is unavailable.")
    else:
        st.info("CPI forecasting predicts the future index level directly; it does not forecast the inflation rate.")
    st.caption("Automatically evaluates candidate models on the selected State/UT and sector using recursive multi-step rolling-origin validation, selects the lowest-RMSE method on common forecast paths, and generates the future forecast.")
    st.caption("Research evaluation note: the bundled independent fixed-origin holdout covers June 2022–May 2023, trained only through May 2022. It is a separate benchmark result; changing uploaded data or model settings does not automatically rerun that archived experiment.")
    st.write("The app now restores all 12 candidate methods from the earlier research versions so they can be compared against the focused seven-model set under the same rolling-origin evaluation design.")
    st.info("Comparison groups: Naive and Seasonal Naive are benchmarks; Moving Average and Drift are simple statistical baselines; Holt Linear, Theta, ETS and SARIMA are statistical forecasting methods; Random Forest and Extra Trees are tree ensembles; Gradient Boosting and XGBoost are boosting methods. Restoring a model does not imply it is better. Retain models only when common-origin out-of-sample results support them.")
    st.info("For a fair experiment, keep the same State/UT, sector, forecast horizon, rolling-origin window and minimum training history when comparing methods. The leaderboard re-scores successful individual models on identical common forecast dates. Models that fail or have insufficient origins are reported rather than silently ranked.")
    st.info("For a fair ranking, compare the Forecast origins column. Models may have different successful origins; ensemble calculations use common dates. Prefer common-origin comparisons when reporting final research results.")
    st.markdown("#### Multi-step repeated-origin validation")
    st.caption("This experiment re-fits each selected model at each expanding-window origin, then recursively forecasts 1 to H months ahead. It evaluates the actual multi-step task, rather than selecting a 12-month forecast model only by one-step RMSE.")
    mh1,mh2,mh3=st.columns(3)
    mh_horizon=mh1.selectbox("Validation horizon",[3,6,12],index=2,key="mh_horizon")
    mh_origins=mh2.selectbox("Number of rolling origins",list(range(3,13)),index=5,key="mh_origins",help="More origins provide a more stable evaluation when enough consecutive observations are available.")
    mh_train=mh3.selectbox("Minimum training months",[48,60,72],index=1,key="mh_min_train")
    mh_models=st.multiselect("Models for multi-step test",MODEL_NAMES,default=["Naive","Drift","Holt Linear","Theta","ETS (Holt-Winters)","SARIMA","Random Forest","Extra Trees"],key="mh_models")
    if st.button("Run repeated-origin multi-step validation",key="run_mh_backtest",type="primary",use_container_width=True):
        if not mh_models: st.warning("Select at least one model.")
        else:
            with st.spinner("Running recursive multi-step validation; statistical and tree models are refitted at each forecast step..."):
                mh_results,mh_fail=multi_horizon_backtest(forecast_series,mh_models,mh_horizon,mh_origins,mh_train)
            if mh_results.empty:
                st.error("No valid multi-step forecast runs. Reduce minimum training months or horizon.")
            else:
                agg=mh_results.groupby(["Model","Horizon"]).agg(MAE=("Error",lambda x:float(np.mean(np.abs(x)))),RMSE=("Error",lambda x:float(np.sqrt(np.mean(np.square(x))))),Bias=("Error","mean"),N=("Error","size")).reset_index()
                overall=mh_results.groupby("Model").agg(MAE=("Error",lambda x:float(np.mean(np.abs(x)))),RMSE=("Error",lambda x:float(np.sqrt(np.mean(np.square(x))))),Bias=("Error","mean"),Predictions=("Error","size")).reset_index().sort_values("RMSE")
                st.session_state["export_multi_step_predictions"] = mh_results.copy()
                st.session_state["export_multi_step_metrics"] = agg.copy()
                st.session_state["multi_step_validation_snapshot"] = {"state":state,"sector":sector,"series":forecast_target,"horizon":mh_horizon,"origins":mh_origins,"min_train":mh_train,"models":list(mh_models),"predictions":mh_results.copy(),"metrics":agg.copy(),"overall":overall.copy(),"failures":mh_fail.copy()}

    saved_mh=st.session_state.get("multi_step_validation_snapshot")
    if saved_mh and saved_mh.get("state")==state and saved_mh.get("sector")==sector and saved_mh.get("series", "YoY Inflation (%)")==forecast_target:
        mh_results_saved=saved_mh["predictions"]; agg_saved=saved_mh["metrics"]; overall_saved=saved_mh["overall"]; mh_fail_saved=saved_mh["failures"]
        st.caption(f"Saved multi-step validation · {saved_mh['horizon']} months · {saved_mh['origins']} origins · {len(saved_mh['models'])} models")
        st.markdown(f"**Error by forecast horizon** ({series_unit}; N counts origin-target predictions)")
        st.dataframe(agg_saved.round(3),use_container_width=True,hide_index=True)
        fig=px.line(agg_saved,x="Horizon",y="RMSE",color="Model",markers=True,title="RMSE increases or changes with forecast horizon")
        fig.update_layout(height=420,xaxis_title="Months ahead",yaxis_title=f"RMSE ({series_unit})")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("**Overall across all evaluated horizons**")
        st.dataframe(overall_saved.round(3),use_container_width=True,hide_index=True)
        st.download_button("Download multi-step predictions",mh_results_saved.to_csv(index=False).encode(),"Repeated_Origin_Multi_Horizon_Predictions.csv","text/csv",key="persist_mh_predictions")
        st.download_button("Download horizon metrics",agg_saved.to_csv(index=False).encode(),"Repeated_Origin_Horizon_Metrics.csv","text/csv",key="persist_mh_metrics")
        if isinstance(mh_fail_saved,pd.DataFrame) and not mh_fail_saved.empty:
            st.warning("Some model-origin runs failed; inspect the failure log before interpreting comparisons.")
            st.dataframe(mh_fail_saved,use_container_width=True,hide_index=True)
            st.download_button("Download multi-step failure log",mh_fail_saved.to_csv(index=False).encode(),"Repeated_Origin_Failures.csv","text/csv",key="persist_mh_failures")
        st.info("Origins are overlapping expanding-window forecasts. Prediction rows are not independent; report the number of origins separately and avoid treating every horizon row as an independent sample.")
    st.divider()
    st.subheader("Automatic Evaluate and Forecast")
    horizon=st.slider("Forecast horizon used for evaluation and final forecast (months)",3,24,12)
    origins=st.slider("Rolling forecast origins (each origin predicts the full horizon)",3,12,8)
    min_train=st.select_slider("Minimum training history (months)",[48,60,72],value=72)
    models=st.multiselect("Candidate methods",MODEL_NAMES,default=MODEL_NAMES,
                         help="All 12 individual methods are selected by default for the controlled multi-step comparison.")
    include_ensembles=st.checkbox("Evaluate pairwise and multi-model combinations",value=True)
    recent=longest_contiguous(forecast_series)
    st.caption(f"Observed {forecast_target} points: {int(forecast_series.notna().sum())} · Longest consecutive recent run: {len(recent)} months.")
    st.info("Model selection is based on recursive multi-step errors across the selected forecast horizon. A 12-month forecast is therefore compared using 12-month-ahead paths, not only one-month-ahead errors. All candidates are scored on the same complete forecast origins and horizons.")
    if st.button("Run Automatic Evaluation and Forecast",type="primary",use_container_width=True):
        if not models: st.warning("Choose at least one candidate model.")
        elif len(recent)<min_train+horizon: st.error(f"This series needs at least {min_train+horizon} consecutive {forecast_target} observations for the selected training history and forecast horizon; it has {len(recent)}.")
        else:
            with st.spinner("Running recursive multi-step backtests for the selected methods..."):
                mh_results,mh_fail=multi_horizon_backtest(forecast_series,models,horizon,origins,min_train)
            if mh_results.empty:
                st.error("No candidate model produced a valid multi-step backtest. Reduce the horizon or minimum training history, or inspect the data continuity.")
            else:
                # Retain only complete origin × horizon paths, then compare every method on the same paths.
                model_frames={}
                for model_name in models:
                    part=mh_results[mh_results["Model"]==model_name].copy()
                    if part.empty: continue
                    complete_origins=part.groupby("Origin")["Horizon"].nunique()
                    complete_origins=complete_origins[complete_origins==horizon].index
                    part=part[part["Origin"].isin(complete_origins)].copy()
                    if not part.empty:
                        part=part.rename(columns={"Target date":"Date","Forecast":"Predicted"})
                        model_frames[model_name]=part.set_index(["Origin","Horizon"]).sort_index()[["Date","Actual","Predicted"]]
                if len(model_frames)<1:
                    st.error("No method completed a full forecast horizon. Try a shorter horizon or lower minimum training history.")
                else:
                    common_origins=set.intersection(*[set(frame.index.get_level_values("Origin")) for frame in model_frames.values()])
                    common_origins=sorted(common_origins)
                    if len(common_origins)<3:
                        st.error("Fewer than three common complete forecast origins are available across the successful methods. No best-model ranking is reported; try fewer candidate methods, a shorter horizon or a lower minimum training history.")
                    else:
                        common_index=pd.MultiIndex.from_product([common_origins,range(1,horizon+1)],names=["Origin","Horizon"])
                        model_frames={m:frame.loc[common_index] for m,frame in model_frames.items()}
                        first_model=next(iter(model_frames))
                        actual=model_frames[first_model]["Actual"]
                        score_rows=[]
                        aligned={m:frame["Predicted"] for m,frame in model_frames.items()}
                        scale_train=forecast_series.loc[forecast_series.index<common_origins[0]].dropna()
                        scale_common=float(np.mean(np.abs(np.diff(scale_train.values)))) if len(scale_train)>1 else np.nan
                        for model_name,pred in aligned.items():
                            err=pred-actual; mae=float(np.mean(np.abs(err)))
                            score_rows.append({"Model":model_name,"Type":"Individual","Origins":len(common_origins),"Forecast errors":len(common_index),
                                "MAE":mae,"RMSE":float(np.sqrt(np.mean(np.square(err)))),
                                "MASE":mae/scale_common if scale_common and np.isfinite(scale_common) else np.nan,"Bias":float(err.mean())})

                        ensemble_components={}
                        if include_ensembles and len(aligned)>=2:
                            for a,b in itertools.combinations(aligned,2):
                                name=f"Pairwise: {a} + {b}"; pred=(aligned[a]+aligned[b])/2.0
                                ensemble_components[name]=[a,b]; err=pred-actual; mae=float(np.mean(np.abs(err)))
                                score_rows.append({"Model":name,"Type":"Pairwise ensemble","Origins":len(common_origins),"Forecast errors":len(common_index),
                                    "MAE":mae,"RMSE":float(np.sqrt(np.mean(np.square(err)))),
                                    "MASE":mae/scale_common if scale_common and np.isfinite(scale_common) else np.nan,"Bias":float(err.mean())})
                            names=list(aligned); name="Ensemble: equal-weight all models"
                            pred=pd.concat([aligned[m] for m in names],axis=1).mean(axis=1)
                            ensemble_components[name]=names; err=pred-actual; mae=float(np.mean(np.abs(err)))
                            score_rows.append({"Model":name,"Type":"Multi-model ensemble","Origins":len(common_origins),"Forecast errors":len(common_index),
                                "MAE":mae,"RMSE":float(np.sqrt(np.mean(np.square(err)))),
                                "MASE":mae/scale_common if scale_common and np.isfinite(scale_common) else np.nan,"Bias":float(err.mean())})

                        score=pd.DataFrame(score_rows).sort_values("RMSE").reset_index(drop=True)
                        winner=str(score.iloc[0].Model)
                        winner_components=ensemble_components.get(winner)
                        if winner_components is None:
                            winner_pred=aligned[winner]
                        else:
                            winner_pred=pd.concat([aligned[m] for m in winner_components],axis=1).mean(axis=1)
                        abs_err=np.abs(actual-winner_pred)

                        train=longest_contiguous(forecast_series); fc=None
                        if len(train)>=min_train:
                            if winner_components is None:
                                hist=train.copy(); preds=[]
                                for _ in range(horizon):
                                    value=float(one_step(hist,winner)); preds.append(value)
                                    hist.loc[hist.index[-1]+pd.offsets.MonthBegin(1)]=value
                            else:
                                component_paths=[]
                                for member in winner_components:
                                    hist=train.copy(); path=[]
                                    for _ in range(horizon):
                                        value=float(one_step(hist,member)); path.append(value)
                                        hist.loc[hist.index[-1]+pd.offsets.MonthBegin(1)]=value
                                    component_paths.append(path)
                                preds=np.mean(np.asarray(component_paths,dtype=float),axis=0).tolist()
                            dates_fc=pd.date_range(train.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
                            q90=float(np.quantile(abs_err,.90))
                            fc=pd.DataFrame({"Date":dates_fc,"Forecast":preds,"Lower empirical band":np.asarray(preds)-q90,"Upper empirical band":np.asarray(preds)+q90})
                            fc["Forecast Target"]=forecast_target; fc["State/UT"]=state; fc["Sector"]=sector; fc["Selected Model"]=winner; fc["Forecast Horizon (months)"]=horizon; fc["Validation Origins"]=len(common_origins)
                            st.session_state["export_automatic_forecast"]=fc.copy()
                        else:
                            st.session_state.pop("export_automatic_forecast",None)

                        st.session_state["export_automatic_model_comparison"]=score.copy()
                        st.session_state["automatic_evaluation_snapshot"]={
                            "state":state,"sector":sector,"series":forecast_target,"horizon":horizon,"origins":len(common_origins),
                            "score":score.copy(),"backtests":{m:bt.copy() for m,bt in model_frames.items()},
                            "winner":winner,"winner_components":winner_components,"forecast":fc.copy() if fc is not None else None,
                            "min_train":min_train,"validation_method":"recursive multi-step common-origin evaluation",
                            "failures":mh_fail.copy() if isinstance(mh_fail,pd.DataFrame) else pd.DataFrame()}
                        if isinstance(mh_fail,pd.DataFrame) and not mh_fail.empty:
                            st.warning("Some model-origin runs failed. They were not included in the common-origin ranking; inspect the failure log.")
                            st.dataframe(mh_fail,use_container_width=True,hide_index=True)



    # Persist automatic evaluation output across Streamlit reruns (e.g. when another workflow is run).
    saved_auto = st.session_state.get("automatic_evaluation_snapshot")
    if saved_auto and saved_auto.get("state")==state and saved_auto.get("sector")==sector and saved_auto.get("series", "YoY Inflation (%)")==forecast_target:
        score_saved=saved_auto["score"]
        backtests_saved=saved_auto["backtests"]
        winner_saved=saved_auto["winner"]
        st.caption(f"Saved automatic result · {saved_auto.get('validation_method','multi-step validation')} · {saved_auto.get('horizon')} months · {saved_auto.get('origins')} origins")
        st.markdown("#### Individual models and combinations")
        st.dataframe(score_saved.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","MASE":"{:.3f}","Bias":"{:.3f}"}),use_container_width=True,hide_index=True)
        st.success(f"Selected: {winner_saved} · lowest {saved_auto.get('horizon')}-month multi-step RMSE {score_saved.iloc[0]['RMSE']:.3f} {series_unit}.")
        st.markdown("**How to interpret:** Lower MAE and RMSE mean smaller errors. MASE below 1 means the model beats the training-period naive scale. Positive bias means forecasts were too high on average; negative bias means they were too low. All candidates were compared on the same complete forecast origins and horizons.")
        winner_components=saved_auto.get("winner_components")
        if backtests_saved:
            first_saved=next(iter(backtests_saved.values()))
            latest_origin=max(first_saved.index.get_level_values("Origin"))
            if winner_components is None:
                validation_saved=backtests_saved[winner_saved].loc[latest_origin]
                validation_pred=validation_saved["Predicted"].to_numpy()
            else:
                parts=[backtests_saved[m].loc[latest_origin,"Predicted"].to_numpy() for m in winner_components]
                validation_pred=np.mean(np.vstack(parts),axis=0)
            validation_dates=first_saved.loc[latest_origin,"Date"].tolist()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=forecast_series.index,y=forecast_series.values,name=f"Historical {forecast_target}"))
            fig.add_trace(go.Scatter(x=validation_dates,y=validation_pred,name=f"Validation path from {latest_origin:%b %Y}",mode="lines+markers"))
            st.plotly_chart(fig,use_container_width=True)
        forecast_saved=saved_auto.get("forecast")
        if isinstance(forecast_saved,pd.DataFrame) and not forecast_saved.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=forecast_series.index,y=forecast_series.values,name=f"Historical {forecast_target}"))
            fig.add_trace(go.Scatter(x=forecast_saved.Date,y=forecast_saved.Forecast,name="Forecast",mode="lines+markers"))
            fig.add_trace(go.Scatter(x=forecast_saved.Date,y=forecast_saved["Upper empirical band"],line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_saved.Date,y=forecast_saved["Lower empirical band"],fill="tonexty",line=dict(width=0),name="Empirical 90th-percentile absolute-error envelope"))
            fig.update_layout(height=420,title=f"{winner_saved} forecast · {forecast_target} · {state} · {sector}",xaxis_title="",yaxis_title=series_axis)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(forecast_saved,use_container_width=True,hide_index=True)
            st.download_button("Download forecast CSV",forecast_saved.to_csv(index=False).encode(),f"forecast_{forecast_target.split()[0]}_{state}_{sector}.csv","text/csv",key="persistent_auto_forecast_download")
            st.info(f"The empirical envelope is based on absolute errors pooled across the selected multi-step validation horizons, in {series_unit}. It is descriptive, not a formally calibrated prediction interval; test coverage on a separate untouched holdout before making interval-coverage claims.")
        else:
            st.warning(f"Automatic model comparison is saved, but a future forecast could not be generated with the current minimum training requirement ({saved_auto.get('min_train')} months).")


    st.divider()
    st.subheader("Manual evaluation and forecasting")
    st.caption("Choose one method yourself. This runs its rolling-origin evaluation and then generates a recursive forecast using that same method. It does not change the automatic model selection above.")
    manual_model=st.selectbox("Select a model to evaluate and forecast",MODEL_NAMES,key="manual_forecast_model")
    if st.button("Evaluate selected model and forecast",key="manual_eval_forecast",use_container_width=True):
        try:
            manual_bt=rolling_eval(forecast_series,manual_model,origins=origins,min_train=min_train)
            manual_mae=float(mean_absolute_error(manual_bt["Actual"],manual_bt["Predicted"]))
            manual_rmse=float(np.sqrt(mean_squared_error(manual_bt["Actual"],manual_bt["Predicted"])))
            manual_bias=float((manual_bt["Predicted"]-manual_bt["Actual"]).mean())
            manual_scale_train=longest_contiguous(forecast_series.loc[forecast_series.index < manual_bt.index.min()])
            manual_scale=float(np.mean(np.abs(np.diff(manual_scale_train.to_numpy(dtype=float))))) if len(manual_scale_train)>1 else np.nan
            manual_mase=manual_mae/manual_scale if manual_scale and np.isfinite(manual_scale) else np.nan
            manual_history=longest_contiguous(forecast_series)
            manual_fc=None
            manual_error=None
            if len(manual_history)<min_train:
                manual_error=f"Need at least {min_train} consecutive observations. Available: {len(manual_history)}."
            else:
                hist=manual_history.copy(); manual_preds=[]
                for _ in range(horizon):
                    value=float(one_step(hist,manual_model)); manual_preds.append(value)
                    hist.loc[hist.index[-1]+pd.offsets.MonthBegin(1)]=value
                manual_dates=pd.date_range(manual_history.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
                manual_abs_err=np.abs(manual_bt["Actual"]-manual_bt["Predicted"])
                manual_q90=float(np.quantile(manual_abs_err,0.90))
                manual_fc=pd.DataFrame({"Date":manual_dates,"Forecast":manual_preds,"Lower empirical band":np.array(manual_preds)-manual_q90,"Upper empirical band":np.array(manual_preds)+manual_q90})
                manual_fc["Forecast Target"]=forecast_target; manual_fc["State/UT"]=state; manual_fc["Sector"]=sector; manual_fc["Selected Model"]=manual_model; manual_fc["Forecast Horizon (months)"]=horizon; manual_fc["Validation Origins"]=len(manual_bt)
                st.session_state["export_manual_forecast"]=manual_fc.copy()
            st.session_state["manual_evaluation_snapshot"]={"model":manual_model,"state":state,"sector":sector,"series":forecast_target,"horizon":horizon,"origins":len(manual_bt),"mae":manual_mae,"rmse":manual_rmse,"bias":manual_bias,"mase":manual_mase,"backtest":manual_bt.copy(),"forecast":manual_fc.copy() if manual_fc is not None else None,"error":manual_error,"min_train":min_train}
            st.session_state["manual_cpi_result"]={"model":manual_model,"state":state,"sector":sector,"horizon":horizon,"origins":len(manual_bt),"mae":manual_mae,"rmse":manual_rmse,"bias":manual_bias,"mase":manual_mase,"backtest":manual_bt.copy()}
        except Exception as e:
            st.error(f"Manual evaluation/forecast failed for {manual_model}: {e}")

    saved_manual=st.session_state.get("manual_evaluation_snapshot")
    if saved_manual and saved_manual.get("state")==state and saved_manual.get("sector")==sector and saved_manual.get("series", "YoY Inflation (%)")==forecast_target:
        manual_bt_saved=saved_manual["backtest"]; manual_fc_saved=saved_manual.get("forecast")
        st.caption(f"Saved manual result · {saved_manual['model']} · {saved_manual['origins']} rolling origins")
        st.success(f"Manual result: {saved_manual['model']} · {saved_manual['origins']} rolling origins · RMSE {saved_manual['rmse']:.3f} {series_unit}.")
        m1,m2,m3,m4=st.columns(4)
        m1.metric(f"MAE ({series_unit})",f"{saved_manual['mae']:.3f}"); m2.metric(f"RMSE ({series_unit})",f"{saved_manual['rmse']:.3f}"); m3.metric(f"MASE",f"{saved_manual.get('mase',np.nan):.3f}" if np.isfinite(saved_manual.get("mase",np.nan)) else "—"); m4.metric(f"Bias ({series_unit})",f"{saved_manual['bias']:.3f}")
        backfig=go.Figure(); backfig.add_trace(go.Scatter(x=manual_bt_saved.index,y=manual_bt_saved["Actual"],name="Actual")); backfig.add_trace(go.Scatter(x=manual_bt_saved.index,y=manual_bt_saved["Predicted"],name="One-step predictions"))
        backfig.update_layout(height=360,title=f"{saved_manual['model']} · rolling-origin evaluation",yaxis_title=series_axis)
        st.plotly_chart(backfig,use_container_width=True)
        st.markdown("#### Residual diagnostics for the selected model")
        residuals=(manual_bt_saved["Actual"]-manual_bt_saved["Predicted"]).dropna()
        st.caption("Residual = actual − one-step-ahead prediction. These diagnostics use the rolling-origin evaluation errors, not in-sample fitted residuals.")
        if len(residuals)>=5:
            rfig=go.Figure(); rfig.add_trace(go.Scatter(x=residuals.index,y=residuals.values,mode="lines+markers",name="Residual")); rfig.add_hline(y=0,line_dash="dash")
            rfig.update_layout(height=300,title="Rolling-origin forecast residuals",xaxis_title="Date",yaxis_title=series_unit)
            st.plotly_chart(rfig,use_container_width=True)
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                from statsmodels.graphics.tsaplots import plot_acf
                import matplotlib.pyplot as plt
                max_lag=min(12,max(1,len(residuals)//5))
                lb_res=acorr_ljungbox(residuals,lags=[max_lag],return_df=True)
                st.dataframe(lb_res.rename_axis("Lag").reset_index().round(4),use_container_width=True,hide_index=True)
                st.caption("Ljung–Box null: no residual autocorrelation up to the displayed lag. A small p-value indicates remaining serial dependence; interpret cautiously because rolling-origin errors overlap in time.")
                if len(residuals)>10:
                    fig,ax=plt.subplots(figsize=(9,2.8)); plot_acf(residuals,lags=min(12,len(residuals)//2-1),ax=ax); st.pyplot(fig); plt.close(fig)
            except Exception as diag_exc:
                st.info(f"Residual autocorrelation diagnostics unavailable: {diag_exc}")
        else:
            st.info("At least five successful rolling-origin errors are needed for residual diagnostics.")
        if isinstance(manual_fc_saved,pd.DataFrame) and not manual_fc_saved.empty:
            fcfig=go.Figure(); fcfig.add_trace(go.Scatter(x=forecast_series.index,y=forecast_series.values,name=f"Historical {forecast_target}")); fcfig.add_trace(go.Scatter(x=manual_fc_saved["Date"],y=manual_fc_saved["Forecast"],name=f"{saved_manual['model']} forecast")); fcfig.add_trace(go.Scatter(x=manual_fc_saved["Date"],y=manual_fc_saved["Upper empirical band"],line=dict(width=0),showlegend=False)); fcfig.add_trace(go.Scatter(x=manual_fc_saved["Date"],y=manual_fc_saved["Lower empirical band"],fill="tonexty",line=dict(width=0),name="Empirical 90th-percentile absolute-error envelope"))
            fcfig.update_layout(height=390,title=f"Manual {saved_manual['model']} forecast · {forecast_target} · {state} · {sector}",yaxis_title=series_axis)
            st.plotly_chart(fcfig,use_container_width=True)
            st.dataframe(manual_fc_saved,use_container_width=True,hide_index=True)
            st.download_button("Download manual forecast CSV",manual_fc_saved.to_csv(index=False).encode(),f"manual_forecast_{forecast_target.split()[0]}_{state}_{sector}_{saved_manual['model'].replace(' ','_')}.csv","text/csv",key="download_manual_forecast")
            st.info(f"Manual mode forecasts {forecast_target} with the selected model regardless of its ranking. Empirical bands are descriptive historical-error bands in {series_unit}, not calibrated prediction intervals.")
        elif saved_manual.get("error"):
            st.error(saved_manual["error"])

with tabs[1]:
    st.subheader("Data quality, active dataset and downloads")
    st.markdown("#### Active dataset manager")
    st.write(f"**Currently active:** {st.session_state.active_dataset_label}")
    st.caption("All analytical tabs, including EDA, state comparisons, forecasting and live evaluation, read from this active dataset. The original bundled CSV is never overwritten.")
    if st.button("Switch back to current combined / original dataset", key="reset_active_dataset", use_container_width=False):
        st.session_state.active_data = None
        st.session_state.active_dataset_label = "Original dataset" if not st.session_state.upload_name else "Combined dataset"
        st.rerun()
    miss=data[states].isna().sum().sort_values(ascending=False)
    a,b,c=st.columns(3)
    a.metric("Possible CPI cells",f"{len(data)*len(states):,}")
    b.metric("Observed CPI cells",f"{available_cpi:,}")
    c.metric("Missing CPI cells",f"{missing_total:,}",help="Individual blank CPI entries across State/UT × month × sector.")
    miss_df=miss[miss>0].rename("Missing cells").reset_index().rename(columns={"index":"State/UT"})
    if miss_df.empty: st.success("No blank CPI cells in the current combined dataset.")
    else: st.dataframe(miss_df,use_container_width=True,hide_index=True)
    # Calendar gap report by sector
    st.markdown("#### Missing calendar months by sector")
    all_months=pd.date_range(data.Date.min(),data.Date.max(),freq="MS")
    gap_rows=[]
    for sec in sectors:
        present=set(data.loc[data.Sector==sec,"Date"])
        gaps=sorted(set(all_months)-present)
        gap_rows.append({"Sector":sec,"Missing month rows":len(gaps),"Months":", ".join(pd.Timestamp(x).strftime("%b %Y") for x in gaps[:20])})
    st.dataframe(pd.DataFrame(gap_rows),use_container_width=True,hide_index=True)
    if not st.session_state.merge_log.empty:
        st.markdown("#### Conflicting uploaded values")
        st.dataframe(st.session_state.merge_log,use_container_width=True,hide_index=True)
        st.download_button("Download merge conflict log",st.session_state.merge_log.to_csv(index=False).encode(),"CPI_Merge_Conflict_Log.csv","text/csv")
    else:
        st.caption("No conflicting uploaded cells have been logged in this session.")
    export=data.copy()
    export["Year"]=export.Date.dt.year
    export["Name"]=export.Date.dt.month.map(MONTH_NAMES)
    export=export[["Sector","Year","Name"]+states]
    st.download_button("Download combined wide dataset (CSV)",export.to_csv(index=False).encode(),"India_CPI_Combined_Dataset.csv","text/csv")
    st.download_button("Download long dataset with YoY inflation (CSV)",long.to_csv(index=False).encode(),"India_CPI_Long_YoY.csv","text/csv")
    st.markdown("#### Optional short-gap imputation for analysis")
    st.caption("The source/combined dataset above remains unchanged. This creates a separate analytical copy and records which cells were imputed. Only short internal gaps are interpolated; leading/trailing gaps are never extrapolated.")
    impute_method=st.selectbox("Imputation method",["No imputation (recommended source view)","Time interpolation for internal gaps up to 2 months","Time interpolation for internal gaps up to 3 months"])
    if impute_method!="No imputation (recommended source view)":
        max_gap=2 if "2 months" in impute_method else 3
        filled=data.copy()
        audit=[]
        for sec in sectors:
            sec_idx=filled.index[filled.Sector==sec]
            for state_col in states:
                original=filled.loc[sec_idx].set_index("Date")[state_col].sort_index().asfreq("MS")
                interp=original.interpolate(method="time",limit=max_gap,limit_area="inside")
                changed=original.isna() & interp.notna()
                for dt in original.index[changed]:
                    audit.append({"Date":dt,"Sector":sec,"State/UT":state_col,"Original CPI":np.nan,"Imputed CPI":float(interp.loc[dt]),"Method":f"Time interpolation, max {max_gap} consecutive months"})
                filled.loc[filled.Sector==sec,state_col]=filled.loc[filled.Sector==sec,"Date"].map(interp)
        audit_df=pd.DataFrame(audit)
        st.write(f"Prepared {len(audit_df):,} eligible imputed CPI cells in a separate analytical copy. Nothing is activated until you confirm.")
        if not audit_df.empty:
            st.dataframe(audit_df,use_container_width=True,hide_index=True)
            filled_export=filled.copy()
            filled_export["Year"]=filled_export.Date.dt.year
            filled_export["Name"]=filled_export.Date.dt.month.map(MONTH_NAMES)
            st.download_button("Download prepared cleaned dataset",filled_export[["Sector","Year","Name"]+states].to_csv(index=False).encode(),"India_CPI_Analytical_Imputed.csv","text/csv")
            st.download_button("Download imputation audit log",audit_df.to_csv(index=False).encode(),"India_CPI_Imputation_Audit.csv","text/csv")
            if st.button("Apply cleaned dataset and set as active",type="primary",key="apply_cleaned_dataset"):
                st.session_state.active_data=filled.copy()
                st.session_state.active_dataset_label=f"Cleaned dataset · time interpolation up to {max_gap} months · {len(audit_df)} cells changed"
                st.rerun()
        else:
            st.info("No eligible internal gaps were filled by the selected method; active dataset unchanged.")
    st.markdown("#### Existing imputation provenance")
    st.caption("The supplied cleaned dataset contains some values imputed during earlier data preparation. Use the audit file to identify these observations; original source values are not overwritten by this website.")
    audit_file = Path(__file__).parent / "CPI_Imputation_Audit_28_States.csv"
    if audit_file.exists():
        st.download_button("Download retained-state imputation audit", audit_file.read_bytes(),
                           "CPI_Imputation_Audit_28_States.csv", "text/csv")
    st.markdown("#### Methodological safeguards")
    st.markdown("""
    - The bundled source dataset is retained as the starting point.
    - Uploaded files are merged only after an explicit base-year compatibility confirmation.
    - Uploaded non-missing values or existing non-missing values are selected according to the user's merge rule.
    - Conflicting cells are recorded in a downloadable log.
    - Missing CPI values remain missing; the app does not convert them to zero or automatically impute them.
    - YoY inflation is calculated as `100 × (CPI[t] / CPI[t-12] − 1)` with the monthly calendar preserved.
    - National overview averages are unweighted State/UT averages, not official all-India CPI.
    """)


with tabs[5]:
    st.subheader("Model Evaluation Guidance")
    st.markdown("""
    This section explains how to interpret the forecasting evaluation outputs. It does not display archived research results or static model rankings.

    #### Evaluation metrics

    - **MAE (Mean Absolute Error):** average absolute difference between forecast and actual values, measured in the selected target’s units. Lower is better.
    - **RMSE (Root Mean Squared Error):** square root of the average squared error in the selected target’s units. It penalises large forecast misses more heavily. Lower is better.
    - **MASE (Mean Absolute Scaled Error):** scales the model's MAE against a naïve benchmark. Values below 1 indicate lower error than the reference naïve scale, provided the same scaling definition is used.
    - **Bias:** average forecast minus actual value. Positive bias indicates over-forecasting; negative bias indicates under-forecasting.

    #### Validation methods

    - **Rolling-origin evaluation:** repeatedly fits a model using observations available up to each origin and forecasts the next observation. It estimates one-step-ahead performance.
    - **Multi-step repeated-origin validation:** at each origin, forecasts several months ahead recursively. This is more relevant when the intended use is a multi-month forecast.
    - **Fixed-origin holdout:** fits once before a reserved test period and forecasts the entire period without updating with actual test observations.

    #### Research protocol

    1. Compare models on identical forecast origins and the same State/UT, sector and evaluation window wherever possible.
    2. Match the validation horizon to the forecast horizon of interest. A model that performs best one month ahead may not be best 12 months ahead.
    3. Keep the final test period untouched during model selection.
    4. Report model failures, number of successful origins and series eligibility.
    5. Report results separately by State/UT and sector; do not assume one model is best everywhere.
    6. Treat overlapping repeated-origin forecast errors as dependent observations when interpreting statistical comparisons.

    #### Prediction uncertainty

    - An empirical error band is not automatically a calibrated prediction interval.
    - Before reporting 80%, 90% or 95% prediction intervals, assess empirical coverage and average interval width on data not used to calibrate the intervals.
    - Report the calibration method, nominal level, observed coverage, interval width and evaluation sample size.

    #### Important interpretation notes

    - Forecast errors are measured in the units of the selected target: CPI index points for CPI forecasts or percentage points for YoY forecasts.
    - A lower RMSE does not guarantee lower MAE or lower bias.
    - Results from one fixed holdout period should not be described as repeated-origin validation.
    - The dataset currently ends in May 2023; forecasts beyond this date cannot be verified against actual values using this dataset alone.
    """)
st.divider()
st.caption("India CPI Research Intelligence · Research prototype · Verify source, base year and official CPI definitions before publication.")


def build_complete_workbook():
    """Create one Excel workbook containing the active data, analysis tables, saved forecast outputs and charts."""
    from io import BytesIO
    from openpyxl import load_workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    output = BytesIO()
    selected = long[(long["State/UT"] == state) & (long["Sector"] == sector)].sort_values("Date").copy()
    selected_yoy = selected.set_index("Date")["Inflation_YoY"]
    selected_cpi = selected.set_index("Date")["CPI"]
    overview = long[long["Sector"] == sector].groupby("Date", as_index=False).agg(
        Mean_CPI=("CPI", "mean"), Mean_YoY_Inflation=("Inflation_YoY", "mean"),
        States_Available=("CPI", "count")
    )
    eda_stats = selected_cpi.describe().rename("CPI index").to_frame().reset_index().rename(columns={"index":"Statistic"})
    yoy_stats = selected_yoy.describe().rename("YoY inflation (%)").to_frame().reset_index().rename(columns={"index":"Statistic"})
    diagnostic_rows = []
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        yy = selected_yoy.dropna()
        if len(yy) >= 36:
            diagnostic_rows.append({"Test":"ADF","Statistic":float(adfuller(yy,autolag="AIC")[0]),"p-value":float(adfuller(yy,autolag="AIC")[1]),"Null hypothesis":"Unit root / non-stationarity"})
            kp = kpss(yy, regression="c", nlags="auto")
            diagnostic_rows.append({"Test":"KPSS","Statistic":float(kp[0]),"p-value":float(kp[1]),"Null hypothesis":"Level stationarity"})
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(yy,lags=[6,12,18,24],return_df=True)
            for lag,row in lb.iterrows():
                diagnostic_rows.append({"Test":f"Ljung-Box lag {lag}","Statistic":float(row["lb_stat"]),"p-value":float(row["lb_pvalue"]),"Null hypothesis":"No autocorrelation up to tested lag"})
    except Exception as exc:
        diagnostic_rows.append({"Test":"Diagnostic calculation","Statistic":None,"p-value":None,"Null hypothesis":f"Unavailable: {exc}"})
    diagnostics = pd.DataFrame(diagnostic_rows)
    seasonality = selected.dropna(subset=["Inflation_YoY"]).assign(
        Month=lambda frame: frame["Date"].dt.month
    ).groupby("Month")["Inflation_YoY"].agg(["count","mean","median","std"]).reset_index()
    seasonality["Month_Name"] = seasonality["Month"].map(MONTH_NAMES)
    rolling = pd.DataFrame({
        "Date": selected_cpi.index,
        "CPI": selected_cpi.values,
        "YoY_Inflation": selected_yoy.values,
        "YoY_Rolling_Mean_12M": selected_yoy.rolling(12).mean().values,
        "YoY_Rolling_SD_12M": selected_yoy.rolling(12).std().values,
    })
    state_latest = long[(long["Sector"] == sector)].sort_values("Date").groupby("State/UT").tail(1)[
        ["State/UT","Date","CPI","Inflation_YoY"]
    ].sort_values("State/UT")
    sector_compare = long[long["State/UT"] == state].sort_values("Date")[
        ["Date","Sector","CPI","Inflation_YoY"]
    ]

    tables = {
        "Active_CPI_Data": data.sort_values(["Sector","Date"]),
        "Long_YoY_Data": long,
        "Overview_Summary": overview,
        "Selected_Series": selected[["Date","State/UT","Sector","CPI","Inflation_YoY"]],
        "State_Comparison": state_latest,
        "Sector_Comparison": sector_compare,
        "EDA_CPI_Statistics": eda_stats,
        "EDA_YoY_Statistics": yoy_stats,
        "EDA_Monthly_Seasonality": seasonality,
        "EDA_Rolling_Statistics": rolling,
        "EDA_Diagnostics": diagnostics,
    }
    saved_sheets = [
        ("Automatic_Model_Comparison","export_automatic_model_comparison"),
        ("Automatic_Forecast","export_automatic_forecast"),
        ("Manual_Forecast","export_manual_forecast"),
        ("MultiStep_Predictions","export_multi_step_predictions"),
        ("MultiStep_Metrics","export_multi_step_metrics"),
    ]
    current_context = {"State/UT": state, "Sector": sector, "Forecast Target": forecast_target}
    for sheet,key in saved_sheets:
        value = st.session_state.get(key)
        if isinstance(value,pd.DataFrame) and not value.empty:
            # Forecast exports carry explicit context. Do not include a previous target's results.
            if sheet in ("Automatic_Forecast", "Manual_Forecast"):
                if "Forecast Target" not in value.columns or not (value["Forecast Target"] == forecast_target).all():
                    continue
                if "State/UT" in value.columns and not (value["State/UT"] == state).all(): continue
                if "Sector" in value.columns and not (value["Sector"] == sector).all(): continue
            elif sheet in ("Automatic_Model_Comparison", "MultiStep_Predictions", "MultiStep_Metrics"):
                snapshot_key = {"Automatic_Model_Comparison":"automatic_evaluation_snapshot", "MultiStep_Predictions":"multi_step_validation_snapshot", "MultiStep_Metrics":"multi_step_validation_snapshot"}[sheet]
                snap = st.session_state.get(snapshot_key, {})
                if snap.get("state") != state or snap.get("sector") != sector or snap.get("series") != forecast_target:
                    continue
            tables[sheet] = value

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)

    output.seek(0)
    wb = load_workbook(output)
    charts_ws = wb.create_sheet("Charts")
    charts_ws["A1"] = "India CPI Research Intelligence — Charts"
    charts_ws["A1"].font = Font(bold=True, size=15)
    charts_ws["A2"] = f"Active dataset: {st.session_state.active_dataset_label}"
    charts_ws["A3"] = f"Selected State/UT: {state} | Sector: {sector}"
    charts_ws["A4"] = "Charts are generated from the active dataset and saved forecast outputs included in this workbook."

    def add_line_chart(sheet_name, title, y_col, anchor, y_title):
        if sheet_name not in wb.sheetnames:
            return
        ws = wb[sheet_name]
        if ws.max_row < 3:
            return
        chart = LineChart()
        chart.title = title
        chart.style = 13
        chart.y_axis.title = y_title
        chart.x_axis.title = "Date"
        data_ref = Reference(ws, min_col=y_col, min_row=1, max_row=ws.max_row)
        cats_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.height = 8
        chart.width = 16
        charts_ws.add_chart(chart, anchor)

    add_line_chart("Overview_Summary","Unweighted mean State/UT CPI index",2,"A6","CPI index")
    add_line_chart("Overview_Summary","Unweighted mean State/UT YoY inflation",3,"J6","Inflation (%)")
    add_line_chart("Selected_Series","Selected CPI index",4,"A22","CPI index")
    add_line_chart("Selected_Series","Selected YoY inflation",5,"J22","Inflation (%)")
    add_line_chart("EDA_Rolling_Statistics","Rolling mean YoY inflation",4,"A38","Inflation (%)")
    add_line_chart("EDA_Rolling_Statistics","Rolling volatility (12-month SD)",5,"J38","Standard deviation")
    if "Automatic_Forecast" in wb.sheetnames:
        add_line_chart("Automatic_Forecast",f"Automatic {forecast_target} forecast",2,"A54",series_axis)
    if "Manual_Forecast" in wb.sheetnames:
        add_line_chart("Manual_Forecast",f"Manual {forecast_target} forecast",2,"J54",series_axis)
    if "MultiStep_Metrics" in wb.sheetnames:
        ws = wb["MultiStep_Metrics"]
        # A compact horizon-metric table is included in the sheet; a chart can be built when its columns are present.
        if ws.max_row > 2 and "Horizon" in [c.value for c in ws[1]]:
            headers = [c.value for c in ws[1]]
            if "RMSE" in headers:
                xcol = headers.index("Horizon")+1
                ycol = headers.index("RMSE")+1
                chart = LineChart()
                chart.title = "Multi-step RMSE by horizon"
                chart.y_axis.title = f"RMSE ({series_unit})"
                chart.x_axis.title = "Months ahead"
                chart.add_data(Reference(ws,min_col=ycol,min_row=1,max_row=ws.max_row),titles_from_data=True)
                chart.set_categories(Reference(ws,min_col=xcol,min_row=2,max_row=ws.max_row))
                chart.height=8; chart.width=16
                charts_ws.add_chart(chart,"A38")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions if ws.max_row > 1 else "A1"
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="245B78")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for col_cells in ws.columns:
            col_letter = get_column_letter(col_cells[0].column)
            max_len = min(max((len(str(c.value)) if c.value is not None else 0) for c in col_cells[:250]), 42)
            ws.column_dimensions[col_letter].width = max(12, max_len + 2)
    output2 = BytesIO()
    wb.save(output2)
    return output2.getvalue()

st.divider()
st.subheader("Download complete analysis workbook")
st.caption("One Excel file containing the active CPI dataset, long-format YoY data, selected-series tables, EDA summaries and diagnostics, saved forecasting/evaluation outputs from this session, and Excel charts. Forecast sheets appear after you run the corresponding forecasting or validation procedure.")
try:
    complete_workbook = build_complete_workbook()
    st.download_button(
        "Download all outputs, tables and graphs (Excel workbook)",
        data=complete_workbook,
        file_name="India_CPI_Complete_Analysis_and_Outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True
    )
except Exception as export_error:
    st.error(f"Could not prepare the complete analysis workbook: {export_error}")
