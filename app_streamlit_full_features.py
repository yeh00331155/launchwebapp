
# app_streamlit_full_features.py
# -----------------------------------------------------------
# 雲林 A1/A2 嚴重度預測（完整欄位輸入；與訓練特徵一致）
# 需求：同目錄內需有
#   - 113雲林0924.xlsx（sheet: Yulin，用於提供下拉選單選項）
#   - tfidf.joblib, scaler.joblib, kmeans.joblib, catboost_model.cbm, model_meta.json
#
# 使用：
#   pip install streamlit pandas numpy scikit-learn catboost joblib openpyxl
#   streamlit run app_streamlit_full_features.py
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from datetime import datetime, date, time
from catboost import CatBoostClassifier, Pool

st.set_page_config(page_title="雲林 A1/A2 嚴重度預測（完整欄位）", layout="wide")

EXCEL_PATH = "113雲林0924.xlsx"
SHEET_NAME = "Yulin"

# ===== 載入模型產物 =====
@st.cache_resource
def load_artifacts():
    tfv = joblib.load("tfidf.joblib")
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans.joblib")
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    with open("model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return tfv, scaler, kmeans, model, meta

# ===== 載入 Excel 以提供下拉選項（不參與預測） =====
@st.cache_data
def load_choices():
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, dtype={'Date': str, 'Time': str})
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.warning(f"⚠️ 無法讀取 {EXCEL_PATH}：{e}\n將以空的選項清單呈現（可使用自訂輸入）。")
        df = pd.DataFrame()

    def choices(col):
        if col not in df.columns:
            return []
        vals = df[col].dropna().astype(str)
        vals = vals[vals.str.len()>0]
        return sorted(vals.unique().tolist())

    return {
        'Location': choices('Location'),
        'Weather': choices('Weather'),
        'Lighting': choices('Lighting'),
        'Road': choices('Road'),
        'Ground Pavemen': choices('Ground Pavemen'),
        'Pavement Defect': choices('Pavement Defect'),
        'Road Obstacles': choices('Road Obstacles'),
        'Sight Distance Quality': choices('Sight Distance Quality'),
        'Traffic Signal': choices('Traffic Signal'),
        'Lane Separation Facility': choices('Lane Separation Facility'),
        'Accident Type': choices('Accident Type'),
        'Crash Manner': choices('Crash Manner'),
        'Road Condition': choices('Road Condition'),
    }

TFV, SCALER, KMEANS, MODEL, META = load_artifacts()
CHOICES = load_choices()

# ===== 小工具：選單 + 自訂 =====
def select_with_custom(label, options):
    # options: list[str]
    opts = ['（自訂輸入）'] + (options or [])
    pick = st.selectbox(label, opts, index=1 if len(opts)>1 else 0, key=f"sb_{label}")
    if pick == '（自訂輸入）':
        return st.text_input(f"{label}（自訂）")
    return pick

# ===== UI =====
st.title("雲林 A1/A2 嚴重度預測（完整欄位輸入）")

with st.sidebar:
    st.header("基本（數值/座標/時間）")
    d = st.date_input("Date（事故日期）", value=date(2024,1,1))
    t = st.time_input("Time（事故時間）", value=time(8,0))

    sl = st.number_input("Speed Limit（速限, km/h）", min_value=0.0, max_value=130.0, value=50.0, step=5.0)
    lng = st.number_input("Longitude（經度）", value=120.431, step=0.001, format="%.6f")
    lat = st.number_input("Latitude（緯度）", value=23.708, step=0.001, format="%.6f")
    road_cond = select_with_custom("Road Condition", CHOICES.get('Road Condition'))

st.subheader("類別欄位（輸入將直接送入模型）")
col1, col2, col3 = st.columns(3)
with col1:
    location = select_with_custom("Location", CHOICES.get('Location'))
    weather  = select_with_custom("Weather", CHOICES.get('Weather'))
    lighting = select_with_custom("Lighting", CHOICES.get('Lighting'))
    road     = select_with_custom("Road", CHOICES.get('Road'))
with col2:
    ground_pav = select_with_custom("Ground Pavemen", CHOICES.get('Ground Pavemen'))
    pave_def   = select_with_custom("Pavement Defect", CHOICES.get('Pavement Defect'))
    obstacles  = select_with_custom("Road Obstacles", CHOICES.get('Road Obstacles'))
    sight_q    = select_with_custom("Sight Distance Quality", CHOICES.get('Sight Distance Quality'))
with col3:
    signal     = select_with_custom("Traffic Signal", CHOICES.get('Traffic Signal'))
    lane_sep   = select_with_custom("Lane Separation Facility", CHOICES.get('Lane Separation Facility'))
    acc_type   = select_with_custom("Accident Type", CHOICES.get('Accident Type'))
    crash_man  = select_with_custom("Crash Manner", CHOICES.get('Crash Manner'))

st.subheader("Cause Analysis（文字描述，可空白）")
cause = st.text_area("Cause Analysis", height=100)

run = st.button("計算 A1/A2 機率", type="primary")

# ===== 推論：特徵工程（需與訓練一致） =====
ROAD_MAP = {'冰雪':1,'泥濘':2,'油滑':3,'濕潤':4,'乾燥':5}

def featurize_from_inputs():
    # 時間特徵
    Hour = int(t.hour)
    Month = int(d.month)
    Weekday = int(d.weekday())
    is_weekend = int(Weekday >= 5)
    is_peak = int(Hour in [7,8,9,17,18,19])

    # 數值與座標
    SpeedLimit = float(sl)
    lon_r = round(float(lng), 5)
    lat_r = round(float(lat), 5)

    # RoadCond_Ord
    RoadCond_Ord = int(ROAD_MAP.get(str(road_cond), 3))

    # geo_cell 與 KMeans 分群
    geo_cell = f"{round(lon_r,3)}_{round(lat_r,3)}"
    cluster_base_cols = ['Hour','Month','Weekday','Speed Limit','RoadCond_Ord','lon_r','lat_r']
    base_df = pd.DataFrame([[Hour,Month,Weekday,SpeedLimit,RoadCond_Ord,lon_r,lat_r]],
                           columns=cluster_base_cols)
    Z = SCALER.transform(base_df.fillna(0))
    accident_cluster = str(KMEANS.predict(Z)[0])

    # 數值特徵（順序照 meta['num_feats']）
    num_feats = META['num_feats']
    num_vals = {
        'Hour':Hour, 'Month':Month, 'Weekday':Weekday, 'Speed Limit':SpeedLimit,
        'RoadCond_Ord':RoadCond_Ord, 'is_weekend':is_weekend, 'is_peak':is_peak,
        'lon_r':lon_r, 'lat_r':lat_r
    }
    X_num = pd.DataFrame([[num_vals[c] for c in num_feats]], columns=num_feats)

    # TF-IDF（Cause Analysis）
    tfidf_cols = META['tfidf_cols']
    X_tfidf = TFV.transform([cause or ""]).toarray()
    df_tfidf = pd.DataFrame(X_tfidf, columns=tfidf_cols)

    # 類別特徵（順序照 meta['cat_feats']，含 geo_cell/accident_cluster）
    cat_feats = META['cat_feats']
    cat_vals = {
        'Location': location, 'Weather': weather, 'Lighting': lighting, 'Road': road,
        'Ground Pavemen': ground_pav, 'Pavement Defect': pave_def, 'Road Obstacles': obstacles,
        'Sight Distance Quality': sight_q, 'Traffic Signal': signal, 'Lane Separation Facility': lane_sep,
        'Accident Type': acc_type, 'Crash Manner': crash_man, 'geo_cell': geo_cell,
        'accident_cluster': accident_cluster
    }
    X_cat = pd.DataFrame([cat_vals], columns=cat_feats).astype(str)

    # 拼接（num -> tfidf -> cat），並回傳 Pool 需要的 cat_idxs
    X = pd.concat([X_num, df_tfidf, X_cat], axis=1)
    all_cols = list(X.columns)
    cat_idxs = [all_cols.index(c) for c in cat_feats]
    return X, cat_idxs, {
        "Hour":Hour,"Month":Month,"Weekday":Weekday,"is_weekend":is_weekend,"is_peak":is_peak,
        "Speed Limit":SpeedLimit,"RoadCond_Ord":RoadCond_Ord,"lon_r":lon_r,"lat_r":lat_r,
        "geo_cell":geo_cell,"accident_cluster":accident_cluster
    }

if run:
    try:
        X, cat_idxs, used = featurize_from_inputs()
        pool = Pool(X, cat_features=cat_idxs)
        proba = MODEL.predict_proba(pool)[0]
        idx_A1 = META['idx_A1']
        best_thr = META['best_thr']
        pA1 = float(proba[idx_A1]); pA2 = 1.0 - pA1
        pred = "A1" if pA1 >= best_thr else "A2"

        st.success(f"**預測結果：{pred}**（threshold={best_thr:.3f}）")
        c1, c2 = st.columns(2)
        with c1: st.metric("P(A1)", f"{pA1:.3f}")
        with c2: st.metric("P(A2)", f"{pA2:.3f}")

        with st.expander("送入模型的特徵（檢查用）", expanded=False):
            st.json({
                "num_feats_order": META['num_feats'],
                "cat_feats_order": META['cat_feats'],
                "tfidf_cols": len(META['tfidf_cols']),
                "derived": used
            })
    except Exception as e:
        st.error(f"推論失敗：{e}")
        st.exception(e)
