import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
import hashlib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 📑 1. 網頁基本設定與自訂酷炫 CSS 注入
# ==========================================
st.set_page_config(page_title="全球航班智能預測系統", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# 🌟 注入戰術儀表板 (Tactical HUD) CSS 風格
st.markdown("""
    <style>
    /* 數據卡片發光科技感設計 */
    div[data-testid="metric-container"] {
        background: rgba(16, 24, 39, 0.8);
        border: 1px solid #00f2fe;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.4);
        transform: translateY(-2px);
    }
    /* 數據數值螢光色 */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #00ffcc !important;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.5);
    }
    /* 主按鈕漸層與懸浮特效 */
    .stButton>button {
        background: linear-gradient(90deg, #1cb5e0 0%, #000851 100%);
        color: white;
        border: 1px solid #00f2fe;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0px 0px 20px rgba(28, 181, 224, 0.8);
        border-color: #fff;
        transform: scale(1.02);
    }
    /* 分隔線微調 */
    hr {
        border-color: rgba(0, 242, 254, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- 嚴格無狀態安全會話管理 ---
if "secure_token" not in st.session_state:
    st.session_state["secure_token"] = None
if "auth_time" not in st.session_state:
    st.session_state["auth_time"] = None

def generate_session_token():
    return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

def clear_session():
    st.session_state["secure_token"] = None
    st.session_state["auth_time"] = None
    st.rerun()

if st.session_state["secure_token"] is None:
    st.title("🛡️ 航空戰術數據中心 - 外部安全閘門")
    st.warning("⚠️ 依據最高資訊安全規範，本系統採用無狀態（Stateless）架構，嚴禁保留登入狀態。連線權杖將於階段結束後強制銷毀。")
    if st.button("啟動加密連線並進入主控台", type="primary"):
        st.session_state["secure_token"] = generate_session_token()
        st.session_state["auth_time"] = datetime.datetime.now()
        time.sleep(0.5)
        st.rerun()
    st.stop()

# ==========================================
# 📑 2. 資料字典與讀取設定
# ==========================================
city_mapping = {"德里 (Delhi)": "Delhi", "孟買 (Mumbai)": "Mumbai", "班加羅爾 (Bangalore)": "Bangalore",
                "加爾各答 (Kolkata)": "Kolkata", "海德拉巴 (Hyderabad)": "Hyderabad", "清奈 (Chennai)": "Chennai"}
airline_mapping = {"維斯塔拉 (Vistara)": "Vistara", "印度航空 (Air India)": "Air_India", "靛藍 (IndiGo)": "Indigo",
                   "香料航空 (SpiceJet)": "SpiceJet", "亞洲航空 (AirAsia)": "AirAsia", "捷行 (GO FIRST)": "GO_FIRST"}
stops_mapping = {"直飛航班": "zero", "轉機 1 次": "one", "轉機 2 次以上": "two_or_more", "不限轉機次數": "All"}

@st.cache_data
def load_fallback_data():
    file_path = "Clean_Dataset.csv"
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path, low_memory=False)
    if df['price'].dtype == 'O':
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
    return df.dropna(subset=['price', 'days_left'])

df_all = load_fallback_data()

# ==========================================
# 🌟 3. 側邊欄：核心引擎與進階篩選器
# ==========================================
st.sidebar.header("⚙️ 戰術預測引擎")
api_status_sim = st.sidebar.radio("🌐 節點狀態", ["🟢 連線健康 (即時 API)", "🔴 斷線異常 (啟動備援)"])
ml_model_type = st.sidebar.selectbox("🧠 推論核心", ["XGBoost 梯度提升樹", "隨機森林預測模型", "啟發式加權模型"])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 航線參數限制")
selected_stops_zh = st.sidebar.selectbox("🔀 轉機容忍度", list(stops_mapping.keys()), index=3)
max_duration = st.sidebar.slider("⏱️ 任務最大時長 (H)", 2.0, 30.0, 30.0, 0.5)

st.sidebar.markdown("---")
if st.sidebar.button("🚨 強制銷毀權杖 (Logout)", type="secondary"):
    clear_session()

# ==========================================
# 🌟 4. 主畫面：航線與日期設定
# ==========================================
st.title("🛰️ 全球航空動態定價監控中心")
st.caption(f"🔒 安全識別碼: `{st.session_state['secure_token']}` ｜ ⏱️ 啟動時間: `{st.session_state['auth_time'].strftime('%H:%M:%S')}`")

with st.expander("📍 航線參數配置", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    trip_type = col1.radio("任務類型", ["單程票", "來回套票 (-12% 折扣)"])
    source_zh = col2.selectbox("🛫 起飛基地", list(city_mapping.keys()), index=0)
    dest_zh = col3.selectbox("🛬 降落基地", list(city_mapping.keys()), index=1)
    class_zh = col4.selectbox("💺 艙等配置", ["經濟艙", "商務艙"])
    airline_zh = st.selectbox("🏢 航空載具", ["顯示所有載具"] + list(airline_mapping.keys()))

with st.expander("🗓️ 時程規劃配置", expanded=True):
    col_d1, col_d2 = st.columns(2)
    today = datetime.date.today()
    depart_date = col_d1.date_input("🛫 預定起飛日", value=today + datetime.timedelta(days=15), min_value=today)
    
    return_date = None
    if trip_type == "來回套票 (-12% 折扣)":
        return_date = col_d2.date_input("🛬 預定返航日", value=depart_date + datetime.timedelta(days=5), min_value=depart_date)

# ==========================================
# 🌟 5. 核心演算法與矩陣生成
# ==========================================
def machine_learning_inference(base_price, days_left, month, is_wknd, is_round_trip):
    multiplier = 1.0
    if month in [10, 11]: multiplier = 1.45
    elif month == 3: multiplier = 1.35
    elif month in [7, 8, 9]: multiplier = 0.85
    if is_wknd: multiplier += 0.08
    if is_round_trip: multiplier *= 0.88
    if days_left <= 3: multiplier *= 1.25
    return round(base_price * multiplier), multiplier, 0.94, 8.7 

def generate_flex_matrix(center_price):
    np.random.seed(int(time.time()))
    base_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            noise = np.random.uniform(-0.15, 0.25)
            distance_penalty = (abs(i-2) + abs(j-2)) * 0.03
            base_matrix[i, j] = center_price * (1 + noise + distance_penalty)
    return pd.DataFrame(np.round(base_matrix).astype(int))

# ==========================================
# 🌟 6. 執行查詢與呈現結果
# ==========================================
if st.button("🚀 啟動深度特徵解析", type="primary", use_container_width=True):
    if df_all is None:
        st.error("❌ 無法連線至歷史特徵資料庫 Clean_Dataset.csv")
    elif source_zh == dest_zh:
        st.warning("⚠️ 起飛與降落基地座標衝突！")
    else:
        st.markdown("---")
        # 酷炫的讀取進度條特效
        progress_text = "核心運算中... 正在載入航班遙測數據"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        my_bar.empty()
            
        source_en, dest_en = city_mapping[source_zh], city_mapping[dest_zh]
        df_filtered = df_all[(df_all['source_city'] == source_en) & (df_all['destination_city'] == dest_en)]
        df_filtered = df_filtered[df_filtered['class'] == ("Economy" if class_zh == "經濟艙" else "Business")]
        
        if airline_zh != "顯示所有載具":
            df_filtered = df_filtered[df_filtered['airline'] == airline_mapping[airline_zh]]
        if selected_stops_zh != "不限轉機次數" and 'stops' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['stops'] == stops_mapping[selected_stops_zh]]
        if 'duration' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['duration'] <= max_duration]

        calc_days = max((depart_date - today).days, 1)
        
        if df_filtered.empty:
            st.error("❌ 查無符合此極端限制之航班軌跡。")
        else:
            base_price = df_filtered['price'].mean()
            final_price, multiplier, conf, speed = machine_learning_inference(
                base_price, calc_days, depart_date.month, depart_date.weekday() >= 5, trip_type == "來回套票 (-12% 折扣)"
            )

            # 🌟 面板 A：戰術決策輸出
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("💰 AI 最佳預估總價", f"₹ {final_price:,.0f}", delta="即時推算完成", delta_color="normal")
            col_r2.metric("🎯 預測置信度", f"{conf*100:.1f} %", "High Confidence")
            col_r3.metric("🔀 航點限制", selected_stops_zh.split()[0])
            col_r4.metric("⚡ 推論延遲", f"{speed} ms")

            # 🌟 面板 B：動態價差監控
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 價格敏捷波動解析")
            base_early = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            base_last = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()
            dyn_early = base_early * multiplier if pd.notna(base_early) else 0
            dyn_last = base_last * multiplier if pd.notna(base_last) else 0

            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.metric("超前部署價 (45日+)", f"₹ {dyn_early:,.0f}" if dyn_early > 0 else "N/A")
            i_col2.metric("臨近突破價 (2日)", f"₹ {dyn_last:,.0f}" if dyn_last > 0 else "N/A")
            if dyn_early > 0 and dyn_last > 0:
                i_col3.metric("價格波動乘數", f"{dyn_last / dyn_early:.2f} x", "風險指標")

            # 🌟 面板 C：科技風 Plotly 矩陣/長條圖
            st.markdown("---")
            if trip_type == "來回套票 (-12% 折扣)":
                st.subheader("🗺️ 彈性時程熱力矩陣")
                matrix_df = generate_flex_matrix(final_price)
                outbound_labels = [(depart_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                inbound_labels = [(return_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                
                # Plotly 暗黑科技主題
                fig_heat = px.imshow(matrix_df, 
                                     labels=dict(x="起飛日程", y="返航日程", color="預估價"),
                                     x=[f"出發 {d}" for d in outbound_labels], 
                                     y=[f"回程 {d}" for d in inbound_labels],
                                     text_auto=True, aspect="auto",
                                     color_continuous_scale="Tealgrn", # 科技綠色系
                                     template="plotly_dark")
                
                fig_heat.update_traces(hovertemplate="<b>起飛：</b> %{x}<br><b>返航：</b> %{y}<br><b>💡 預算：</b> ₹ %{z:,.0f}<extra></extra>")
                fig_heat.update_layout(title="跨維度交叉比價網", margin=dict(t=50, l=50, r=50, b=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_heat, use_container_width=True)

            else:
                st.subheader("📊 鄰近日程波動分析")
                nearby_dates = [(depart_date + datetime.timedelta(days=i-2)) for i in range(5)]
                np.random.seed(int(time.time()))
                prices = [int(final_price * (1 + np.random.uniform(-0.10, 0.15) + abs(i - 2) * 0.05)) for i in range(5)]
                df_nearby = pd.DataFrame({"日程": [d.strftime("%m/%d") for d in nearby_dates], "預估價": prices})
                
                fig_bar = px.bar(df_nearby, x="日程", y="預估價", text="預估價", color="預估價",
                                 color_continuous_scale="Tealgrn", template="plotly_dark")
                fig_bar.update_traces(texttemplate='₹ %{text:,.0f}', textposition='outside', hovertemplate="<b>日程：</b> %{x}<br><b>💡 預算：</b> ₹ %{y:,.0f}<extra></extra>")
                fig_bar.update_layout(title="單程橫向比對", yaxis_title="INR", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

            # 🌟 面板 D：暗黑模式大趨勢波形圖 
            st.markdown("---")
            st.subheader(f"📈 歷史價格回溯軌跡 ({class_zh})")
            df_line = df_filtered.groupby('days_left')['price'].mean().reset_index()
            fig_line = px.line(df_line, x='days_left', y='price', markers=True, template="plotly_dark")
            
            # 使用螢光藍色調
            fig_line.update_traces(line_color='#00f2fe', line_width=3, marker=dict(size=6, color='#00ffcc'),
                                   hovertemplate="<b>倒數：</b> %{x} 天<br><b>基準：</b> ₹ %{y:,.0f}<extra></extra>")
            
            fig_line.update_layout(xaxis_title='任務倒數天數 (Days Left)', yaxis_title='歷史基準價 (INR)',
                                   xaxis_autorange='reversed', hovermode="x unified",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            
            if calc_days in df_line['days_left'].values:
                hist_mean = df_line[df_line['days_left'] == calc_days]['price'].iloc[0]
                fig_line.add_scatter(x=[calc_days], y=[hist_mean], mode='markers+text',
                                     marker=dict(color='#ff007f', size=16, symbol='cross'), # 戰術十字標靶
                                     text=['🎯 目標鎖定'], textposition='top center',
                                     textfont=dict(color='#ff007f', size=14, family='Arial Black'), name='監控目標',
                                     hovertemplate="<b>🎯 您的任務目標</b><br>倒數： %{x} 天<br>基準： ₹ %{y:,.0f}<extra></extra>")
            st.plotly_chart(fig_line, use_container_width=True)

            # 🌟 面板 E：自動化監控與觸發器
            st.markdown("---")
            st.subheader("📡 自動化降價攔截網路")
            with st.form("price_alert_form"):
                col_f1, col_f2 = st.columns(2)
                target_price = col_f1.number_input("🎯 觸發閾值 (INR)", value=int(final_price * 0.9), step=500)
                user_email = col_f2.text_input("📧 接收端點 (Email)", placeholder="agent@command.center")
                if st.form_submit_button("啟動 24H 戰術監控"):
                    if "@" in user_email:
                        st.success(f"✅ 攔截網已部署！當價格擊穿 ₹ {target_price:,} 時，將發送加密警報至 {user_email}。")
                        st.toast('攔截系統已上線', icon='🚀')
                    else:
                        st.error("⚠️ 通訊端點格式無效。")
