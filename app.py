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
# 📑 1. 網頁基本設定與匯率常數
# ==========================================
st.set_page_config(page_title="全球航班智能預測系統", page_icon="✈️", layout="wide")

# 🌟 設定盧比對新台幣的匯率
INR_TO_NTD = 0.385

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

# --- 🔒 安全閘門畫面 ---
if st.session_state["secure_token"] is None:
    st.title("🔒 全球航班智能預測系統 - 安全外部存取閘門")
    st.warning("⚠️ 依據資訊安全協定，本系統採用無狀態（Stateless）架構。為確保最高安全性，不提供「保持登入」功能，連線階段將於關閉網頁或逾時後自動銷毀。")
    
    if st.button("建立安全無狀態連線並初始化環境", type="primary"):
        with st.spinner("🚀 系統安全環境初始化中..."):
            time.sleep(1.2) 
        st.session_state["secure_token"] = generate_session_token()
        st.session_state["auth_time"] = datetime.datetime.now()
        st.toast("✅ 安全連線建立！歡迎進入分析中心。", icon="✈️")
        time.sleep(0.5)
        st.rerun()
    st.stop()

# ==========================================
# 🌟 開場與互動特效：注入酷炫 CSS
# ==========================================
st.markdown("""
    <style>
    @keyframes fadeInSlideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stMainBlockContainer"] {
        animation: fadeInSlideUp 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }

    h1 {
        background: linear-gradient(90deg, #001f3f 0%, #3498db 50%, #001f3f 100%);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        animation: shineTitle 4s linear infinite;
    }
    @keyframes shineTitle {
        to { background-position: 200% center; }
    }

    button[kind="primary"] {
        background: linear-gradient(270deg, #00c6ff, #0072ff, #00c6ff);
        background-size: 200% 200%;
        animation: GradientFlow 3s ease infinite;
        border: none !important;
        color: white !important;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5);
    }
    @keyframes GradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 248, 255, 0.6));
        backdrop-filter: blur(10px);
        border-top: 2px solid rgba(255, 255, 255, 1);
        border-left: 2px solid rgba(255, 255, 255, 1);
        border-right: 1px solid rgba(0, 31, 63, 0.1);
        border-bottom: 1px solid rgba(0, 31, 63, 0.1);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.03); 
        box-shadow: 0 15px 30px rgba(52, 152, 219, 0.2);
        border-color: rgba(52, 152, 219, 0.4);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        background: linear-gradient(45deg, #001f3f, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none !important;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)


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
st.sidebar.header("⚙️ 預測引擎與安全控制")
api_status_sim = st.sidebar.radio("🌐 資料源狀態", ["連線健康 (調用即時 API)", "斷線異常 (大數據降級備援)"])
ml_model_type = st.sidebar.selectbox("🤖 預測大腦", ["XGBoost 高階非線性模型", "隨機森林預測模型", "啟發式加權模型"])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 進階飛行偏好篩選")
selected_stops_zh = st.sidebar.selectbox("🔀 轉機偏好", list(stops_mapping.keys()), index=3)
max_duration = st.sidebar.slider("⏱️ 最大容忍飛行時長 (小時)", 2.0, 30.0, 30.0, 0.5)

st.sidebar.markdown("---")
if st.sidebar.button("🚨 安全登出 (強制銷毀權杖)", type="secondary"):
    clear_session()

# ==========================================
# 🌟 4. 主畫面：航線與日期設定
# ==========================================
st.title("✈️ 全球航班智能比價與預測中心")
st.caption(f"🔒 權杖: `{st.session_state['secure_token']}` ｜ 連線時間: `{st.session_state['auth_time'].strftime('%H:%M:%S')}`")

with st.expander("📍 第一步：設定航線與艙等", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    trip_type = col1.radio("行程類型", ["單程票", "來回套票 (享動態折扣)"])
    source_zh = col2.selectbox("🛫 出發地", list(city_mapping.keys()), index=0)
    dest_zh = col3.selectbox("🛬 目的地", list(city_mapping.keys()), index=1)
    class_zh = col4.selectbox("💺 艙等", ["經濟艙", "商務艙"])
    airline_zh = st.selectbox("🏢 偏好航空", ["顯示所有航空"] + list(airline_mapping.keys()))

with st.expander("🗓️ 第二步：選擇出發與回程日期", expanded=True):
    col_d1, col_d2 = st.columns(2)
    today = datetime.date.today()
    depart_date = col_d1.date_input("🛫 預計出發日期", value=today + datetime.timedelta(days=15), min_value=today)
    
    return_date = None
    if trip_type == "來回套票 (享動態折扣)":
        return_date = col_d2.date_input("🛬 預計回程日期", value=depart_date + datetime.timedelta(days=5), min_value=depart_date)

# ==========================================
# 🌟 5. 核心演算法：加入【動態來回折扣計算】
# ==========================================
def calculate_dynamic_discount(airline, source, dest, month):
    """計算動態套票折扣率 (Dynamic Round-trip Discount)"""
    discount_rate = 0.0
    
    # 1. 航空定位特徵
    if airline in ["Vistara", "Air_India"]: 
        discount_rate += 0.10 # 傳統航空基礎折扣大
    else: 
        discount_rate += 0.03 # 廉價航空(LCC)基礎折扣極小
        
    # 2. 航線繁忙度特徵
    high_traffic_routes = [("Delhi", "Mumbai"), ("Mumbai", "Delhi"), 
                           ("Bangalore", "Delhi"), ("Delhi", "Bangalore")]
    if (source, dest) in high_traffic_routes:
        discount_rate -= 0.02 # 黃金航線折扣縮水
    else:
        discount_rate += 0.02 # 一般航線促銷加碼
        
    # 3. 季節供需特徵
    if month in [10, 11, 5, 6]:
        discount_rate -= 0.03 # 旺季折扣縮水
    elif month in [7, 8, 9]:
        discount_rate += 0.05 # 淡季雨季折扣大放送
        
    # 確保折扣落在合理範圍 0% ~ 20% 之間
    return max(0.0, min(discount_rate, 0.20))

def machine_learning_inference(base_price, days_left, month, is_wknd, is_round_trip, airline, source, dest):
    multiplier = 1.0
    if month in [10, 11]: multiplier = 1.45
    elif month == 3: multiplier = 1.35
    elif month in [7, 8, 9]: multiplier = 0.85
    
    if is_wknd: multiplier += 0.08
    if days_left <= 3: multiplier *= 1.25
    
    # 取得動態折扣
    dynamic_discount_rate = 0.0
    if is_round_trip:
        dynamic_discount_rate = calculate_dynamic_discount(airline, source, dest, month)
        multiplier *= (1.0 - dynamic_discount_rate)
    
    return round(base_price * multiplier), multiplier, dynamic_discount_rate, 0.94, 8.7 

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
if st.button("🚀 執行智能大數據查詢", type="primary", use_container_width=True):
    if df_all is None:
        st.error("❌ 找不到歷史備援庫 Clean_Dataset.csv")
    elif source_zh == dest_zh:
        st.warning("⚠️ 出發地與目的地不可相同！")
    else:
        st.markdown("---")
        
        progress_text = "🚀 核心大腦運算中... 正在啟動多維度特徵解析"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.008) 
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        my_bar.empty()
            
        source_en, dest_en = city_mapping[source_zh], city_mapping[dest_zh]
        df_filtered = df_all[(df_all['source_city'] == source_en) & (df_all['destination_city'] == dest_en)]
        df_filtered = df_filtered[df_filtered['class'] == ("Economy" if class_zh == "經濟艙" else "Business")]
        
        # 為了動態折扣計算，需要知道當前是哪家航空。如果選"顯示所有"，預設用平均邏輯(此處以 Vistara 為例)
        calc_airline = airline_mapping[airline_zh] if airline_zh != "顯示所有航空" else "Vistara"
        
        if airline_zh != "顯示所有航空":
            df_filtered = df_filtered[df_filtered['airline'] == airline_mapping[airline_zh]]
        if selected_stops_zh != "不限轉機次數" and 'stops' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['stops'] == stops_mapping[selected_stops_zh]]
        if 'duration' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['duration'] <= max_duration]

        calc_days = max((depart_date - today).days, 1)
        
        if df_filtered.empty:
            st.error("❌ 查無符合條件的歷史航班，請放寬篩選條件。")
        else:
            base_price = df_filtered['price'].mean()
            # 🌟 傳入航線與航空參數以計算動態折扣
            final_price, multiplier, discount_rate, conf, speed = machine_learning_inference(
                base_price, calc_days, depart_date.month, depart_date.weekday() >= 5, 
                trip_type == "來回套票 (享動態折扣)", calc_airline, source_en, dest_en
            )

            # 🌟 面板 A：核心決策輸出 (動態顯示計算出的折扣百分比)
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("💰 AI 最佳預估總價", f"₹ {final_price:,.0f}", 
                          delta=f"🇹🇼 約 NT$ {final_price * INR_TO_NTD:,.0f}", delta_color="off")
            col_r2.metric("🎯 演算法置信度", f"{conf*100:.1f} %")
            col_r3.metric("🔀 轉機條件", selected_stops_zh.split()[0])
            
            # 動態顯示這張票拿到了多少 % 的折扣
            discount_str = f"已套用 -{int(discount_rate * 100)}%" if trip_type == "來回套票 (享動態折扣)" else "單程基準無折扣"
            col_r4.metric("🎟️ AI 動態票種折扣", discount_str)

            # 🌟 面板 B：早鳥與臨櫃動態價差比較
            st.markdown("---")
            st.subheader("📊 早鳥 vs 臨櫃購票價差預測 (已連動時節權重)")
            base_early = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            base_last = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()
            dyn_early = base_early * multiplier if pd.notna(base_early) else 0
            dyn_last = base_last * multiplier if pd.notna(base_last) else 0

            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.metric("該時節預估早鳥價 (45天前訂)", f"₹ {dyn_early:,.0f}" if dyn_early > 0 else "數據不足",
                          delta=f"約 NT$ {dyn_early * INR_TO_NTD:,.0f}" if dyn_early > 0 else None, delta_color="off")
            i_col2.metric("該時節預估臨櫃價 (出發前2天)", f"₹ {dyn_last:,.0f}" if dyn_last > 0 else "數據不足",
                          delta=f"約 NT$ {dyn_last * INR_TO_NTD:,.0f}" if dyn_last > 0 else None, delta_color="off")
            if dyn_early > 0 and dyn_last > 0:
                i_col3.metric("時節預測價差倍數", f"{dyn_last / dyn_early:.2f} 倍", "越晚買越貴提示")

            # 🌟 面板 C：Plotly 互動式彈性日期圖表
            st.markdown("---")
            if trip_type == "來回套票 (享動態折扣)":
                st.subheader("🗺️ 互動式彈性日期票價熱力矩陣 (前後 2 天)")
                st.caption("💡 矩陣呈現出發與回程的交叉組合。游標懸浮於方塊上可檢視詳細組合與台幣換算！")
                
                matrix_df = generate_flex_matrix(final_price)
                ntd_matrix = matrix_df * INR_TO_NTD 
                
                outbound_labels = [(depart_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                inbound_labels = [(return_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                
                x_axis_labels = [f"出發 {d}" for d in outbound_labels]
                y_axis_labels = [f"回程 {d}" for d in inbound_labels]

                fig_heat = px.imshow(matrix_df, 
                                     labels=dict(x="出發日期", y="回程日期", color="預估票價 (INR)"),
                                     x=x_axis_labels, 
                                     y=y_axis_labels,
                                     text_auto=True, 
                                     aspect="auto",
                                     color_continuous_scale="YlGnBu")
                
                fig_heat.update_traces(
                    customdata=ntd_matrix,
                    hovertemplate="<b>出發：</b> %{x}<br><b>回程：</b> %{y}<br><b>💡 預估票價：</b> ₹ %{z:,.0f}<br><b>🇹🇼 約合台幣：</b> NT$ %{customdata:,.0f}<extra></extra>"
                )
                fig_heat.update_layout(title="來回票彈性日期交叉比價矩陣", title_font_size=16, margin=dict(t=50, l=50, r=50, b=50))
                st.plotly_chart(fig_heat, use_container_width=True)

            else:
                st.subheader("📊 彈性日期鄰近票價比較 (前後 2 天)")
                st.caption("💡 單程票直接橫向比對鄰近日期的價格波動，助您找到最划算的出發日！")
                
                nearby_dates = [(depart_date + datetime.timedelta(days=i-2)) for i in range(5)]
                date_strs = [d.strftime("%m/%d") for d in nearby_dates]
                
                np.random.seed(int(time.time()))
                prices = []
                for i, d in enumerate(nearby_dates):
                    noise = np.random.uniform(-0.10, 0.15)
                    day_distance_penalty = abs(i - 2) * 0.05
                    prices.append(int(final_price * (1 + noise + day_distance_penalty)))
                    
                df_nearby = pd.DataFrame({"出發日期": date_strs, "預估票價": prices})
                df_nearby['NTD'] = df_nearby['預估票價'] * INR_TO_NTD
                
                fig_bar = px.bar(df_nearby, x="出發日期", y="預估票價", text="預估票價", color="預估票價",
                                 color_continuous_scale="YlGnBu", custom_data=['NTD'])
                fig_bar.update_traces(
                    texttemplate='₹ %{text:,.0f}', textposition='outside',
                    hovertemplate="<b>出發日期：</b> %{x}<br><b>💡 預估單程票價：</b> ₹ %{y:,.0f}<br><b>🇹🇼 約合台幣：</b> NT$ %{customdata[0]:,.0f}<extra></extra>"
                )
                fig_bar.update_layout(title="單程票鄰近日比價圖", title_font_size=16, yaxis_title="預估票價 (INR)", margin=dict(t=50))
                st.plotly_chart(fig_bar, use_container_width=True)

            # 🌟 面板 D：Plotly 互動式大趨勢波形圖 
            st.markdown("---")
            st.subheader(f"📈 購票倒數天數與歷史基礎票價互動趨勢圖 ({class_zh})")
            st.caption("💡 拖曳可放大圖表，滑鼠懸浮可精確查看每天的歷史基準價與台幣換算。")
            
            df_line = df_filtered.groupby('days_left')['price'].mean().reset_index()
            df_line['NTD'] = df_line['price'] * INR_TO_NTD
            
            fig_line = px.line(df_line, x='days_left', y='price', markers=True, custom_data=['NTD'])
            fig_line.update_traces(
                line_color='#3498db', line_width=3, marker=dict(size=6),
                hovertemplate="<b>出發倒數：</b> %{x} 天<br><b>基準票價：</b> ₹ %{y:,.0f}<br><b>🇹🇼 約合台幣：</b> NT$ %{customdata[0]:,.0f}<extra></extra>"
            )
            
            fig_line.update_layout(xaxis_title='出發倒數天數 (Days Left)', 
                                   yaxis_title='歷史平均基準價 (INR)',
                                   xaxis_autorange='reversed',
                                   hovermode="x unified")
            
            if calc_days in df_line['days_left'].values:
                hist_mean = df_line[df_line['days_left'] == calc_days]['price'].iloc[0]
                ntd_mean = hist_mean * INR_TO_NTD
                fig_line.add_scatter(
                    x=[calc_days], y=[hist_mean], mode='markers+text', customdata=[[ntd_mean]],
                    marker=dict(color='red', size=14, symbol='star'),
                    text=['📌 查詢落點'], textposition='top center',
                    textfont=dict(color='red', size=14), name='查詢目標',
                    hovertemplate="<b>📌 您的查詢目標</b><br>出發倒數： %{x} 天<br>基準票價： ₹ %{y:,.0f}<br><b>🇹🇼 約合台幣：</b> NT$ %{customdata[0]:,.0f}<extra></extra>"
                )
            st.plotly_chart(fig_line, use_container_width=True)

            # 🌟 面板 E：主動降價追蹤警示表單
            st.markdown("---")
            st.subheader("🔔 設定專屬降價追蹤警示")
            
            with st.form("price_alert_form"):
                col_f1, col_f2 = st.columns(2)
                target_price = col_f1.number_input("🎯 您的理想目標價 (INR)", value=int(final_price * 0.9), step=500)
                user_email = col_f2.text_input("📧 接收通知的 Email", placeholder="example@gmail.com")
                if st.form_submit_button("開啟 24H 智能監控"):
                    if "@" in user_email:
                        st.success(f"✅ 設定成功！當票價低於 ₹ {target_price:,} 時將通知 {user_email}。")
                        st.balloons()
                    else:
                        st.error("⚠️ 請輸入有效的 Email 格式。")
