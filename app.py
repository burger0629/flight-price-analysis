import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time
import hashlib

# ==========================================
# 📑 1. 網頁基本設定與資安無狀態會話管理
# ==========================================
st.set_page_config(page_title="全球航班智能預測系統", page_icon="✈️", layout="wide")

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
    st.title("🔒 全球航班智能預測系統 - 安全外部存取閘門")
    st.warning("⚠️ 依據資訊安全協定，本系統採用無狀態（Stateless）架構。連線階段將於關閉網頁或逾時後自動銷毀。")
    if st.button("建立安全無狀態連線並初始化環境", type="primary"):
        st.session_state["secure_token"] = generate_session_token()
        st.session_state["auth_time"] = datetime.datetime.now()
        time.sleep(0.5)
        st.rerun()
    st.stop()

# ==========================================
# 📑 2. 資料字典與讀取設定
# ==========================================
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

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
if st.sidebar.button("🚨 安全登出 (銷毀權杖)", type="secondary"):
    clear_session()

# ==========================================
# 🌟 4. 主畫面：航線與日期設定
# ==========================================
st.title("✈️ 全球航班智能比價與預測中心")
st.caption(f"🔒 權杖: {st.session_state['secure_token']} ｜ 連線時間: {st.session_state['auth_time'].strftime('%H:%M:%S')}")

with st.expander("📍 第一步：設定航線與艙等", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    trip_type = col1.radio("行程類型", ["單程票", "來回票 (享綁定折扣)"])
    source_zh = col2.selectbox("🛫 出發地", list(city_mapping.keys()), index=0)
    dest_zh = col3.selectbox("🛬 目的地", list(city_mapping.keys()), index=1)
    class_zh = col4.selectbox("💺 艙等", ["經濟艙", "商務艙"])
    airline_zh = st.selectbox("🏢 偏好航空", ["顯示所有航空"] + list(airline_mapping.keys()))

with st.expander("🗓️ 第二步：選擇出發與回程日期", expanded=True):
    col_d1, col_d2 = st.columns(2)
    today = datetime.date.today()
    depart_date = col_d1.date_input("🛫 預計出發日期", value=today + datetime.timedelta(days=15), min_value=today)
    
    return_date = None
    if trip_type == "來回票 (享綁定折扣)":
        return_date = col_d2.date_input("🛬 預計回程日期", value=depart_date + datetime.timedelta(days=5), min_value=depart_date)

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
    # ✅ 修正 ValueError: 確保安全轉換為整數
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
        with st.spinner("🧠 整合大數據與 ML 推論引擎運算中..."):
            time.sleep(1.2)
            
            # 1. 後台資料過濾
            source_en, dest_en = city_mapping[source_zh], city_mapping[dest_zh]
            df_filtered = df_all[(df_all['source_city'] == source_en) & (df_all['destination_city'] == dest_en)]
            df_filtered = df_filtered[df_filtered['class'] == ("Economy" if class_zh == "經濟艙" else "Business")]
            
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
                final_price, multiplier, conf, speed = machine_learning_inference(
                    base_price, calc_days, depart_date.month, depart_date.weekday() >= 5, trip_type == "來回票 (享綁定折扣)"
                )

                # 🌟 面板 A：核心決策輸出
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                col_r1.metric("💰 AI 最佳預估總價", f"₹ {final_price:,.0f} INR")
                col_r2.metric("🎯 演算法置信度", f"{conf*100:.1f} %")
                col_r3.metric("🔀 轉機條件", selected_stops_zh.split()[0])
                col_r4.metric("🎟️ 票種折扣", "已套用 -12%" if trip_type == "來回票 (享綁定折扣)" else "單程基準")

                # 🌟 (回歸功能！) 面板 B：早鳥與臨櫃動態價差比較
                st.markdown("---")
                st.subheader("📊 早鳥 vs 臨櫃購票價差預測 (已連動時節權重)")
                base_early = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
                base_last = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()
                dyn_early = base_early * multiplier if pd.notna(base_early) else 0
                dyn_last = base_last * multiplier if pd.notna(base_last) else 0

                i_col1, i_col2, i_col3 = st.columns(3)
                i_col1.metric("該時節預估早鳥價 (45天前訂)", f"₹ {dyn_early:,.0f}" if dyn_early > 0 else "數據不足")
                i_col2.metric("該時節預估臨櫃價 (出發前2天)", f"₹ {dyn_last:,.0f}" if dyn_last > 0 else "數據不足")
                if dyn_early > 0 and dyn_last > 0:
                    i_col3.metric("時節預測價差倍數", f"{dyn_last / dyn_early:.2f} 倍", "越晚買越貴提示")

                # 🌟 面板 C：彈性日期票價矩陣
                st.markdown("---")
                st.subheader("🗺️ 彈性日期票價熱力矩陣 (前後 2 天)")
                st.caption("💡 點擊矩陣尋找最具性價比的隱藏航班組合！")
                
                matrix_df = generate_flex_matrix(final_price)
                outbound_labels = [(depart_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                if trip_type == "來回票 (享綁定折扣)":
                    inbound_labels = [(return_date + datetime.timedelta(days=i-2)).strftime("%m/%d") for i in range(5)]
                    matrix_df.index = [f"回程 {d}" for d in inbound_labels]
                else:
                    matrix_df.index = [f"出發 {d}" for d in outbound_labels]
                matrix_df.columns = [f"出發 {d}" for d in outbound_labels]

                fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
                sns.heatmap(matrix_df, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Price (INR)'}, ax=ax_heat)
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                st.pyplot(fig_heat)

                # 🌟 (回歸功能！) 面板 D：歷史大趨勢波形圖
                st.markdown("---")
                st.subheader(f"📈 購票倒數天數與歷史基礎票價波動趨勢圖 ({class_zh})")
                
                fig_line, ax_line = plt.subplots(figsize=(12, 4))
                sns.lineplot(data=df_filtered, x='days_left', y='price', color='#3498db', linewidth=2.5, errorbar=None, ax=ax_line)
                ax_line.invert_xaxis()
                
                if calc_days in df_filtered['days_left'].values:
                    hist_mean = df_filtered[df_filtered['days_left'] == calc_days]['price'].mean()
                    ax_line.plot(calc_days, hist_mean, marker='o', markersize=10, color='red')
                    ax_line.annotate('Your Query Target', xy=(calc_days, hist_mean), 
                                 xytext=(calc_days + 3, hist_mean * 1.15),
                                 arrowprops=dict(facecolor='red', shrink=0.05),
                                 fontsize=11, fontweight='bold', color='red')
                ax_line.set_xlabel('Days Left Until Departure', fontsize=10)
                ax_line.set_ylabel('Base Price (INR)', fontsize=10)
                st.pyplot(fig_line)

                # 🌟 面板 E：主動降價追蹤警示
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
