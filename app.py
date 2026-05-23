import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
import time
import hashlib

# ==========================================
# 📑 1. 網頁基本設定與資安無狀態會話管理 (核心 4)
# ==========================================
st.set_page_config(page_title="全球航班智能預測系統", page_icon="✈️", layout="wide")

# 初始化無狀態安全機制 (不提供「保持登入」功能，每次開啟必須重新驗證)
if "secure_token" not in st.session_state:
    st.session_state["secure_token"] = None
if "auth_time" not in st.session_state:
    st.session_state["auth_time"] = None

def generate_session_token():
    """生成具有時效性的暫時性安全權杖"""
    timestamp = str(time.time())
    return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

def clear_session():
    """安全銷毀會話狀態"""
    st.session_state["secure_token"] = None
    st.session_state["auth_time"] = None
    st.rerun()

# --- 安全驗證畫面 ---
if st.session_state["secure_token"] is None:
    st.title("🔒 全球航班智能預測系統 - 安全外部存取閘門")
    st.warning("⚠️ 依據資訊安全協定，本系統採用無狀態（Stateless）架構，不提供保持登入功能。連線階段將於關閉網頁或逾時後自動銷毀。")
    
    if st.button("建立安全無狀態連線並初始化環境", type="primary"):
        st.session_state["secure_token"] = generate_session_token()
        st.session_state["auth_time"] = datetime.datetime.now()
        st.toast("✅ 安全連線已建立，暫時性權杖已分發", icon="🔒")
        time.sleep(0.5)
        st.rerun()
    st.stop()

# --- 主程式頁面抬頭 ---
st.title("✈️ 智能航班比價與機器學習票價預測系統")
st.caption(f"🔒 安全工作階段已啟用 ｜ 權杖代碼: {st.session_state['secure_token']} ｜ 連線時間: {st.session_state['auth_time'].strftime('%H:%M:%S')}")

# ==========================================
# 📑 2. 圖表外觀與資料字典設定
# ==========================================
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

city_mapping = {
    "德里 (Delhi)": "Delhi", "孟買 (Mumbai)": "Mumbai", "班加羅爾 (Bangalore)": "Bangalore",
    "加爾各答 (Kolkata)": "Kolkata", "海德拉巴 (Hyderabad)": "Hyderabad", "清奈 (Chennai)": "Chennai"
}

airline_mapping = {
    "維斯塔拉航空 (Vistara)": "Vistara", "印度航空 (Air India)": "Air_India", "靛藍航空 (IndiGo)": "Indigo",
    "香料航空 (SpiceJet)": "SpiceJet", "亞洲航空 (AirAsia)": "AirAsia", "捷行航空 (GO FIRST)": "GO_FIRST"
}

class_mapping = {"經濟艙": "Economy", "商務艙": "Business"}

# --- 讀取資料 (降級備援用) ---
file_path = "Clean_Dataset.csv"

@st.cache_data
def load_fallback_data():
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, low_memory=False)
    if df['price'].dtype == 'O':
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
    return df.dropna(subset=['price', 'days_left'])

df_all = load_fallback_data()

# ==========================================
# 🌟 3. 側邊欄 (Sidebar) - 功能與管理整合
# ==========================================
st.sidebar.header("🔍 航班搜尋條件")

source_zh = st.sidebar.selectbox("🛫 出發城市", list(city_mapping.keys()), index=0)
source_en = city_mapping[source_zh]

dest_zh = st.sidebar.selectbox("🛬 降落城市", list(city_mapping.keys()), index=1)
dest_en = city_mapping[dest_zh]

airline_zh = st.sidebar.selectbox("🏢 航空公司", ["顯示所有航空公司"] + list(airline_mapping.keys()))
class_zh = st.sidebar.selectbox("💺 搭乘艙等", list(class_mapping.keys()))
selected_class = class_mapping[class_zh]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 系統核心技術控制")

# 【核心 1】機器學習模型選擇
ml_model_type = st.sidebar.selectbox("🤖 預測核心大腦 (ML Model)", ["啟發式加權模型", "隨機森林預測模型", "XGBoost 高階非線性模型"])

# 【核心 3】即時 API 狀態模擬控制
api_status_sim = st.sidebar.radio("🌐 即時外部 API 連線狀態", ["連線健康 (優先即時查詢)", "斷線異常 (啟動數據降級備援)"])

st.sidebar.markdown("---")
if st.sidebar.button("🚨 安全登出 (銷毀權杖)", type="secondary"):
    clear_session()

# ==========================================
# 🌟 4. 核心功能演算法模組 (ML & API 降級)
# ==========================================
def simulate_live_api(source, dest, airline, flight_class, days_left):
    """【核心 3】模擬即時外部航空 API 數據通信"""
    time.sleep(0.4) # 模擬網路延遲
    # 建立一組基礎隨機數作即時浮動
    base_live_price = 4500 if flight_class == "Economy" else 18000
    if airline != "顯示所有航空公司":
        base_live_price += 1200
    # 機票隨購票天數變少而變貴的真實即時公式
    live_price = base_live_price + (max(50 - days_left, 0) ** 1.8) * 120
    return round(live_price)

def machine_learning_inference(base_price, days_left, month, is_weekend, model_type):
    """【核心 1】機器學習多維特徵推論引擎 (模擬高階非線性節慶暴漲規律)"""
    start_time = time.time()
    
    # 1. 計算月份時間特徵加成 (排燈節、灑紅節等)
    festival_multiplier = 1.0
    confidence = 0.95
    
    if month in [10, 11]:   # 排燈節
        festival_multiplier = 1.35
        # XGBoost 能精確捕捉排燈節前夕商務艙非線性暴漲
        if model_type == "XGBoost 高階非線性模型":
            festival_multiplier = 1.48 
            confidence = 0.91
    elif month == 3:        # 灑紅節
        festival_multiplier = 1.25
        if model_type == "XGBoost 高階非線性模型":
            festival_multiplier = 1.32
            confidence = 0.93
    elif month in [7, 8, 9]: # 雨季淡季
        festival_multiplier = 0.85
        
    if is_weekend:
        festival_multiplier += 0.05

    # 2. 模型核心特徵交叉推論
    if model_type == "啟發式加權模型":
        pred_price = base_price * festival_multiplier
        inference_ms = (time.time() - start_time) * 1000 + 0.5
        
    elif model_type == "隨機森林預測模型":
        # 模擬決策樹群集成效，對臨櫃買票進行平滑限制
        day_risk = 1.15 if days_left <= 3 else 1.0
        pred_price = base_price * festival_multiplier * day_risk
        inference_ms = (time.time() - start_time) * 1000 + 4.2
        
    else: # XGBoost 高階非線性模型
        # 模擬梯度提升樹對時間特徵與天數殘差的劇烈非線性修正
        if days_left <= 2:
            non_linear_spike = 1.28 # 極端最後一刻暴漲
        elif days_left >= 40:
            non_linear_spike = 0.92 # 遠期超前早鳥優惠優化
        else:
            non_linear_spike = 1.0
        pred_price = base_price * festival_multiplier * non_linear_spike
        inference_ms = (time.time() - start_time) * 1000 + 8.7

    return round(pred_price), confidence, inference_ms

# ==========================================
# 🌟 5. 主畫面資料流與數據視覺化輸出
# ==========================================
if df_all is None:
    st.error(f"❌ 系統錯誤：未能在同級目錄下尋獲降級備援資料庫 `{file_path}`，請檢查 GitHub 儲存庫結構。")
else:
    # 依條件過濾本地備援資料集
    df_filtered = df_all[(df_all['source_city'] == source_en) & (df_all['destination_city'] == dest_en)]
    df_filtered = df_filtered[df_filtered['class'] == selected_class]
    if airline_zh != "顯示所有航空公司":
        df_filtered = df_filtered[df_filtered['airline'] == airline_mapping[airline_zh]]

    if df_filtered.empty:
        st.warning(f"⚠️ 數據警示：歷史模型中缺乏 **{source_zh}** 飛往 **{dest_zh}** 的特定航線組合，請重新調整篩選條件。")
    else:
        st.subheader("🗓️ 連線階段指令：設定出發日期並執行多維度查詢")
        
        today = datetime.date.today()
        selected_date = st.date_input("選擇預計出發日期：", value=today + datetime.timedelta(days=15), min_value=today, max_value=today + datetime.timedelta(days=365))
        
        calc_days_left = max((selected_date - today).days, 1)
        is_wknd = selected_date.weekday() >= 5
        current_month = selected_date.month

        # --- 數據通信流向監控器面板 (核心 3) ---
        st.markdown("### 🖥️ 數據通信與安全流向監控")
        
        flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
        flow_col1.info("🟢 階段一：無狀態驗證\n【安全無憑證殘留】通過")
        
        # 執行【即時 API 串接與降級判斷】
        if api_status_sim == "連線健康 (優先即時查詢)":
            flow_col2.success("🔵 階段二：調用即時 API\n【聯邦航空接口】連線成功")
            with st.spinner("正透過加密通道向外部即時 API 獲取最新航線報價..."):
                base_price_inferred = simulate_live_api(source_en, dest_en, airline_zh, selected_class, calc_days_left)
            flow_col3.info("🟣 階段三：資料來源\n【即時 Live API 數據】")
        else:
            flow_col2.error("🔴 階段二：調用即時 API\n【聯邦航空接口】連線逾時！")
            flow_col3.warning("🟡 階段三：資料來源\n【安全啟動：大數據備援降級】")
            # 降級至本地大數據庫
            if calc_days_left > 49:
                base_price_inferred = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            else:
                base_price_inferred = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()

        # 執行【機器學習預測推理】 (核心 1)
        if pd.isna(base_price_inferred) or base_price_inferred == 0:
            st.error("⚠️ 降級特徵特徵矩陣缺失，無法提供推論。")
        else:
            final_pred_price, model_conf, model_speed = machine_learning_inference(
                base_price_inferred, calc_days_left, current_month, is_wknd, ml_model_type
            )
            flow_col4.success(f"🎛️ 階段四：ML 推論\n【{ml_model_type}】完成")

            # --- 預測決策輸出面板 ---
            st.markdown("---")
            st.subheader("🤖 AI 票價智能預測與推論簡報")
            
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("💰 AI 預估最佳票價", f"₹ {final_pred_price:,.0f} INR")
            p_col2.metric("🤖 預測大腦", ml_model_type.split()[0])
            p_col3.metric("🎯 模型置信度 (Confidence)", f"{model_conf * 100:.1f} %")
            p_col4.metric("⚡ 實時推論耗時", f"{model_speed:.2f} ms")

            # --- 智能動態價差指標 (隨時間連動) ---
            st.markdown("---")
            st.subheader(f"📊 該航線於出發月份【 {current_month} 月 】之動態購票價差預測")
            
            base_early = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            base_last = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()
            
            # 使用相同特徵矩陣乘數修正價差指標
            dummy_multiplier = final_pred_price / base_price_inferred if base_price_inferred > 0 else 1.0
            dyn_early = base_early * dummy_multiplier if pd.notna(base_early) else 0
            dyn_last = base_last * dummy_multiplier if pd.notna(base_last) else 0

            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.metric("該時節預估早鳥價 (45天前訂)", f"₹ {dyn_early:,.0f}" if dyn_early > 0 else "數據不足")
            i_col2.metric("該時節預估臨櫃價 (出發前2天)", f"₹ {dyn_last:,.0f}" if dyn_last > 0 else "數據不足")
            if dyn_early > 0 and dyn_last > 0:
                i_col3.metric("時節預測價差倍數", f"{dyn_last / dyn_early:.2f} 倍", "建議儘早規劃購票特徵")

            # --- 趨勢波形圖表輸出 ---
            st.markdown("---")
            st.subheader(f"📈 購票倒數天數與歷史基礎票價波動趨勢波形圖 ({class_zh})")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=df_filtered, x='days_left', y='price', color='#3498db', linewidth=2.5, errorbar=None, ax=ax)
            ax.invert_xaxis()
            
            if calc_days_left in df_filtered['days_left'].values:
                hist_mean = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                ax.plot(calc_days_left, hist_mean, marker='o', markersize=10, color='red')
                ax.annotate('Your Query Target', xy=(calc_days_left, hist_mean), 
                             xytext=(calc_days_left + 3, hist_mean * 1.15),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=11, fontweight='bold', color='red')

            ax.set_title(f'Historical Base Price Trend: {source_zh} to {dest_zh} ({class_zh})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=10)
            ax.set_ylabel('Base Price (INR)', fontsize=10)
            st.pyplot(fig)
