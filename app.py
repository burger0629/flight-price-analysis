import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="全方位機票比價系統", page_icon="✈️", layout="wide")
st.title("✈️ 智能航班比價與趨勢預測系統")
st.markdown("設定您的航線與日期，AI 將為您分析歷史票價趨勢與最佳購買時機。")

# --- 2. 圖表外觀設定 ---
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# --- 3. 讀取資料 ---
file_path = "Clean_Dataset.csv"

if not os.path.exists(file_path):
    st.error(f"❌ 找不到資料檔 `{file_path}`，請確定檔案有上傳到 GitHub！")
else:
    @st.cache_data
    def load_data():
        df = pd.read_csv(file_path, low_memory=False)
        # 清洗價格欄位
        if df['price'].dtype == 'O':
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
        return df.dropna(subset=['price', 'days_left'])

    with st.spinner("正在載入全球航班數據庫..."):
        df_all = load_data()

        # ==========================================
        # 🌟 新增：側邊欄 (Sidebar) 互動篩選器
        # ==========================================
        st.sidebar.header("🔍 尋找您的航班")
        
        # 取得資料庫中不重複的選項
        cities = sorted(df_all['source_city'].dropna().unique())
        airlines = sorted(df_all['airline'].dropna().unique())
        classes = sorted(df_all['class'].dropna().unique())

        # 建立下拉式選單
        source = st.sidebar.selectbox("🛫 出發城市", cities, index=0)
        
        # 目的地預設選第二個城市，避免與出發地相同
        dest_index = 1 if len(cities) > 1 else 0
        dest = st.sidebar.selectbox("🛬 降落城市", cities, index=dest_index)
        
        selected_airline = st.sidebar.selectbox("🏢 航空公司", ["所有航空"] + airlines)
        selected_class = st.sidebar.selectbox("💺 艙等", classes)

        # 根據使用者的選擇過濾資料集 (Data Filtering)
        df_filtered = df_all[(df_all['source_city'] == source) & (df_all['destination_city'] == dest)]
        df_filtered = df_filtered[df_filtered['class'] == selected_class]
        
        if selected_airline != "所有航空":
            df_filtered = df_filtered[df_filtered['airline'] == selected_airline]

        # ==========================================
        # 🌟 主畫面：動態時節與票價試算器
        # ==========================================
        if df_filtered.empty:
            st.warning(f"⚠️ 找不到從 **{source}** 飛往 **{dest}** 的 **{selected_airline}** ({selected_class}) 航班紀錄，請嘗試放寬篩選條件！")
        else:
            st.success(f"✅ 成功找到 **{len(df_filtered):,}** 筆符合條件的歷史航班紀錄。")
            
            st.markdown("---")
            st.subheader("🗓️ 選擇出發日期與動態時節定價")
            
            today = datetime.date.today()
            selected_date = st.date_input(
                "請選擇預計出發日期：",
                value=today + datetime.timedelta(days=15),
                min_value=today,
                max_value=today + datetime.timedelta(days=49)
            )

            calc_days_left = (selected_date - today).days
            calc_days_left = max(calc_days_left, 1)

            # --- 動態時節演算法 (模擬真實市場供需) ---
            month = selected_date.month
            season_name = "一般平日"
            price_multiplier = 1.0

            if month in [5, 6]:
                season_name = "☀️ 暑假旅遊旺季"
                price_multiplier = 1.15
            elif month == 3:
                season_name = "🎨 灑紅節 (Holi) 期間"
                price_multiplier = 1.20
            elif month in [10, 11]:
                season_name = "🪔 排燈節 (Diwali) 返鄉潮"
                price_multiplier = 1.25
            elif selected_date.weekday() >= 5:
                season_name = "🎉 週末假日"
                price_multiplier = 1.05

            st.info(f"💡 距離出發：**{calc_days_left}** 天 ｜ 預測時節：**{season_name}** ｜ 需求指數：**{price_multiplier}x**")

            # 計算預估票價 (歷史平均 * 時節乘數)
            if calc_days_left in df_filtered['days_left'].values:
                base_price = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                final_price = base_price * price_multiplier
                st.metric("💰 AI 預測最佳票價 (INR)", f"₹ {final_price:,.0f}", f"受 {season_name} 影響")
            else:
                st.warning("⚠️ 此特定天數缺乏足夠歷史數據，請參考下方整體趨勢圖。")

            # --- 統計數據儀表板 (基於篩選後的數據) ---
            st.markdown("---")
            st.subheader(f"📊 {source} ✈️ {dest} 航線提前預訂價差")
            
            early_bird = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            last_minute = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("早鳥均價 (45天前)", f"₹ {early_bird:,.0f}" if pd.notna(early_bird) else "無資料")
            col2.metric("最後一刻 (出發前2天)", f"₹ {last_minute:,.0f}" if pd.notna(last_minute) else "無資料")
            
            if pd.notna(early_bird) and early_bird > 0 and pd.notna(last_minute):
                diff_ratio = last_minute / early_bird
                col3.metric("價差倍數", f"{diff_ratio:.2f} 倍")

            # --- 繪製趨勢圖表 (基於篩選後的數據) ---
            st.markdown("---")
            st.subheader(f"📈 票價隨時間波動趨勢圖 ({selected_class})")
            
            fig, ax = plt.subplots(figsize=(12, 6))

            # 使用 ci=None 避免數據量太少時畫不出陰影而報錯
            sns.lineplot(data=df_filtered, x='days_left', y='price', 
                         color='#3498db', linewidth=2.5, errorbar=None, ax=ax)

            ax.invert_xaxis()
            
            # 標註當前選擇的日期天數位置
            if calc_days_left in df_filtered['days_left'].values:
                current_price = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                ax.plot(calc_days_left, current_price, marker='o', markersize=10, color='red')
                ax.annotate('Your Selected Date', 
                             xy=(calc_days_left, current_price), 
                             xytext=(calc_days_left + 2, current_price * 1.1),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=12, fontweight='bold', color='red')

            ax.set_title(f'Price Trend: {source} to {dest}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=12)
            ax.set_ylabel('Average Price (INR)', fontsize=12)
            
            st.pyplot(fig)
