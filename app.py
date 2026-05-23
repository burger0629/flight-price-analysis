import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="全球航班智能比價系統", page_icon="✈️", layout="wide")
st.title("✈️ 智能航班比價與全年票價趨勢預測系統")
st.markdown("設定航線與出發日期，系統將結合歷史大數據與全年節慶動態定價模型，為您預測最佳購買時機。")

# --- 2. 圖表外觀設定 ---
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# --- 3. 建立中文化對照表 (對應印度 Clean_Dataset 的原始資料) ---
city_mapping = {
    "德里 (Delhi)": "Delhi",
    "孟買 (Mumbai)": "Mumbai",
    "班加羅爾 (Bangalore)": "Bangalore",
    "加爾各答 (Kolkata)": "Kolkata",
    "海德拉巴 (Hyderabad)": "Hyderabad",
    "清奈 (Chennai)": "Chennai"
}

airline_mapping = {
    "維斯塔拉航空 (Vistara)": "Vistara",
    "印度航空 (Air India)": "Air_India",
    "靛藍航空 (IndiGo)": "Indigo",
    "香料航空 (SpiceJet)": "SpiceJet",
    "亞洲航空 (AirAsia)": "AirAsia",
    "捷行航空 (GO FIRST)": "GO_FIRST"
}

class_mapping = {
    "經濟艙": "Economy", 
    "商務艙": "Business"
}

# --- 4. 讀取資料 ---
file_path = "Clean_Dataset.csv"

if not os.path.exists(file_path):
    st.error(f"❌ 找不到資料檔 `{file_path}`，請確定檔案有上傳到 GitHub！")
else:
    @st.cache_data
    def load_data():
        df = pd.read_csv(file_path, low_memory=False)
        if df['price'].dtype == 'O':
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
        return df.dropna(subset=['price', 'days_left'])

    with st.spinner("正在載入全球航班數據庫..."):
        df_all = load_data()

        # ==========================================
        # 🌟 側邊欄 (Sidebar) - 全中文顯示
        # ==========================================
        st.sidebar.header("🔍 航班條件篩選")
        
        # 讓使用者選中文，後端拿對應的英文去過濾資料
        source_zh = st.sidebar.selectbox("🛫 出發城市", list(city_mapping.keys()), index=0)
        source_en = city_mapping[source_zh]
        
        dest_zh = st.sidebar.selectbox("🛬 降落城市", list(city_mapping.keys()), index=1)
        dest_en = city_mapping[dest_zh]
        
        airline_zh = st.sidebar.selectbox("🏢 航空公司", ["顯示所有航空公司"] + list(airline_mapping.keys()))
        class_zh = st.sidebar.selectbox("💺 搭乘艙等", list(class_mapping.keys()))
        selected_class = class_mapping[class_zh]

        # 進行後台資料過濾
        df_filtered = df_all[(df_all['source_city'] == source_en) & (df_all['destination_city'] == dest_en)]
        df_filtered = df_filtered[df_filtered['class'] == selected_class]
        
        if airline_zh != "顯示所有航空公司":
            airline_en = airline_mapping[airline_zh]
            df_filtered = df_filtered[df_filtered['airline'] == airline_en]

        # ==========================================
        # 🌟 主畫面：最大化年份動態定價試算
        # ==========================================
        if df_filtered.empty:
            st.warning(f"⚠️ 找不到從 **{source_zh}** 飛往 **{dest_zh}** 的 **{airline_zh}** ({class_zh}) 航班紀錄，請嘗試放寬篩選條件！")
        else:
            st.success(f"✅ 成功對接該航線 **{len(df_filtered):,}** 筆歷史大數據模型。")
            
            st.markdown("---")
            st.subheader("🗓️ 選擇出發日期（支援全年 365 天查詢）")
            
            today = datetime.date.today()
            selected_date = st.date_input(
                "請選擇您的預計出發日期：",
                value=today + datetime.timedelta(days=15),
                min_value=today,
                max_value=today + datetime.timedelta(days=365)
            )

            calc_days_left = (selected_date - today).days
            calc_days_left = max(calc_days_left, 1)

            # --- 全年份 12 個月動態時節與權重演算法 ---
            month = selected_date.month
            price_multiplier = 1.0
            season_details = ""

            if month in [1, 2]:
                season_name = "❄️ 冬季旅遊淡季"
                price_multiplier = 0.90
                season_details = "年節過後市場需求放緩，機票價格維持低檔。"
            elif month == 3:
                season_name = "🎨 灑紅節 (Holi) 慶典旺季"
                price_multiplier = 1.25
                season_details = "印度傳統重要節慶，返鄉與慶祝人潮導致運能緊張，票價大幅上揚。"
            elif month == 4:
                season_name = "🌸 春季平季"
                price_multiplier = 1.00
                season_details = "氣候宜人，旅遊市場供需平衡，維持標準基準價。"
            elif month in [5, 6]:
                season_name = "☀️ 暑期暑假旅遊旺季"
                price_multiplier = 1.15
                season_details = "學生假期與觀光旺季，航班機位需求顯著增加。"
            elif month in [7, 8, 9]:
                season_name = "🌧️ 季風雨季旅遊淡季"
                price_multiplier = 0.85
                season_details = "受降雨與氣候影響為傳統旅遊淡季，航空公司普遍推出促銷優惠。"
            elif month in [10, 11]:
                season_name = "🪔 排燈節 (Diwali) 傳統黃金旺季"
                price_multiplier = 1.35
                season_details = "年度最核心盛大節慶，全民返鄉與旅遊潮引發年度最高峰定價。"
            elif month == 12:
                season_name = "🎄 年終聖誕與跨年狂歡旺季"
                price_multiplier = 1.30
                season_details = "年底跨年假期及全球商務度假潮，票價明顯走高。"

            # 週末加成
            if selected_date.weekday() >= 5:
                season_name += " + 🎉 週末效應"
                price_multiplier += 0.05

            # 智慧型歷史數據匹配
            if calc_days_left > 49:
                base_price = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
                mode_text = "填補遠期早鳥基準價"
            else:
                base_price = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                mode_text = "精準匹配購票天數"

            # 輸出分析報告
            st.info(f"💡 **出發倒數：** {calc_days_left} 天 ｜ **預測時節：** {season_name} ｜ **市場動態需求指數：** {price_multiplier:.2f}x")
            
            if pd.notna(base_price):
                final_price = base_price * price_multiplier
                
                col_p1, col_p2 = st.columns([2, 3])
                with col_p1:
                    st.metric("💰 AI 預估當日票價", f"₹ {final_price:,.0f} INR", f"已整合 {season_name} 權重")
                with col_p2:
                    st.caption(f"**市場動態簡報：** {season_details}（計算模式：{mode_text}）")
            else:
                st.warning("⚠️ 此篩選組合之遠期數據樣本較少，請參考下方大趨勢走向。")

            # --- 4. 數據指標展示 (🌟 已修正：讓指標隨著時間連動變更) ---
            st.markdown("---")
            st.subheader(f"📊 預測該航線在【{month}月份】的購票價差")
            
            # 抓取歷史基礎值
            base_early_bird = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            base_last_minute = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()
            
            # 🌟 乘上時間乘數，讓這兩個指標也會隨著日期調整而變動！
            dynamic_early_bird = base_early_bird * price_multiplier if pd.notna(base_early_bird) else None
            dynamic_last_minute = base_last_minute * price_multiplier if pd.notna(base_last_minute) else None

            col1, col2, col3 = st.columns(3)
            
            if dynamic_early_bird:
                col1.metric(f"預估早鳥價 (45天前訂)", f"₹ {dynamic_early_bird:,.0f}", f"原始基準: ₹ {base_early_bird:,.0f}")
            else:
                col1.metric(f"預估早鳥價", "數據不足")
                
            if dynamic_last_minute:
                col2.metric(f"預估當天價 (臨櫃搶票)", f"₹ {dynamic_last_minute:,.0f}", f"原始基準: ₹ {base_last_minute:,.0f}")
            else:
                col2.metric(f"預估當天價", "數據不足")
            
            if pd.notna(dynamic_early_bird) and dynamic_early_bird > 0 and pd.notna(dynamic_last_minute):
                diff_ratio = dynamic_last_minute / dynamic_early_bird
                col3.metric("時節預測價差倍數", f"{diff_ratio:.2f} 倍", "不論淡旺季，越晚買通常越貴")

            # --- 5. 繪製趨勢圖表 ---
            st.markdown("---")
            st.subheader(f"📈 購票倒數天數與票價波動趨勢波形圖 ({class_zh})")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=df_filtered, x='days_left', y='price', color='#3498db', linewidth=2.5, errorbar=None, ax=ax)
            ax.invert_xaxis()
            
            if calc_days_left in df_filtered['days_left'].values:
                current_price = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                ax.plot(calc_days_left, current_price, marker='o', markersize=10, color='red')
                ax.annotate('Your Selection', xy=(calc_days_left, current_price), 
                             xytext=(calc_days_left + 3, current_price * 1.1),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=11, fontweight='bold', color='red')

            ax.set_title(f'Historical Pricing Data Trend: {source_zh} to {dest_zh} ({class_zh})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=11)
            ax.set_ylabel('Average Price (INR)', fontsize=11)
            
            st.pyplot(fig)
