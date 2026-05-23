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
        # 🌟 側邊欄 (Sidebar) 互動篩選器 - 全面中文化
        # ==========================================
        st.sidebar.header("🔍 航班條件篩選")
        
        # 取得資料庫中不重複的城市與航空名稱
        cities = sorted(df_all['source_city'].dropna().unique())
        airlines = sorted(df_all['airline'].dropna().unique())

        # 1. 出發與目的地選擇
        source = st.sidebar.selectbox("🛫 出發城市", cities, index=0)
        dest_index = 1 if len(cities) > 1 else 0
        dest = st.sidebar.selectbox("🛬 降落城市", cities, index=dest_index)
        
        # 2. 航空公司選擇
        selected_airline = st.sidebar.selectbox("🏢 航空公司", ["顯示所有航空公司"] + airlines)
        
        # 3. 艙等選擇 (將後台對應轉換為中文顯示)
        class_mapping = {"經濟艙": "Economy", "商務艙": "Business"}
        selected_class_zh = st.sidebar.selectbox("💺 搭乘艙等", list(class_mapping.keys()))
        selected_class = class_mapping[selected_class_zh]

        # 根據側邊欄中文條件進行後台資料過濾
        df_filtered = df_all[(df_all['source_city'] == source) & (df_all['destination_city'] == dest)]
        df_filtered = df_filtered[df_filtered['class'] == selected_class]
        
        if selected_airline != "顯示所有航空公司":
            df_filtered = df_filtered[df_filtered['airline'] == selected_airline]

        # ==========================================
        # 🌟 主畫面：最大化年份動態定價試算
        # ==========================================
        if df_filtered.empty:
            st.warning(f"⚠️ 找不到從 **{source}** 飛往 **{dest}** 的 **{selected_airline}** ({selected_class_zh}) 航班紀錄，請嘗試放寬篩選條件！")
        else:
            st.success(f"✅ 成功對接該航線 **{len(df_filtered):,}** 筆歷史大數據模型。")
            
            st.markdown("---")
            st.subheader("🗓️ 選擇出發日期（支援全年 365 天查詢）")
            
            today = datetime.date.today()
            
            # 🌟 最大化時間段：引進整年日曆選單 (從今天起到 365 天後)
            selected_date = st.date_input(
                "請選擇您的預計出發日期：",
                value=today + datetime.timedelta(days=15),
                min_value=today,
                max_value=today + datetime.timedelta(days=365)
            )

            calc_days_left = (selected_date - today).days
            calc_days_left = max(calc_days_left, 1)

            # --- 🌟 全年份 12 個月動態時節與權重演算法 ---
            month = selected_date.month
            price_multiplier = 1.0
            season_details = ""

            if month in [1, 2]:
                season_name = "❄️ 冬季旅遊淡記"
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

            # --- 🌟 智慧型歷史數據匹配 ---
            # 如果查詢天數超過歷史資料上限(49天)，則採用長期早鳥穩定均價
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
                
                # 儀表板大字呈現
                col_p1, col_p2 = st.columns([2, 3])
                with col_p1:
                    st.metric("💰 AI 預估最佳票價", f"₹ {final_price:,.0f} INR", f"已整合 {season_name} 權重")
                with col_p2:
                    st.caption(f"**市場動態簡報：** {season_details}（計算模式：{mode_text}）")
            else:
                st.warning("⚠️ 此篩選組合之遠期數據樣本較少，請參考下方大趨勢走向。")

            # --- 4. 數據指標展示 ---
            st.markdown("---")
            st.subheader("📊 該特定航線之早鳥與最後一刻價差")
            
            early_bird = df_filtered[df_filtered['days_left'] >= 45]['price'].mean()
            last_minute = df_filtered[df_filtered['days_left'] <= 2]['price'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("早鳥均價 (45天前預訂)", f"₹ {early_bird:,.0f}" if pd.notna(early_bird) else "數據不足")
            col2.metric("最後一刻均價 (出發前2天內)", f"₹ {last_minute:,.0f}" if pd.notna(last_minute) else "數據不足")
            
            if pd.notna(early_bird) and early_bird > 0 and pd.notna(last_minute):
                diff_ratio = last_minute / early_bird
                col3.metric("極端價差倍數", f"{diff_ratio:.2f} 倍", "越晚買越貴提示")

            # --- 5. 繪製趨勢圖表 ---
            st.markdown("---")
            st.subheader(f"📈 購票倒數天數與票價波動趨勢波形圖 ({selected_class_zh})")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=df_filtered, x='days_left', y='price', color='#3498db', linewidth=2.5, errorbar=None, ax=ax)
            ax.invert_xaxis()
            
            # 在歷史圖表上動態點出使用者選擇的黃金交叉點（如果在49天內的話）
            if calc_days_left in df_filtered['days_left'].values:
                current_price = df_filtered[df_filtered['days_left'] == calc_days_left]['price'].mean()
                ax.plot(calc_days_left, current_price, marker='o', markersize=10, color='red')
                ax.annotate('Your Selection', xy=(calc_days_left, current_price), 
                             xytext=(calc_days_left + 3, current_price * 1.1),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=11, fontweight='bold', color='red')

            ax.set_title(f'Historical Pricing Data Trend: {source} to {dest} ({selected_class_zh})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=11)
            ax.set_ylabel('Average Price (INR)', fontsize=11)
            
            st.pyplot(fig)
