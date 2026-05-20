import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime # 🌟 新增：用來處理日期的內建套件

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="機票價格分析", page_icon="✈️", layout="wide")
st.title("✈️ 印度航班：提前預訂票價趨勢分析")
st.markdown("這份儀表板分析了「距離出發的剩餘天數」對「經濟艙票價」的影響。")

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
        if 'class' in df.columns:
            df_eco = df[df['class'].str.lower() == 'economy'].copy()
        else:
            df_eco = df.copy()
        
        if df_eco['price'].dtype == 'O':
            df_eco['price'] = pd.to_numeric(df_eco['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
            
        return df_eco.dropna(subset=['price', 'days_left'])

    with st.spinner("正在讀取資料與繪製圖表..."):
        df_eco = load_data()

        if 'days_left' not in df_eco.columns:
            st.error("❌ 資料集中找不到 `days_left` 欄位！")
        else:
            
            # ==========================================
            # 🌟 新增功能區塊：日期選擇與票價試算器
            # ==========================================
            st.markdown("---")
            st.subheader("🗓️ 模擬出發日期與票價試算")
            
            # 設定今天的日期
            today = datetime.date.today()
            
            # 使用 Streamlit 的日期選擇器
            selected_date = st.date_input(
                "請選擇您的預計出發日期：",
                value=today + datetime.timedelta(days=15), # 預設選在 15 天後
                min_value=today,                           # 限制不能選擇過去的日子
                max_value=today + datetime.timedelta(days=49) # 因為資料集最多只有 49~50 天的數據
            )

            # 計算距離出發還有幾天
            calc_days_left = (selected_date - today).days
            if calc_days_left == 0:
                calc_days_left = 1 # 避免出現 0 天 (假設今天買就是剩 1 天)

            st.info(f"💡 距離出發還有 **{calc_days_left}** 天")

            # 從歷史資料中撈出這個天數的平均票價
            if calc_days_left in df_eco['days_left'].values:
                estimated_price = df_eco[df_eco['days_left'] == calc_days_left]['price'].mean()
                st.success(f"💰 根據歷史數據，此時預訂的平均經濟艙票價約為： **₹ {estimated_price:,.0f}** INR")
            else:
                st.warning("⚠️ 歷史數據中缺乏此天數的資料。")
            # ==========================================


            # --- 4. 統計數據儀表板 ---
            st.markdown("---")
            st.subheader("📊 提前預訂價差統計")
            
            early_bird = df_eco[df_eco['days_left'] >= 45]['price'].mean()
            last_minute = df_eco[df_eco['days_left'] <= 2]['price'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("早鳥票均價 (提前45天+)", f"₹ {early_bird:,.0f}")
            col2.metric("最後一刻均價 (出發前2天)", f"₹ {last_minute:,.0f}")
            
            if pd.notnull(early_bird) and early_bird > 0:
                diff_ratio = last_minute / early_bird
                col3.metric("價差倍數", f"{diff_ratio:.2f} 倍")

            # --- 5. 繪製趨勢圖表 ---
            st.markdown("---")
            st.subheader("📈 預訂天數與票價趨勢圖")
            
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.lineplot(data=df_eco, x='days_left', y='price', 
                         color='#2ecc71', linewidth=2.5, label='Average Price Trend', ax=ax)

            ax.invert_xaxis()

            if 1 in df_eco['days_left'].values:
                last_day_price = df_eco[df_eco['days_left']==1]['price'].mean()
                max_price = df_eco['price'].max()
                ax.annotate('Last Minute (Highest Price)', 
                             xy=(1, last_day_price), 
                             xytext=(15, max_price * 0.8),
                             arrowprops=dict(facecolor='#e74c3c', shrink=0.05),
                             fontsize=12, fontweight='bold', color='#e74c3c')

            ax.set_title('Impact of Days Left on Ticket Price (Economy Class)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=12)
            ax.set_ylabel('Average Price (INR)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
