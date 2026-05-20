import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="機票價格分析", page_icon="✈️", layout="wide")
st.title("✈️ 印度航班：提前預訂票價趨勢分析")
st.markdown("這份儀表板分析了「距離出發的剩餘天數」對「經濟艙票價」的影響。")

# --- 2. 圖表外觀設定 (避免雲端主機中文字體亂碼) ---
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# --- 3. 讀取資料 ---
file_path = "Clean_Dataset.csv"

if not os.path.exists(file_path):
    st.error(f"❌ 找不到資料檔 `{file_path}`，請確定檔案有上傳到 GitHub！")
else:
    # 使用快取加速讀取
    @st.cache_data
    def load_data():
        df = pd.read_csv(file_path, low_memory=False)
        # 篩選經濟艙 (如果有這個欄位的話)
        if 'class' in df.columns:
            df_eco = df[df['class'].str.lower() == 'economy'].copy()
        else:
            df_eco = df.copy()
        
        # 確保價格欄位是數字
        if df_eco['price'].dtype == 'O':
            df_eco['price'] = pd.to_numeric(df_eco['price'].astype(str).str.replace(r'[,\"\s]', '', regex=True), errors='coerce')
            
        return df_eco.dropna(subset=['price', 'days_left'])

    with st.spinner("正在讀取資料與繪製圖表... (第一次載入可能需要幾秒鐘)"):
        df_eco = load_data()

        if 'days_left' not in df_eco.columns:
            st.error("❌ 資料集中找不到 `days_left` 欄位！請確認您的 CSV 檔案。")
        else:
            # --- 4. 統計數據儀表板 ---
            st.subheader("📊 提前預訂價差統計")
            
            # 計算早鳥與最後一刻的均價
            early_bird = df_eco[df_eco['days_left'] >= 45]['price'].mean()
            last_minute = df_eco[df_eco['days_left'] <= 2]['price'].mean()

            # 使用 Streamlit 內建的排版將數據並排顯示
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

            # 畫折線圖
            sns.lineplot(data=df_eco, x='days_left', y='price', 
                         color='#2ecc71', linewidth=2.5, label='Average Price Trend', ax=ax)

            # 反轉 X 軸，讓「1天」在最右邊，符合靠近出發日的直覺
            ax.invert_xaxis()

            # 標註最後一刻的價格
            if 1 in df_eco['days_left'].values:
                last_day_price = df_eco[df_eco['days_left']==1]['price'].mean()
                max_price = df_eco['price'].max()
                ax.annotate('Last Minute (Highest Price)', 
                             xy=(1, last_day_price), 
                             xytext=(15, max_price * 0.8),
                             arrowprops=dict(facecolor='#e74c3c', shrink=0.05),
                             fontsize=12, fontweight='bold', color='#e74c3c')

            # 設定圖表標籤 (使用英文避免 Streamlit 雲端缺乏中文字體的問題)
            ax.set_title('Impact of Days Left on Ticket Price (Economy Class)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Days Left Until Departure', fontsize=12)
            ax.set_ylabel('Average Price (INR)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 將 matplotlib 的圖表顯示在 Streamlit 網頁上
            st.pyplot(fig)
