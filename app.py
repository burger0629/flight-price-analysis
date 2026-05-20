import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 網頁版專屬設定 ---
st.set_page_config(page_title="機票價格分析", page_icon="✈️")
st.title("✈️ 2022 印度航班票價分析")
st.markdown("分析平日、週末與灑紅節連假的票價差異。")

# --- 解決雲端主機中文字體顯示問題 ---
# 雲端主機通常沒有微軟正黑體，我們改用英文標籤或預設字體以防亂碼
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# --- 讀取資料 ---
file_path = "Clean_Dataset.csv" # 網頁版檔案會在同一個資料夾

if not os.path.exists(file_path):
    st.error(f"❌ 找不到資料檔 `{file_path}`，請確定檔案有上傳到 GitHub！")
else:
    st.success("✅ 成功讀取資料集，正在繪製圖表...")
    
    # 使用 Streamlit 快取功能加速讀取
    @st.cache_data
    def load_data():
        df = pd.read_csv(file_path, low_memory=False)
        df['price'] = pd.to_numeric(
            df['price'].astype(str).str.replace(r'[,\"\s\t\n]', '', regex=True),
            errors='coerce'
        )
        df = df.dropna(subset=['price'])
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        # 篩選經濟艙
        return df[df['class'].str.lower() == 'economy']
        
    df = load_data()

    # --- 標記時段 ---
    def get_day_type(d):
        if pd.Timestamp('2022-03-17') <= d <= pd.Timestamp('2022-03-20'):
            return 'Holi Festival (灑紅節)'
        elif d.weekday() >= 5:
            return 'Weekend (週末)'
        else:
            return 'Weekday (平日)'
    
    df['時段類型'] = df['date'].apply(get_day_type)

    # --- 統計分析 ---
    order = ['Weekday (平日)', 'Weekend (週末)', 'Holi Festival (灑紅節)']
    df_stats = df.groupby('時段類型')['price'].mean().reindex(order).reset_index()
    df_stats.columns = ['時段類型', '平均票價']
    
    st.write("📊 **統計數據報告：**")
    st.dataframe(df_stats) # 在網頁上顯示表格

    # --- 繪製圖表 ---
    fig, ax_fig = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=df_stats, x='時段類型', y='平均票價', hue='時段類型',
                palette='coolwarm', dodge=False, ax=ax_fig)
    if ax_fig.get_legend():
        ax_fig.get_legend().remove()

    base_series = df_stats[df_stats['時段類型'] == 'Weekday (平日)']['平均票價']
    base_val = base_series.values[0] if not base_series.empty else 0
    y_max = df_stats['平均票價'].max()
    label_offset = y_max * 0.02 if pd.notnull(y_max) else 50

    for i, row in df_stats.iterrows():
        v = row['平均票價']
        if pd.notnull(v) and v > 0:
            if row['時段類型'] == 'Weekday (平日)':
                label = "Base"
            elif base_val > 0:
                diff_pct = ((v - base_val) / base_val) * 100
                label = f"+{diff_pct:.1f}%"
            else:
                label = "N/A"
            ax_fig.text(i, v + label_offset, label, ha='center', va='bottom',
                     fontsize=12, fontweight='bold', color='black')

    ax_fig.set_title('2022 India Flight Price Analysis', fontsize=16, fontweight='bold')
    ax_fig.set_xlabel('Time Period', fontsize=12)
    ax_fig.set_ylabel('Average Price (INR)', fontsize=12)
    ax_fig.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 🌟 關鍵：將圖表輸出到 Streamlit 網頁上
    st.pyplot(fig)