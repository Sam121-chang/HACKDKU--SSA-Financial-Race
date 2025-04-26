import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------
# 投资组合优化 | Portfolio Optimization
# ---------------------

st.title('金融AI平台 - 投资组合优化与欺诈检测')

st.header('📈 投资组合优化')

# 用户选择股票
stock_symbols = st.text_input('输入股票代码（用逗号分隔，比如 AAPL, MSFT, GOOGL）', 'AAPL, MSFT, GOOGL')
selected_stocks = [s.strip() for s in stock_symbols.split(',')]

# 下载数据
@st.cache_data
def download_data(symbols):
    data = yf.download(symbols, start="2022-01-01", end="2024-01-01")['Adj Close']
    return data

data = download_data(selected_stocks)

if not data.empty:
    st.subheader('股票价格走势')
    st.line_chart(data)

    # 计算日收益率
    returns = data.pct_change().dropna()

    # 深度强化学习代理
    class PortfolioOptimizationAgent(nn.Module):
        def __init__(self, n_stocks, n_actions):
            super(PortfolioOptimizationAgent, self).__init__()
            self.fc1 = nn.Linear(n_stocks, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, n_actions)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return torch.softmax(self.fc3(x), dim=-1)

    # 初始化Agent
    agent = PortfolioOptimizationAgent(
        n_stocks=len(selected_stocks),  
        n_actions=len(selected_stocks)
    )

    # 训练进度条 | Training progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 简单训练
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    num_epochs = 200
    returns_tensor = torch.tensor(returns.values, dtype=torch.float)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        allocations = agent(returns_tensor)
        portfolio_returns = (returns_tensor * allocations).sum(dim=1)
        loss = -portfolio_returns.mean()  # 最大化收益
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            progress_bar.progress(epoch / num_epochs)
            status_text.text(f"训练中... Epoch {epoch}/{num_epochs}")

    progress_bar.progress(1.0)
    status_text.text("训练完成！")

    # 显示最终分配
    final_allocations = agent(returns_tensor).mean(dim=0).detach().numpy()
    allocation_df = pd.DataFrame({
        'Stock': selected_stocks,
        'Allocation': final_allocations
    })

    st.subheader('最终投资组合分配')
    st.bar_chart(allocation_df.set_index('Stock'))

# ---------------------
# 金融欺诈检测 | Fraud Detection
# ---------------------

st.header('🔒 金融欺诈检测')

uploaded_file = st.file_uploader("上传金融交易数据 (CSV格式)", type=['csv'])

if uploaded_file is not None:
    fraud_data = pd.read_csv(uploaded_file)

    st.subheader('数据预览')
    st.write(fraud_data.head())

    # 选择用于建模的数值型特征
    numeric_cols = fraud_data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_features = st.multiselect('选择特征用于欺诈检测', numeric_cols, default=numeric_cols)

        if selected_features:
            X = fraud_data[selected_features]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Isolation Forest
            model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            preds = model.fit_predict(X_scaled)
            fraud_data['Anomaly'] = preds

            st.subheader('检测结果')
            fig, ax = plt.subplots()
            sns.scatterplot(x=X[selected_features[0]], y=X[selected_features[1]], hue=fraud_data['Anomaly'], palette={1: 'blue', -1: 'red'}, ax=ax)
            st.pyplot(fig)

            st.subheader('异常交易')
            anomalies = fraud_data[fraud_data['Anomaly'] == -1]
            st.write(anomalies)

    else:
        st.error("未找到数值型特征，无法进行欺诈检测。")
