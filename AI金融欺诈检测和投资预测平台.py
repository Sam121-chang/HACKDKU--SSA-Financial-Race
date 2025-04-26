# -*- coding: utf-8 -*-
import sys
if sys.version_info >= (3, 12):
    import setuptools   # 替代被移除的distutils模块

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 定义Q-Learning智能体
class PortfolioOptimizationAgent:
    def __init__(self, n_stocks, n_actions, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.n_stocks = n_stocks
        self.n_actions = n_actions
        self.q_table = np.random.rand(500, n_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[int(state)])

    def update_q_value(self, state, action, reward, next_state):
        if int(next_state) >= self.q_table.shape[0]:
            next_state = self.q_table.shape[0] - 1
        max_future_q = np.max(self.q_table[int(next_state)])
        current_q = self.q_table[int(state)][action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[int(state)][action] = new_q
        self.exploration_rate *= self.exploration_decay

# 计算投资组合收益
def calculate_portfolio_return(weights, returns):
    return np.sum(weights * returns)

# 页面标题
st.title('AI金融欺诈检测与投资预测平台')

# 侧边栏选择功能
st.sidebar.title("功能选择")
mode = st.sidebar.radio("请选择功能", ("📈 投资组合优化", "🛡️ 欺诈检测"))

# 投资组合优化模块
if mode == "📈 投资组合优化":
    st.header('📈 投资组合优化')

    # 股票代码输入
    stock_symbols = st.text_input('输入股票代码（用逗号分隔，如AAPL,MSFT,TSLA）', 'AAPL,MSFT,TSLA')
    selected_stocks = [symbol.strip() for symbol in stock_symbols.split(',')]

    if selected_stocks:
        # 下载股票数据
        data = yf.download(selected_stocks, start='2022-01-01', end='2024-01-01')
        closing_prices = data['Close']

        # 计算每日收益率
        returns = closing_prices.pct_change().dropna()

        # 初始化Q-Learning智能体
        st.subheader('初始化投资组合优化环境...')
        agent = PortfolioOptimizationAgent(n_stocks=len(selected_stocks), n_actions=len(selected_stocks))

        # 开始训练Q-Learning智能体
        st.subheader('训练中...')
        num_episodes = 500
        initial_state = 0

        for episode in range(num_episodes):
            state = initial_state
            for i in range(len(returns) - 1):
                action = agent.select_action(state)
                reward = returns.iloc[i, action]
                next_state = state + 1
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
            if (episode + 1) % 100 == 0:
                st.text(f'训练中...第 {episode+1}/{num_episodes} 次训练')

        st.success('投资组合优化完成！')

        # 生成优化后的投资组合
        optimized_portfolio = {}
        for i, stock in enumerate(selected_stocks):
            optimized_portfolio[stock] = agent.q_table[-1][i]

        # 归一化投资比例
        total = sum(optimized_portfolio.values())
        for stock in optimized_portfolio:
            optimized_portfolio[stock] /= total

        # 显示优化结果
        st.subheader('投资优化组合结果')
        st.table(pd.DataFrame(list(optimized_portfolio.items()), columns=["股票代码", "投资比例"]))

        # 绘制柱状图
        fig, ax = plt.subplots()
        ax.bar(optimized_portfolio.keys(), optimized_portfolio.values())
        ax.set_xlabel('股票代码')
        ax.set_ylabel('投资比例')
        ax.set_title('投资优化组合 (Optimized Investment Portfolio)')
        st.pyplot(fig)

        # 绘制历史表现图表
        st.subheader('投资组合的历史表现')

        # 模拟投资组合的回报
        portfolio_weights = np.array(list(optimized_portfolio.values()))
        portfolio_returns = returns.dot(portfolio_weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        fig, ax = plt.subplots()
        ax.plot(cumulative_returns, label='投资组合累计回报')
        ax.set_xlabel('日期')
        ax.set_ylabel('累计回报')
        ax.set_title('投资组合的历史表现')
        st.pyplot(fig)

        # 计算并显示风险评估指标
        st.subheader('投资组合的风险评估')

        # 波动率
        volatility = portfolio_returns.std() * np.sqrt(252)
        st.write(f"年化波动率: {volatility:.2%}")

        # 夏普比率
        risk_free_rate = 0.03
        sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / volatility
        st.write(f"夏普比率: {sharpe_ratio:.2f}")

# 欺诈检测模块
elif mode == "🛡️ 欺诈检测":
    st.header('🛡️ 欺诈检测')

    # 上传CSV文件
    uploaded_file = st.file_uploader("上传包含交易记录的CSV文件", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("数据预览:")
        st.dataframe(data.head())

        if 'fraud' not in data.columns:
            st.error('CSV文件必须包含“fraud”列')
        else:
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"欺诈检测模型训练完成！准确率：{accuracy:.2%}")

            # 显示欺诈检测预测结果
            st.subheader("欺诈检测预测结果")

            prediction_df = X_test.copy()
            prediction_df['真实是否欺诈'] = y_test.values
            prediction_df['预测是否欺诈'] = y_pred
            prediction_df['预测结果'] = np.where(prediction_df['预测是否欺诈'] == 1, '欺诈', '正常')

            display_df = prediction_df[['amount', '真实是否欺诈', '预测是否欺诈', '预测结果']]
            st.write(display_df)
