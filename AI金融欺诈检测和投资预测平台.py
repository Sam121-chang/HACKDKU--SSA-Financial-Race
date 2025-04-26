# -*- coding: utf-8 -*-
# 在文件最开头添加（解决Python 3.12兼容性问题）
import sys
if sys.version_info >= (3, 12):
    import setuptools   # 替代被移除的distutils模块

# 导入必要的库 (Import required libraries)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 定义Q-Learning智能体 (Define the Q-Learning agent)
class PortfolioOptimizationAgent:
    def __init__(self, n_stocks, n_actions, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.n_stocks = n_stocks  # 股票数量 (Number of stocks)
        self.n_actions = n_actions  # 每个状态可采取的动作数量 (Number of actions per state)
        self.q_table = np.random.rand(500, n_actions)  # 初始化Q表 (Initialize Q-table with random values)
        self.learning_rate = learning_rate  # 学习率 (Learning rate)
        self.discount_factor = discount_factor  # 折扣因子 (Discount factor)
        self.exploration_rate = exploration_rate  # 探索率 (Exploration rate)
        self.exploration_decay = exploration_decay  # 探索率衰减 (Exploration decay)

    def select_action(self, state):
        # 根据探索率选择动作 (Choose action based on exploration rate)
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)  # 随机选择动作 (Random action)
        else:
            return np.argmax(self.q_table[int(state)])  # 选择Q值最大的动作 (Action with highest Q-value)

    def update_q_value(self, state, action, reward, next_state):
        # 更新Q值 (Update Q-value)
        if int(next_state) >= self.q_table.shape[0]:
            next_state = self.q_table.shape[0] - 1
        max_future_q = np.max(self.q_table[int(next_state)])  # 下一个状态的最大Q值 (Max Q-value for next state)
        current_q = self.q_table[int(state)][action]  # 当前Q值 (Current Q-value)
        # Q学习更新公式 (Q-learning update formula)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[int(state)][action] = new_q  # 更新Q表 (Update Q-table)
        self.exploration_rate *= self.exploration_decay  # 衰减探索率 (Decay exploration rate)

# 计算投资组合收益 (Calculate portfolio return)
def calculate_portfolio_return(weights, returns):
    return np.sum(weights * returns)  # 投资组合总收益 (Total portfolio return)

# 页面标题 (Page title)
st.title('AI金融欺诈检测与投资预测平台 (AI Financial Fraud Detection and Investment Prediction Platform)')

# 侧边栏选择功能 (Sidebar selection)
st.sidebar.title("功能选择 (Function Selection)")
mode = st.sidebar.radio("请选择功能 (Please select function)", ("📈 投资组合优化 (Portfolio Optimization)", "🛡️ 欺诈检测 (Fraud Detection)"))

# 投资组合优化模块 (Portfolio Optimization Module)
if mode == "📈 投资组合优化 (Portfolio Optimization)":
    st.header('📈 投资组合优化 (Portfolio Optimization)')

    # 股票代码输入 (Input stock symbols)
    stock_symbols = st.text_input('输入股票代码（用逗号分隔，如AAPL,MSFT,TSLA） (Enter stock symbols, comma separated)', 'AAPL,MSFT,TSLA')
    selected_stocks = [symbol.strip() for symbol in stock_symbols.split(',')]

    if selected_stocks:
        # 下载股票数据 (Download stock data)
        data = yf.download(selected_stocks, start='2022-01-01', end='2024-01-01')
        closing_prices = data['Close']  # 收盘价数据 (Closing prices)

        # 计算每日收益率 (Calculate daily returns)
        returns = closing_prices.pct_change().dropna()

        # 初始化Q-Learning智能体 (Initialize Q-Learning agent)
        st.subheader('初始化投资组合优化环境... (Initializing portfolio optimization environment...)')
        agent = PortfolioOptimizationAgent(n_stocks=len(selected_stocks), n_actions=len(selected_stocks))

        # 开始训练Q-Learning智能体 (Start training the agent)
        st.subheader('训练中... (Training...)')
        num_episodes = 500  # 训练轮数 (Number of training episodes)
        initial_state = 0

        for episode in range(num_episodes):
            state = initial_state
            for i in range(len(returns) - 1):
                action = agent.select_action(state)  # 选择动作 (Select action)
                reward = returns.iloc[i, action]  # 当前动作的收益 (Reward for the action)
                next_state = state + 1
                agent.update_q_value(state, action, reward, next_state)  # 更新Q值 (Update Q-value)
                state = next_state
            if (episode + 1) % 100 == 0:
                st.text(f'训练中...第 {episode+1}/{num_episodes} 次训练 (Training... {episode+1}/{num_episodes})')

        st.success('投资组合优化完成！(Portfolio optimization completed!)')

        # 生成优化后的投资组合 (Generate optimized investment portfolio)
        optimized_portfolio = {}
        for i, stock in enumerate(selected_stocks):
            optimized_portfolio[stock] = agent.q_table[-1][i]

        # 归一化投资比例 (Normalize investment ratios)
        total = sum(optimized_portfolio.values())
        for stock in optimized_portfolio:
            optimized_portfolio[stock] /= total

        # 显示优化结果 (Display optimized investment portfolio)
        st.subheader('投资优化组合结果 (Optimized Investment Portfolio)')
        st.table(pd.DataFrame(list(optimized_portfolio.items()), columns=["股票代码 (Stock)", "投资比例 (Investment Ratio)"]))

        # 绘制投资分布饼图 (Plot investment distribution pie chart)
        fig, ax = plt.subplots()
        ax.pie(optimized_portfolio.values(), labels=optimized_portfolio.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # 保持饼图为正圆形 (Ensure pie is a circle)
        st.pyplot(fig)

        # 显示彩带 (Show balloons)
        st.balloons()

# 欺诈检测模块 (Fraud Detection Module)
elif mode == "🛡️ 欺诈检测 (Fraud Detection)":
    st.header('🛡️ 欺诈检测 (Fraud Detection)')

    # 上传CSV文件 (Upload CSV file)
    uploaded_file = st.file_uploader("上传包含交易记录的CSV文件 (Upload CSV file with transactions)", type=["csv"])

    if uploaded_file is not None:
        # 读取数据 (Read data)
        data = pd.read_csv(uploaded_file)
        st.write("数据预览 (Data Preview):")
        st.dataframe(data.head())

        if 'fraud' not in data.columns:
            st.error('CSV文件必须包含“fraud”列 (CSV must include "fraud" column).')
        else:
            X = data.drop('fraud', axis=1)  # 特征 (Features)
            y = data['fraud']  # 标签 (Labels)

            # 拆分训练集和测试集 (Split into train and test sets)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 训练随机森林模型 (Train Random Forest model)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # 预测与评估 (Prediction and evaluation)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"欺诈检测模型训练完成！准确率：{accuracy:.2%} (Fraud detection model trained! Accuracy: {accuracy:.2%})")

            # 显示欺诈检测预测结果 (Display fraud detection results)
            st.subheader("欺诈检测预测结果 (Fraud Detection Predictions)")

            # 创建结果表格 (Create result dataframe)
            prediction_df = X_test.copy()
            prediction_df['真实是否欺诈 (Actual Fraud)'] = y_test.values
            prediction_df['预测是否欺诈 (Predicted Fraud)'] = y_pred
            prediction_df['预测结果 (Prediction Result)'] = np.where(
                prediction_df['预测是否欺诈 (Predicted Fraud)'] == 1, '欺诈', '正常')

            # 只显示重要字段 (Only show key columns)
            display_df = prediction_df[['amount', '真实是否欺诈 (Actual Fraud)', '预测是否欺诈 (Predicted Fraud)',
                                        '预测结果 (Prediction Result)']]
            st.write(display_df)
