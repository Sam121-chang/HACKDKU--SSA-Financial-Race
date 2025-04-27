# -*- coding: utf-8 -*-
import sys
if sys.version_info >= (3, 12):
    import setuptools   # 替代被移除的distutils模块

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit 页面基本设置
st.set_page_config(page_title="智能投资平台", layout="wide")

st.title('智能投资平台')
st.markdown('结合强化学习投资组合优化 + 欺诈检测 + 投资心情打卡。')

# 侧边栏选择模块
mode = st.sidebar.selectbox(
    "选择功能模块",
    ("📈 投资组合优化", "🛡️ 欺诈检测", "📝 投资心情打卡")
)

# ============================  投资组合优化模块 ============================ #
if mode == "📈 投资组合优化":
    st.header('📈 投资组合优化')

    # 用户输入股票代码
    tickers_input = st.text_input('输入股票代码（用逗号分隔，如 AAPL,MSFT,GOOG）', 'AAPL,MSFT,GOOG')
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

    # 选择训练轮数
    episodes = st.slider('训练轮数（越多越精确，但耗时更长）', 100, 5000, 1000, step=100)

    # 确认按钮
    if st.button('开始优化'):

        if len(tickers) < 2:
            st.warning('请至少输入两个有效的股票代码。')
        else:
            st.success('正在下载数据并进行训练，请稍候...')

            # 下载股票数据
            data = yf.download(tickers, start="2020-01-01", end="2024-12-31")['Adj Close']
            returns = data.pct_change().dropna()

            # 初始化Q-learning元素
            n_assets = len(tickers)
            n_actions = 100  # 离散动作数量
            q_table = np.zeros((n_actions,) * n_assets)
            learning_rate = 0.1
            discount_factor = 0.95
            epsilon = 0.1  # 探索率

            actions = np.linspace(0, 1, n_actions)

            # 简单随机环境模拟
            def get_reward(weights, returns):
                weights = np.array(weights)
                if not np.isclose(np.sum(weights), 1):
                    return -100  # 惩罚，不合法
                port_return = np.dot(returns.mean(), weights)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                if port_volatility == 0:
                    return -100
                sharpe_ratio = port_return / port_volatility
                return sharpe_ratio

            # 定义Q-learning更新函数
            def q_learning_update(state, next_state, reward, q_table, learning_rate, discount_factor):
                best_next_q_value = np.max(q_table[next_state])  # 找到下一个状态的最大Q值
                q_table[state] = q_table[state] + learning_rate * (reward + discount_factor * best_next_q_value - q_table[state])
                return q_table

            # 在 Q-learning 训练中使用更新的逻辑
            for episode in range(episodes):
                # 随机选择一个状态
                state = tuple([random.randint(0, n_actions - 1) for _ in range(n_assets)])
                weights = [actions[i] for i in state]
                weights = np.array(weights) / np.sum(weights)
        
                # 获取奖励
                reward = get_reward(weights, returns)
            
                # 随机选择下一个状态
                next_state = tuple([random.randint(0, n_actions - 1) for _ in range(n_assets)])
            
                # 使用新的更新规则
                q_table = q_learning_update(state, next_state, reward, q_table, learning_rate, discount_factor)

            # 选择最佳的权重
            best_state = np.argmax(q_table, axis=0)  # 每个资产的最佳状态
            best_weights = np.array([actions[i] for i in best_state])
            best_weights = best_weights / np.sum(best_weights)  # 确保权重和为1

            # 显示优化结果
            st.subheader('投资组合推荐')

            fig, ax = plt.subplots()
            ax.pie(best_weights, labels=tickers, autopct='%1.1f%%', startangle=90, counterclock=False)
            ax.axis('equal')
            st.pyplot(fig)

            # 生成投资建议报告
            st.subheader('📄 投资建议报告')

            report_md = "### 🏦 投资建议\n\n"
            report_md += "**推荐股票及对应投资比例：**\n\n"
            for ticker, weight in zip(tickers, best_weights):
                report_md += f"- **{ticker}**：{weight*100:.2f}%\n"
            report_md += "\n"
            report_md += "> **总结：** 本投资组合基于强化学习优化，旨在在风险控制下追求稳健收益，适合中长期投资者参考。\n"

            st.markdown(report_md)

            # 补充：下载按钮
            csv_download = pd.DataFrame({'股票': tickers, '投资比例': best_weights})
            st.download_button(label="下载投资方案CSV", data=csv_download.to_csv(index=False).encode('utf-8'), file_name='portfolio_recommendation.csv', mime='text/csv')


# ============================  欺诈检测模块 ============================ #
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

            display_df = prediction_df[['真实是否欺诈', '预测是否欺诈', '预测结果']]
            st.write(display_df)

# ============================  投资心情打卡模块 ============================ #
elif mode == "📝 投资心情打卡":
    st.header("📝 投资心情打卡")

    mood = st.radio(
        "今天的投资心情如何？",
        ("🚀 信心满满", "🌤️ 谨慎观望", "🌧️ 略显担忧", "☔ 极度恐慌")
    )

    notes = st.text_area("有什么想记录的吗？（可选）", "")

    if st.button("提交打卡"):
        st.success(f"打卡成功！你的今日心情是：{mood}")
        if notes:
            st.info(f"备注内容：{notes}")
