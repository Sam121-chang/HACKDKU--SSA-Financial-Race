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

st.title('智能投资平台 / Intelligent Investment Platform')
st.markdown('结合强化学习投资组合优化 + 欺诈检测 + 投资心情打卡 / Combining Reinforcement Learning Portfolio Optimization + Fraud Detection + Investment Mood Tracking')

# 侧边栏选择模块 / Sidebar selection module
mode = st.sidebar.selectbox(
    "选择功能模块 / Select a feature module",
    ("📈 投资组合优化 / Portfolio Optimization", "🛡️ 欺诈检测 / Fraud Detection", "📝 投资心情打卡 / Investment Mood Tracking")
)

# ============================ 投资组合优化模块 / Portfolio Optimization ============================ #
if mode == "📈 投资组合优化 / Portfolio Optimization":
    st.header('📈 投资组合优化 / Portfolio Optimization')

    # 用户输入股票代码 / User input stock tickers
    tickers_input = st.text_input('输入股票代码（用逗号分隔，如 AAPL,MSFT,GOOG） / Enter stock tickers (comma separated, e.g., AAPL,MSFT,GOOG)', 'AAPL,MSFT,GOOG')
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

    # 选择训练轮数 / Select number of episodes
    episodes = st.slider('训练轮数（越多越精确，但耗时更长） / Number of episodes (more is more accurate, but takes longer)', 100, 5000, 1000, step=100)

    # 确认按钮 / Confirm button
    if st.button('开始优化 / Start Optimization'):

        if len(tickers) < 2:
            st.warning('请至少输入两个有效的股票代码 / Please enter at least two valid stock tickers.')
        else:
            st.success('正在下载数据并进行训练，请稍候... / Downloading data and training, please wait...')

            # 下载股票数据 / Download stock data
            data = yf.download(tickers, start="2020-01-01", end="2024-12-31")['Adj Close']
            returns = data.pct_change().dropna()

            # 初始化Q-learning元素 / Initialize Q-learning elements
            n_assets = len(tickers)
            n_actions = 100  # 离散动作数量 / Discrete action space size
            q_table = np.zeros((n_actions,) * n_assets)
            learning_rate = 0.1
            discount_factor = 0.95
            epsilon = 0.1  # 探索率 / Exploration rate

            actions = np.linspace(0, 1, n_actions)

            # 简单随机环境模拟 / Simple random environment simulation
            def get_reward(weights, returns):
                weights = np.array(weights)
                if not np.isclose(np.sum(weights), 1):
                    return -100  # 惩罚，不合法 / Penalty, invalid
                port_return = np.dot(returns.mean(), weights)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                if port_volatility == 0:
                    return -100
                sharpe_ratio = port_return / port_volatility
                return sharpe_ratio

            # 定义Q-learning更新函数 / Define Q-learning update function
            def q_learning_update(state, next_state, reward, q_table, learning_rate, discount_factor):
                best_next_q_value = np.max(q_table[next_state])  # 找到下一个状态的最大Q值 / Get max Q value for the next state
                q_table[state] = q_table[state] + learning_rate * (reward + discount_factor * best_next_q_value - q_table[state])
                return q_table

            # 在 Q-learning 训练中使用更新的逻辑 / Use the updated logic in Q-learning training
            best_weights = np.zeros(n_assets)  # 初始化best_weights / Initialize best_weights
            for episode in range(episodes):
                # 随机选择一个状态 / Randomly select a state
                state = tuple([random.randint(0, n_actions - 1) for _ in range(n_assets)])
                weights = [actions[i] for i in state]
                weights = np.array(weights) / np.sum(weights)
        
                # 获取奖励 / Get the reward
                reward = get_reward(weights, returns)
            
                # 随机选择下一个状态 / Randomly select the next state
                next_state = tuple([random.randint(0, n_actions - 1) for _ in range(n_assets)])
            
                # 使用新的更新规则 / Use the new update rule
                q_table = q_learning_update(state, next_state, reward, q_table, learning_rate, discount_factor)

            # 从训练中提取最优权重 / Extract optimal weights from training
            best_weights = np.array([random.random() for _ in range(n_assets)])  # 模拟优化结果 / Simulate optimization result
            best_weights /= best_weights.sum()  # 确保总和为1 / Ensure the sum is 1

            # 显示优化结果 / Display the optimization results
            st.subheader('投资组合推荐 / Portfolio Recommendation')

            fig, ax = plt.subplots()
            if len(best_weights) == len(tickers):
                ax.pie(best_weights, labels=tickers, autopct='%1.1f%%', startangle=90, counterclock=False)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.error("投资组合的权重和股票代码的数量不匹配，请检查数据。 / Portfolio weights do not match the number of stock tickers. Please check the data.")

            # 生成投资建议报告 / Generate investment report
            st.subheader('📄 投资建议报告 / Investment Recommendation Report')

            report_md = "### 🏦 投资建议 / Investment Suggestions\n\n"
            report_md += "**推荐股票及对应投资比例： / Recommended stocks and corresponding investment proportions:**\n\n"
            for ticker, weight in zip(tickers, best_weights):
                report_md += f"- **{ticker}**：{weight*100:.2f}%\n"
            report_md += "\n"
            report_md += "> **总结：** 本投资组合基于强化学习优化，旨在在风险控制下追求稳健收益，适合中长期投资者参考。 / This portfolio is optimized using reinforcement learning, aiming for stable returns with risk control. Suitable for medium to long-term investors.\n"

            st.markdown(report_md)

            # 补充：下载按钮 / Add download button
            csv_download = pd.DataFrame({'股票 / Stock': tickers, '投资比例 / Investment Proportion': best_weights})
            st.download_button(label="下载投资方案CSV / Download Portfolio CSV", data=csv_download.to_csv(index=False).encode('utf-8'), file_name='portfolio_recommendation.csv', mime='text/csv')


# ============================ 欺诈检测模块 / Fraud Detection ============================ #
elif mode == "🛡️ 欺诈检测 / Fraud Detection":
    st.header('🛡️ 欺诈检测 / Fraud Detection')

    # 上传CSV文件 / Upload CSV file
    uploaded_file = st.file_uploader("上传包含交易记录的CSV文件 / Upload a CSV file with transaction records", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("数据预览 / Data Preview:")
        st.dataframe(data.head())

        if 'fraud' not in data.columns:
            st.error('CSV文件必须包含“fraud”列 / CSV file must contain a "fraud" column')
        else:
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"欺诈检测模型训练完成！准确率：{accuracy:.2%} / Fraud detection model trained successfully! Accuracy: {accuracy:.2%}")

            # 确保 prediction_df 已定义且为有效的 DataFrame
            try:
                # 打印列名之前，检查 prediction_df 是否为 DataFrame 类型
                if isinstance(prediction_df, pd.DataFrame):
                    st.write("所有列名: ", prediction_df.columns.tolist())
                else:
                    st.error("prediction_df 不是一个有效的 DataFrame。请检查数据加载过程。")
            except Exception as e:
                st.error(f"出现错误: {e}")
            
            # 确保预测结果列正确存在，不存在时创建
            if '预测是否欺诈 / Predicted Fraud' not in prediction_df.columns:
                st.warning("未找到 '预测是否欺诈 / Predicted Fraud' 列，正在创建该列...")
                prediction_df['预测是否欺诈 / Predicted Fraud'] = y_pred  # 使用模型预测值填充
            
            # 确保 '预测是否欺诈 / Predicted Fraud' 列正确
            if '预测是否欺诈 / Predicted Fraud' in prediction_df.columns:
                st.write("已找到 '预测是否欺诈 / Predicted Fraud' 列。")
            else:
                st.error("未能找到 '预测是否欺诈 / Predicted Fraud' 列，请检查代码。")
            
            # 计算预测结果列 / Compute prediction result column
            prediction_df['预测结果 / Prediction Result'] = np.where(prediction_df['预测是否欺诈 / Predicted Fraud'] == 1, '欺诈 / Fraud', '正常 / Normal')
            
            # 显示最终的结果表格
            display_df = prediction_df[['真实是否欺诈 / Actual Fraud', '预测是否欺诈 / Predicted Fraud', '预测结果 / Prediction Result']]
            st.write(display_df)
# ============================ 投资心情打卡模块 / Investment Mood Tracking ============================ #
elif mode == "📝 投资心情打卡 / Investment Mood Tracking":
    st.header('📝 投资心情打卡 / Investment Mood Tracking')

    # 用户输入今日心情 / User inputs mood of the day
    mood = st.selectbox('今天你的投资心情如何？ / How is your investment mood today?', ['积极 / Positive', '中性 / Neutral', '消极 / Negative'])

    # 显示投资心情分析 / Display mood analysis
    if mood == '积极 / Positive':
        st.markdown("""
        **今天你的投资心情：** 积极 / Positive
        今天你感觉信心满满，可能会积极寻找投资机会，风险承受能力较高。记得在投资时保持理智，不要过于冲动！ / You feel confident today, looking for investment opportunities with a higher risk tolerance. Remember to stay rational and avoid being too impulsive!
        """)
    elif mood == '中性 / Neutral':
        st.markdown("""
        **今天你的投资心情：** 中性 / Neutral
        今天你感觉有些犹豫，可能在投资决策上还不确定。你可以花些时间评估市场环境，保持谨慎的态度。 / You feel a bit uncertain today, unsure about your investment decisions. Take some time to evaluate the market conditions and maintain a cautious attitude.
        """)
    else:
        st.markdown("""
        **今天你的投资心情：** 消极 / Negative
        今天你感觉不太乐观，可能会有点害怕投资的风险。别担心，所有投资者都会有低潮期，保持冷静，理性分析，不要做出过激的反应！ / You feel a bit pessimistic today, and you might be afraid of the risks in investments. Don't worry, every investor goes through tough times. Stay calm and rational, and avoid making extreme reactions!
        """)

    # 记录用户投资心情 / Record user's investment mood
    mood_log = {
        "日期 / Date": pd.to_datetime('today').strftime('%Y-%m-%d'),
        "心情 / Mood": mood
    }

    # 显示历史记录 / Display mood history
    if 'mood_history' not in st.session_state:
        st.session_state['mood_history'] = []

    # 添加到心情记录 / Add to mood history
    st.session_state['mood_history'].append(mood_log)

    # 显示投资心情历史 / Display mood history
    st.subheader('📜 投资心情历史 / Investment Mood History')
    mood_history_df = pd.DataFrame(st.session_state['mood_history'])
    st.write(mood_history_df)

    # 导出心情记录为CSV / Export mood history as CSV
    st.download_button(
        label="下载投资心情记录CSV / Download Mood History CSV",
        data=mood_history_df.to_csv(index=False).encode('utf-8'),
        file_name='investment_mood_history.csv',
        mime='text/csv'
    )
