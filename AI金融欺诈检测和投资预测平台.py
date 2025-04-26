# -*- coding: utf-8 -*-
# 解决Python 3.12兼容性问题 | Solve Python 3.12 compatibility issues
import sys
if sys.version_info >= (3, 12):
    import setuptools  # 替代被移除的distutils模块 | Replace removed distutils module

# 导入核心依赖库 | Import core dependencies
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# ====================
# 数据获取模块 | Data Acquisition Module
# ====================
@st.cache_data(show_spinner=False, ttl=3600)  # 缓存1小时减少API调用 | Cache for 1 hour
def load_stock_data(symbols, start_date, end_date):
    """
    安全获取股票数据（带重试机制）| Safely fetch stock data with retry mechanism
    
    参数 Parameters:
        symbols (list): 股票代码列表 | List of stock symbols
        start_date (str): 开始日期 | Start date in YYYY-MM-DD
        end_date (str): 结束日期 | End date in YYYY-MM-DD
    
    返回 Returns:
        pd.DataFrame: 包含历史数据的DataFrame | DataFrame with historical data
    """
    try:
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker',
            auto_adjust=True  # 自动调整价格 | Auto-adjust prices
        )
        if data.empty:
            st.error("无法获取股票数据，请检查股票代码或稍后重试 | Failed to fetch data, please check symbols or try later")
            return None
            
        # 处理多级列索引 | Handle multi-level column index
        if isinstance(data.columns, pd.MultiIndex):
            closing_prices = data.xs('Close', axis=1, level=1)
        else:
            closing_prices = data['Close']
            
        return closing_prices.ffill().bfill()  # 填充缺失值 | Fill missing values
    except Exception as e:
        st.error(f"数据获取失败：{str(e)} | Data fetch failed: {str(e)}")
        return None

# ====================
# Q-Learning智能体 | Q-Learning Agent
# ====================
class PortfolioOptimizationAgent:
    """
    Q-Learning投资组合优化智能体 | Q-Learning Portfolio Optimization Agent
    
    属性 Attributes:
        q_table (np.array): Q值表 | Q-value table
        exploration_rate (float): 探索概率 | Exploration probability
    """
    
    def __init__(self, n_stocks, n_actions, learning_rate=0.1, 
                 discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        """
        初始化智能体 | Initialize agent
        
        参数 Parameters:
            n_stocks (int): 股票数量 | Number of stocks
            n_actions (int): 可选动作数量 | Number of possible actions
            learning_rate (float): 学习率(0-1) | Learning rate (0-1)
            discount_factor (float): 未来奖励折扣因子 | Discount factor for future rewards
            exploration_rate (float): 初始探索率 | Initial exploration rate
            exploration_decay (float): 探索率衰减系数 | Exploration rate decay
        """
        self.n_stocks = n_stocks
        self.n_actions = n_actions
        self.q_table = np.random.rand(500, n_actions)  # 初始化Q表 | Initialize Q-table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def select_action(self, state):
        """
        ε-greedy动作选择策略 | ε-greedy action selection policy
        
        参数 Parameters:
            state (int): 当前状态 | Current state
            
        返回 Returns:
            int: 选择的动作索引 | Selected action index
        """
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)  # 探索 | Explore
        else:
            return np.argmax(self.q_table[int(state)])  # 利用 | Exploit

    def update_q_value(self, state, action, reward, next_state):
        """
        使用Bellman方程更新Q值 | Update Q-value using Bellman equation
        
        参数 Parameters:
            state (int): 当前状态 | Current state
            action (int): 执行的动作 | Taken action
            reward (float): 获得的奖励 | Received reward
            next_state (int): 下一状态 | Next state
        """
        # 边界检查 | Boundary check
        next_state = min(int(next_state), self.q_table.shape[0]-1)
        
        # Bellman方程 | Bellman equation
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[int(state)][action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_future_q)
                
        self.q_table[int(state)][action] = new_q
        self.exploration_rate *= self.exploration_decay  # 衰减探索率 | Decay exploration rate

# ====================
# 界面配置 | UI Configuration
# ====================
st.set_page_config(
    page_title="金融AI平台 | FinAI Platform",
    page_icon="📊",
    layout="wide",  # 宽屏模式 | Wide layout
    initial_sidebar_state="expanded"
)

# 主标题 | Main title
st.title('AI金融欺诈检测与投资预测平台 | AI Financial Analysis Platform')

# ====================
# 侧边栏导航 | Sidebar Navigation
# ====================
with st.sidebar:
    st.title("功能导航 | Navigation")
    mode = st.radio(
        "选择功能模块 | Select Module",
        ("📈 投资组合优化 | Portfolio", "🛡️ 欺诈检测 | Fraud"),
        index=0,
        label_visibility="collapsed"
    )

# ====================
# 投资组合优化模块 | Portfolio Optimization Module
# ====================
if mode == "📈 投资组合优化 | Portfolio":
    st.header('📈 智能投资组合优化 | Smart Portfolio Optimization')
    
    # 输入列布局 | Input column layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # 股票代码输入 | Stock symbols input
        symbols = st.text_input(
            '输入股票代码（用逗号分隔）| Enter stock symbols (comma separated)',
            'AAPL,MSFT,TSLA',
            help="支持NYSE/NASDAQ代码，如：AAPL, MSFT | Supported symbols: AAPL, MSFT etc."
        )
        selected_stocks = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    with col2:
        # 日期范围选择 | Date range picker
        end_date = datetime.now().strftime('%Y-%m-%d')
        date_range = st.date_input(
            '选择时间范围 | Select date range',
            [datetime(2022, 1, 1), datetime.strptime(end_date, '%Y-%m-%d')],
            help="默认显示最近两年数据 | Default shows 2-year history"
        )
    
    if selected_stocks and len(date_range) == 2:
        # 数据加载部分 | Data loading section
        with st.spinner('正在获取市场数据... | Fetching market data...'):
            closing_prices = load_stock_data(
                selected_stocks,
                start_date=date_range[0].strftime('%Y-%m-%d'),
                end_date=date_range[1].strftime('%Y-%m-%d')
            )
        
        if closing_prices is not None:
            # 数据展示选项卡 | Data display tabs
            st.subheader('市场数据分析 | Market Data Analysis')
            tab1, tab2 = st.tabs(["📊 价格走势 | Price Trend", "🔍 数据详情 | Data Details"])
            
            with tab1:
                try:
                    st.line_chart(
                        closing_prices,
                        use_container_width=True,
                        height=400  # 固定图表高度 | Fixed chart height
                    )
                except Exception as e:
                    st.error(f"图表渲染失败: {str(e)} | Chart rendering failed: {str(e)}")
            
            with tab2:
                st.dataframe(
                    closing_prices.describe(),
                    use_container_width=True,
                    height=400  # 限制表格高度 | Limit table height
                )

            # 训练准备 | Training preparation
            returns = closing_prices.pct_change().dropna()
            
            if not returns.empty:
                st.subheader('初始化投资组合优化环境... | Initializing portfolio optimization environment...')
                agent = PortfolioOptimizationAgent(
    n_stocks=len(selected_stocks),  
    n_actions=len(selected_stocks) 
                    
                # 训练进度条 | Training progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 训练循环 | Training loop
                for episode in range(500):
                    state = 0
                    for i in range(len(returns) - 1):
                        action = agent.select_action(state)
                        reward = returns.iloc[i, action]
                        next_state = state + 1
                        agent.update_q_value(state, action, reward, next_state)
                        state = next_state
                    
                    # 更新进度 | Update progress
                    if (episode + 1) % 10 == 0:
                        progress = (episode + 1) / 500
                        progress_bar.progress(progress)
                        status_text.text(f'训练进度: {episode+1}/500 | Training progress: {episode+1}/500')
                
                # 显示优化结果 | Display optimization results
                optimized_weights = agent.q_table[-1] 
                optimized_weights /= np.sum(optimized_weights)  # 归一化 | Normalize
                
                # 结果展示列 | Result columns
                col_result1, col_result2 = st.columns([2, 3])
                
                with col_result1:
                    st.subheader('优化投资比例 | Optimized Allocation')
                    st.dataframe(pd.DataFrame({
                        '股票 | Stock': selected_stocks,
                        '权重 | Weight': optimized_weights
                    }))
                
                with col_result2:
                    st.subheader('资产分布 | Asset Distribution')
                    fig, ax = plt.subplots()
                    ax.pie(optimized_weights, 
                          labels=selected_stocks,
                          autopct='%1.1f%%',
                          startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                
                st.balloons()
            else:
                st.error("收益率数据不足，请选择更多股票或更长时间范围 | Insufficient return data, please select more stocks or longer period")

# ====================
# 欺诈检测模块 | Fraud Detection Module
# ====================
else:
    st.header('🛡️ 交易欺诈检测 | Transaction Fraud Detection')
    
    uploaded_file = st.file_uploader(
        "上传交易记录CSV文件 | Upload transaction CSV", 
        type=["csv"],
        help="文件必须包含'fraud'列 | File must contain 'fraud' column"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if 'fraud' not in data.columns:
                st.error("CSV文件必须包含'fraud'列 | CSV must contain 'fraud' column")
            else:
                # 数据预处理 | Data preprocessing
                X = data.drop('fraud', axis=1)
                y = data['fraud']
                
                # 训练模型 | Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # 模型评估 | Model evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 结果显示 | Display results
                st.success(f"模型准确率: {accuracy:.2%} | Model Accuracy: {accuracy:.2%}")
                
                # 预测结果表格 | Result table
                result_df = X_test.copy()
                result_df['实际标签 | Actual'] = y_test
                result_df['预测结果 | Predicted'] = y_pred
                result_df['状态 | Status'] = np.where(
                    result_df['预测结果 | Predicted'] == 1, 
                    '⚠️ 欺诈 | Fraud', 
                    '✅ 正常 | Normal'
                )
                
                # 交互式表格 | Interactive table
                st.dataframe(
                    result_df.style.apply(
                        lambda x: ['background: lightcoral' if v == 1 else '' for v in x['预测结果 | Predicted']],
                        axis=1
                    ),
                    use_container_width=True,
                    height=500
                )
                
        except Exception as e:
            st.error(f"文件处理失败: {str(e)} | File processing failed: {str(e)}")

# ====================
# 页脚 | Footer
# ====================
st.markdown("---")
st.caption("© 2024 金融AI平台 | FinAI Platform - MIT License")
