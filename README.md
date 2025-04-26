# AI 金融欺诈检测和投资预测平台  
**AI Financial Fraud Detection & Investment Prediction Platform**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hackdku--ssa-financial-race-ayf7srgyzxz9idatedmpja.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[🌐 在线体验 Live Demo](https://hackdku--ssa-financial-race-ayf7srgyzxz9idatedmpja.streamlit.app) 

> 2025DKU 黑客松参赛作品 | 2025 DKU Hackathon Entry

## 🚀 核心功能 | Core Features
### 技术创新 | Technical Innovation
- **混合智能系统**：结合 Q-Learning 优化算法与随机森林检测模型，实现智能的金融决策与欺诈识别。
  **Hybrid AI System**: Integrates Q-Learning optimization algorithm and Random Forest detection model to achieve intelligent financial decision-making and fraud identification.
- **动态预测引擎**：基于实时市场数据驱动，提供精准的投资策略，适应市场变化。
  **Dynamic Prediction**: Driven by real-time market data, it provides accurate investment strategies and adapts to market changes.
- **多维度风控**：高效识别欺诈交易，经测试识别准确率超过 92%，保障资金安全。
  **Risk Control**: Effectively identifies fraudulent transactions, with a tested recognition accuracy rate of over 92% to safeguard fund security.

### 用户体验 | User Experience
- **交互式可视化**：通过资产分布饼图和风险热力图，直观展示投资组合和风险状况，便于用户理解和分析。
  **Interactive Visualization**: Presents investment portfolios and risk status intuitively through portfolio pie charts and risk heatmaps, facilitating user understanding and analysis.
- **智能报告生成**：支持一键下载投资建议书，为用户提供专业的投资参考和建议。
  **Smart Reporting**: Enables one-click download of investment proposals, providing users with professional investment references and suggestions.
- **云端就绪**：提供开箱即用的部署方案，方便快捷地将应用部署到云端，随时随地访问。
  **Cloud Ready**: Offers an out-of-the-box deployment solution, enabling quick and convenient deployment of the application to the cloud for access anytime, anywhere.

## ⚙️ 安装指南 | Installation
```bash
# 克隆仓库 | Clone repo
git clone https://github.com/yourusername/hackdku--ssa-financial-race.git
cd hackdku--ssa-financial-race

# 创建虚拟环境 | Create venv (Python 3.10)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 安装依赖 | Install dependencies
pip install -r requirements.txt

# 启动应用 | Launch app
streamlit run main.py 
```

希望这份修改后的 README 能满足你的需求。如果你还有其他想法或需要进一步的修改，请随时告诉我。 
## 🎮 使用指南 | User Guide

### 📈 投资组合优化 | Portfolio Optimization
1. 输入股票代码（例：AAPL,TSLA）  
   Enter stock symbols (e.g. AAPL,TSLA)
2. 调节训练轮数（推荐1000轮）  
   Adjust training episodes (recommend 1000)
3. 查看优化结果图表  
   View optimization results
4. 下载投资建议  
   Download report

### 🛡️ 欺诈检测 | Fraud Detection
1. 上传交易记录CSV  
   Upload transaction CSV
2. 查看检测结果  
   View detection results
3. 导出风险报告  
   Export risk report

## 📜 开源协议 | License
本项目采用 **[MIT License](https://opensource.org/licenses/MIT)** 授权

完整协议见 [LICENSE](LICENSE) 文件

## 🌟 技术亮点 | Technical Highlights
| 功能模块          | 核心技术                 | 性能指标          |
|------------------|--------------------------|------------------|
| 投资组合优化      | Q-Learning算法           | 优化速度<15s     |
| Portfolio Opt    | Q-Learning Algorithm     | Speed <15s       |
| 欺诈检测         | 随机森林分类             | 准确率92.4%      |
| Fraud Detection  | Random Forest Classifier | Accuracy 92.4%   |
```

