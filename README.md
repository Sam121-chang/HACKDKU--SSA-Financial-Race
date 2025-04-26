# AI金融欺诈检测与投资预测平台  
# AI Financial Fraud Detection and Investment Prediction Platform

## 📖 项目简介 | Project Description

本项目是一个基于机器学习（Random Forest）和强化学习（Q-Learning）的金融应用平台，支持：
- **投资组合优化**：通过Q-Learning智能体优化多支股票的投资比例。
- **金融欺诈检测**：使用随机森林模型检测交易数据中的潜在欺诈行为。

This project is a financial application platform based on Machine Learning (Random Forest) and Reinforcement Learning (Q-Learning), supporting:
- **Portfolio Optimization**: Optimize stock investment ratios using a Q-Learning agent.
- **Fraud Detection**: Detect potential frauds in transaction data using a Random Forest model.

---

## 🚀 快速开始 | Quick Start

### 安装依赖 | Install Dependencies

```bash
pip install streamlit yfinance pandas numpy scikit-learn matplotlib
```

### 运行项目 | Run the Project

```bash
streamlit run app.py
```

> 请确保你将以上代码保存为 `app.py` 文件。  
> Make sure you save the code above as an `app.py` file.

---

## 📈 功能模块 | Features

### 1. 投资组合优化 | Portfolio Optimization
- 输入感兴趣的股票代码（如：AAPL, MSFT, TSLA）
- 使用Q-Learning方法学习和分配投资比例
- 可视化优化后的投资组合及分布

### 2. 金融欺诈检测 | Financial Fraud Detection
- 上传包含交易记录的CSV文件（文件中必须包含 `fraud` 列）
- 自动训练随机森林分类器
- 预测交易是否为欺诈行为，准确率评估与高亮显示结果

---

## 📂 文件结构 | File Structure

```
├── app.py             # 主程序 Main application
├── README.md          # 项目说明 Project description
└── requirements.txt   # （可选）依赖列表 Optional dependency list
```

---

## 📋 注意事项 | Notes

- 欺诈检测模块要求上传的CSV文件中必须包含 `fraud` 字段。
- 投资组合优化模块目前基于历史数据，不代表未来收益，仅供学习参考。
- 本平台为教育与展示用途，**不构成投资建议**。

---

## 📄 许可证 | License

This project is licensed under the MIT License.  
此项目采用 MIT License 开源协议。

---
