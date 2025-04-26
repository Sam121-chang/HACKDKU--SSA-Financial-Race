# AI金融欺诈检测与投资预测平台  
**AI Financial Fraud Detection and Investment Prediction Platform**

## 📚 项目简介 | Project Introduction
本项目基于Python开发，通过Streamlit搭建交互式界面，集成了：
- 📈 投资组合优化（使用Q-Learning强化学习）
- 🛡️ 金融欺诈检测（使用随机森林分类器）

This project, developed in Python and Streamlit, includes:
- 📈 Portfolio Optimization (via Q-Learning Reinforcement Learning)
- 🛡️ Financial Fraud Detection (via Random Forest Classifier)

---

## 🛠️ 使用到的主要库 | Main Dependencies
```python
# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
```

### 📦 安装依赖 | Install Dependencies
```bash
pip install streamlit yfinance pandas numpy matplotlib scikit-learn
```

---

## 🚀 功能介绍 | Features

### 📈 投资组合优化 (Portfolio Optimization)
- 输入多个股票代码（如：AAPL, MSFT, TSLA）
- 下载历史收盘价数据
- 使用Q-Learning训练投资组合分配策略
- 输出优化后的股票投资比例，并以饼图形式展示

**流程 | Workflow**:
1. 用户输入股票代码
2. 自动拉取历史数据
3. Q-Learning强化学习智能体进行训练
4. 输出投资建议和可视化图表

---

### 🛡️ 欺诈检测 (Fraud Detection)
- 上传包含交易数据的CSV文件（必须包含`fraud`列）
- 使用随机森林模型进行训练与预测
- 输出模型准确率
- 显示预测结果，并高亮标注欺诈交易

**流程 | Workflow**:
1. 上传交易记录文件
2. 自动训练并测试模型
3. 展示预测结果与准确率
4. 以表格高亮显示检测出的欺诈行为

---

## 📂 文件结构 | File Structure
```text
.
├── main_app.py           # 主程序文件 Main Application File
├── README.md             # 项目说明文档 This README
└── (其他文件)            # Other project files
```

---

## ▶️ 如何运行 | How to Run
```bash
streamlit run main_app.py
```
运行后会自动打开浏览器访问本地地址，如：http://localhost:8501  
After running, your browser will open at a local address like: http://localhost:8501

---

## 📄 许可证 | License

This project is licensed under the **MIT License**.  
此项目采用 **MIT License** 开源协议。

---
