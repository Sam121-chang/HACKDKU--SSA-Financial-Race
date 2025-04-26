```markdown
# HACKDKU--SSA-Financial-Race

## 项目概述

这是我参加 **2025昆山杜克大学黑客松比赛（HACKDKU）** 金融赛道的项目。本项目使用 AI 技术构建了一个投资预测平台，旨在帮助投资者做出更精确的决策。通过分析股票数据，预测未来的股票价格走势，提供可靠的投资建议。

## 项目功能

- **股票数据获取**：自动从公开的数据源获取最新的股票市场数据。
- **数据分析**：使用机器学习算法对股票价格进行分析，预测未来的涨跌趋势。
- **可视化展示**：图表展示历史价格与预测结果，帮助用户更直观地理解数据。
- **投资预测**：基于历史数据，使用AI模型预测股票的未来价格。

## 技术栈

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- yFinance API (用于获取股票数据)

## 安装与运行

### 环境要求

- Python 3.7 及以上版本
- 相关依赖库：Pandas, NumPy, Matplotlib, Scikit-learn, yFinance

### 安装步骤

1. 克隆此仓库到本地：
   ```bash
   git clone https://github.com/Sam121-chang/HACKDKU--SSA-Financial-Race.git
   ```
   
2. 进入项目目录：
   ```bash
   cd HACKDKU--SSA-Financial-Race
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 运行主程序：
   ```bash
   python main.py
   ```

### 使用说明

1. 在 `main.py` 文件中，修改要预测的股票符号（例如：`AAPL` 为苹果公司股票）。
2. 运行程序后，系统将从 yFinance 获取该股票的历史数据，并使用训练好的模型进行未来价格的预测。
3. 结果将以图表的形式展示，便于用户查看。

## 贡献

欢迎任何形式的贡献！如果您有任何建议或发现 bug，欢迎提交 Issue 或 Pull Request。

## License

本项目采用 MIT 许可证，详情请查看 [LICENSE](LICENSE) 文件。

---

# HACKDKU--SSA-Financial-Race

## Project Overview

This is my project for the **2025 Hackathon of Duke Kunshan University (HACKDKU)** in the financial track. The project uses AI technology to build an investment prediction platform aimed at helping investors make more accurate decisions. By analyzing stock data, it predicts future stock price trends and provides reliable investment advice.

## Features

- **Stock Data Retrieval**: Automatically fetch the latest stock market data from public data sources.
- **Data Analysis**: Use machine learning algorithms to analyze stock prices and predict future trends.
- **Visualization**: Display historical prices and prediction results in charts for better data understanding.
- **Investment Prediction**: Predict future stock prices based on historical data using AI models.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- yFinance API (for fetching stock data)

## Installation and Running

### Requirements

- Python 3.7 or higher
- Dependencies: Pandas, NumPy, Matplotlib, Scikit-learn, yFinance

### Installation Steps

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Sam121-chang/HACKDKU--SSA-Financial-Race.git
   ```

2. Navigate to the project directory:
   ```bash
   cd HACKDKU--SSA-Financial-Race
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main program:
   ```bash
   python main.py
   ```

### Usage Instructions

1. In the `main.py` file, modify the stock symbol you want to predict (e.g., `AAPL` for Apple Inc.).
2. After running the program, it will fetch historical data for that stock from yFinance and predict future prices using the trained model.
3. The results will be displayed in a chart for better visualization.

## Contributing

Contributions are welcome! If you have any suggestions or find any bugs, feel free to submit an Issue or a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

