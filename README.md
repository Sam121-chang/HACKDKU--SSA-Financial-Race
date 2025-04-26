以下是中英双语版本的 `README.md`，包括项目的功能、安装与运行步骤、技术栈等内容，适合直接复制到你的 GitHub 仓库中：

```markdown
# AI Financial Fraud Detection and Investment Prediction Platform  
AI金融欺诈检测和投资预测平台  

This is an AI-based financial fraud detection and investment prediction platform that aims to predict financial market trends using social media data and stock market data. The platform utilizes Twitter data for market sentiment analysis, combined with historical stock data for predictions, helping users make better investment decisions.  
这是一个基于人工智能的金融欺诈检测和投资预测平台，旨在利用社交媒体数据和股市数据来预测金融市场走势。该平台使用 Twitter 数据进行市场情绪分析，结合股市历史数据进行预测，帮助用户在投资决策中做出更明智的选择。

## Features  
## 功能  

- Fetch Twitter data based on specified keywords to analyze social media sentiment.  
- 根据指定关键词获取 Twitter 数据，分析社交媒体情绪。  
- Display historical stock data of specified stocks and plot price trends.  
- 显示指定股票的历史数据并绘制价格变化图。  
- Supports real-time data fetching to show the most relevant market information.  
- 支持实时获取数据，展示最相关的市场信息。  

## Installation and Running Instructions  
## 安装与运行说明  

### 1. Clone the Repository  
### 1. 克隆仓库  

First, clone this GitHub repository to your local machine:  
首先，克隆此 GitHub 仓库到本地：  
```bash  
git clone https://github.com/Sam121-chang/HACKDKU--SSA-Financial-Race.git  
cd HACKDKU--SSA-Financial-Race  
```  

### 2. Install Dependencies  
### 2. 安装依赖  

This project is written in Python and uses libraries like `Streamlit`, `Tweepy`, and `yfinance`. You can install the dependencies with `pip`:  
此项目使用 Python 编写，使用了 `Streamlit`、`Tweepy` 和 `yfinance` 等库。你可以使用 `pip` 安装所有依赖项：  
```bash  
pip install -r requirements.txt  
```  

### 3. Set Up Twitter API  
### 3. 配置 Twitter API  

To fetch Twitter data, you need to configure your Twitter API credentials. Follow these steps to get your API keys:  
要从 Twitter 获取数据，你需要配置 Twitter API 凭证。请按照以下步骤获取你的 API 密钥：  

1. Visit [Twitter Developer](https://developer.twitter.com/) and apply for a developer account.  
1. 前往 [Twitter Developer](https://developer.twitter.com/) 申请一个开发者账户。  
2. Create a project and generate your API keys and Bearer Token.  
2. 创建一个项目并生成你的 API 密钥和 Bearer Token。  
3. Insert your `Bearer Token` into the `bearer_token` variable in the code.  
3. 将你的 `Bearer Token` 填入代码中的 `bearer_token` 变量。  

### 4. Run the App  
### 4. 运行应用  

Once you've installed the dependencies and set up the Twitter API, you can run the Streamlit app:  
安装完依赖并配置好 Twitter API 后，你可以启动 Streamlit 应用：  
```bash  
streamlit run app.py  
```  

Streamlit will automatically open the browser and display the app interface.  
Streamlit 会自动打开浏览器，并展示应用界面。  

### 5. Use the App  
### 5. 使用应用  

- **Fetch Twitter Data**: Enter a keyword in the app interface to display the most recent tweets about that topic.  
- **获取 Twitter 数据**：在应用界面输入关键词，展示关于该主题的最新推文。  

- **View Stock Data**: Enter a stock ticker (e.g., AAPL) in the app interface to display the historical data and price trend of that stock.  
- **查看股票数据**：在应用界面输入股票代码（例如 AAPL），展示该股票的历史数据和价格趋势。  

## Project Structure  
## 项目结构  

```
├── app.py            # Streamlit main application file  
├── requirements.txt  # Project dependencies file  
└── README.md         # Project description file  
```

## Tech Stack  
## 技术栈  

- **Python**: The primary programming language  
- **Streamlit**: Used to build the web app  
- **Tweepy**: Used to interact with the Twitter API  
- **yfinance**: Used to fetch stock data  

- **Python**: 主要编程语言  
- **Streamlit**: 用于构建 Web 应用  
- **Tweepy**: 用于访问 Twitter API  
- **yfinance**: 用于获取股市数据  

## Contributing  
## 贡献  

Feel free to contribute! If you have any ideas or find a bug, feel free to open an Issue or create a Pull Request.  
欢迎贡献！如果你有任何想法或发现了 bug，欢迎提出 Issue 或 Pull Request。  

## License  
## 许可证  

This project is licensed under the [MIT License](LICENSE).  
此项目采用 [MIT License](LICENSE) 开源协议。
```

这个版本的 `README.md` 包含了中英文的说明，帮助用户理解如何使用和运行这个项目。你可以将这个内容复制并粘贴到你的 GitHub 仓库的 `README.md` 文件中。
