```markdown
# AI Financial Fraud Detection & Investment Prediction Platform

This is a Streamlit-based platform for financial fraud detection and investment prediction. The application uses real-time Twitter data and historical stock data to provide insights into financial markets. The goal of this project is to predict potential market trends and provide a simple interface to monitor financial data.

这是一个基于Streamlit的金融欺诈检测与投资预测平台。该应用程序使用实时Twitter数据和历史股票数据，提供关于金融市场的洞察。该项目的目标是预测潜在的市场趋势，并提供一个简单的界面来监控金融数据。

---

## Features / 功能

- **Twitter Data Integration**: Fetch recent tweets based on keywords.
- **股票数据集成**：获取基于股票代码的历史数据。
- **Real-time Financial Insights**: Display Twitter data and stock prices.
- **实时金融数据展示**：展示Twitter数据和股票价格。

---

## Prerequisites / 环境要求

- Python 3.8 or later / Python 3.8 或更高版本
- pip (Python package manager) / pip（Python包管理器）

---

## Installation / 安装

1. **Clone the repository** / 克隆仓库

   First, clone this repository to your local machine:

   ```bash
   git clone https://github.com/Sam121-chang/HACKDKU--SSA-Financial-Race.git
   ```

2. **Navigate to the project directory** / 进入项目目录

   Change to the project directory:

   ```bash
   cd HACKDKU--SSA-Financial-Race
   ```

3. **Install dependencies** / 安装依赖项

   Ensure you have Python 3.8 or later installed, then run the following command to install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install the required libraries including Streamlit, Tweepy, YFinance, and Pandas.

4. **Set up Twitter API credentials** / 配置Twitter API凭证

   To access Twitter data, you need to set up your Twitter Developer API credentials. Follow these steps:
   
   - Go to the [Twitter Developer Portal](https://developer.twitter.com/) and create a new project.
   - Get your **Bearer Token** and add it to the `bearer_token` variable in the `app.py` file.
   
   在访问Twitter数据之前，你需要设置Twitter开发者API凭证。按以下步骤操作：
   
   - 访问 [Twitter开发者门户](https://developer.twitter.com/)，并创建一个新的项目。
   - 获取你的 **Bearer Token** 并将其添加到 `app.py` 文件中的 `bearer_token` 变量。

5. **Run the Streamlit app** / 运行Streamlit应用

   After installing the dependencies and configuring the API credentials, run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit server, and you can view the app in your browser at `http://localhost:8501`.

---

## Usage / 使用

- **Twitter Data** / Twitter 数据
    - Enter a keyword in the search box to fetch the latest tweets related to that keyword.
    - 输入一个关键词来获取与该关键词相关的最新推文。

- **Stock Data** / 股票数据
    - Enter a stock ticker symbol (e.g., AAPL for Apple) to fetch the historical stock data.
    - 输入股票代码（例如：AAPL表示苹果公司）来获取该股票的历史数据。

---

## Deployment / 部署

You can deploy this app on your own server or use a cloud service like AWS Elastic Beanstalk to host it. Here's how you can deploy it on **AWS Elastic Beanstalk**:

1. **Install AWS CLI and EB CLI** / 安装 AWS CLI 和 EB CLI
    - Follow the [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) to install AWS CLI.
    - Follow the [EB CLI Installation Guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html) to install EB CLI.

2. **Create an Elastic Beanstalk application** / 创建Elastic Beanstalk应用
    - Run the following command to initialize the application:
    
    ```bash
    eb init -p python-3.8 my-financial-prediction-app
    ```

    - Then create a new environment:
    
    ```bash
    eb create my-financial-prediction-env
    ```

3. **Deploy to Elastic Beanstalk** / 部署到Elastic Beanstalk

    After configuring your AWS credentials, deploy the app:

    ```bash
    eb deploy
    ```

    You can then access your app via the Elastic Beanstalk URL.

---

## License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

该项目使用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

---

## Acknowledgements / 致谢

- Thanks to AWS for sponsoring this project.
- Thanks to [Tweepy](https://www.tweepy.org/) for providing an easy way to interact with the Twitter API.
- Thanks to [YFinance](https://pypi.org/project/yfinance/) for providing stock data.


感谢 [Tweepy](https://www.tweepy.org/) 提供了与Twitter API交互的便捷方法。
感谢 [YFinance](https://pypi.org/project/yfinance/) 提供股票数据。
```

### 关键内容解释：
- **如何克隆代码**：提供了克隆仓库的命令。
- **安装依赖项**：通过 `pip install -r requirements.txt` 安装所有必要的库。
- **Twitter API 配置**：解释了如何获取 Twitter API 凭证并将其配置到代码中。
- **运行应用**：展示了如何启动 Streamlit 应用。
- **AWS 部署**：提供了如何将应用部署到 AWS Elastic Beanstalk 的简要说明。
