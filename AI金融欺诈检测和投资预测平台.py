import tweepy
import yfinance as yf
import pandas as pd
import streamlit as st

# 填入你的 API 凭证
# Insert your API credentials 
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAZ30wEAAAAAh6g84WnqJCuDNMF2CdniOiq6cMM%3DzkQcOhitxK49i8Nvlt7E1JfF22ESmqX1qyNqsjihc7ELaOQvcM'

# 创建 Tweepy 客户端实例
# Creat a Tweepy client instance
client = tweepy.Client(bearer_token)


# 定义获取 Twitter 数据的函数
# Define the function to fetch Twitter data
def fetch_twitter_data(query, count=100):
    try:
        # 使用 API v2 的搜索端点获取推文
        # Use the API v2 search endpoint to fetch tweets
        tweets = client.search_recent_tweets(query=query, max_results=count)
        tweet_data = []

        # 如果有推文数据，解析并存储
        # If there are tweets, parse and store them
        if tweets.data:
            for tweet in tweets.data:
                tweet_data.append({
                    'created_at': tweet.created_at, #推文的创建时间 / The creation time of the tweet
                    'text': tweet.text, #推文内容 / The content of the tweet
                    'author_id': tweet.author_id, #推文作者的 ID / The author ID of the tweet
                })
        else:
            print("未找到相关推文。") # If no tweets founds, print message
        return pd.DataFrame(tweet_data) # Return tweet data as a pandas DataFrame
    except tweepy.TweepyException as e:
        print(f"发生错误: {e}") # If an error occurs, print error message
        return pd.DataFrame() # Return empty DataFrame on error


# 获取某支股票的历史数据（示例：苹果公司股票数据）
# Fetch historical stock data for a given ticker (example: Apple stock data)
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y', interval='1d')
    return stock_data


# 主函数
# Main function
def main():
    st.title("AI金融欺诈检测和投资预测平台") # Streamlit app title

    # Twitter 数据部分
    # Twitter Data Section
    query = st.text_input("请输入Twitter关键词", "Apple") # User input for Twitter search query
    num_tweets = st.slider("选择获取的推文数量", 10, 200, 100) # Slider to select number of tweets to fetch

    # 如果输入了查询关键词，则调用 fetch_twitter_data 函数获取推文数据
    # If a query is entered, fetch tweets using fetch_twitter_data function
    if query:
        tweets = fetch_twitter_data(query, num_tweets) # Call functionto get tweets
        if not tweets.empty: # If there are tweets, display them
            st.write(f"展示最新的 {num_tweets} 条关于 '{query}' 的推文:") # Display message for the tweets fetched
            st.dataframe(tweets) # Display tweets in a table format
        else:
            st.write("未找到相关推文。") # If no tweets are found, display a message

    # 股票数据部分
    # Stock Data Section
    ticker = st.text_input("请输入股票代码（例如：AAPL）", "AAPL") # User input for stock ticker
# 如果输入了股票代码，则调用 fetch_stock_data 函数获取股票数据
# If a ticker is entered, fetch stock data using fetch_stock_data function    
    if ticker:
        stock_data = fetch_stock_data(ticker) # Call function to get stock data
        st.write(f"{ticker} 股票的历史数据:") # Display message for the stock data
        st.line_chart(stock_data['Close']) # Plot a line chart of the closing prices


# 运行 Streamlit 应用
# Run the Streamlit app
if __name__ == "__main__":
    main() # Call the main function to run the app / 调用主函数运行应用
