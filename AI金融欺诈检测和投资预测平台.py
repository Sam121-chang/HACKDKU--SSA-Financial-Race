import tweepy
import yfinance as yf
import pandas as pd
import streamlit as st

# 填入你的 API 凭证
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAZ30wEAAAAAh6g84WnqJCuDNMF2CdniOiq6cMM%3DzkQcOhitxK49i8Nvlt7E1JfF22ESmqX1qyNqsjihc7ELaOQvcM'

# 创建 Tweepy 客户端实例
client = tweepy.Client(bearer_token)


# 定义获取 Twitter 数据的函数
def fetch_twitter_data(query, count=100):
    try:
        # 使用 API v2 的搜索端点获取推文
        tweets = client.search_recent_tweets(query=query, max_results=count)
        tweet_data = []
        if tweets.data:
            for tweet in tweets.data:
                tweet_data.append({
                    'created_at': tweet.created_at,
                    'text': tweet.text,
                    'author_id': tweet.author_id,
                })
        else:
            print("未找到相关推文。")
        return pd.DataFrame(tweet_data)
    except tweepy.TweepyException as e:
        print(f"发生错误: {e}")
        return pd.DataFrame()


# 获取某支股票的历史数据（示例：苹果公司股票数据）
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y', interval='1d')
    return stock_data


# 主函数
def main():
    st.title("AI金融欺诈检测和投资预测平台")

    # Twitter 数据部分
    query = st.text_input("请输入Twitter关键词", "Apple")
    num_tweets = st.slider("选择获取的推文数量", 10, 200, 100)

    if query:
        tweets = fetch_twitter_data(query, num_tweets)
        if not tweets.empty:
            st.write(f"展示最新的 {num_tweets} 条关于 '{query}' 的推文:")
            st.dataframe(tweets)
        else:
            st.write("未找到相关推文。")

    # 股票数据部分
    ticker = st.text_input("请输入股票代码（例如：AAPL）", "AAPL")

    if ticker:
        stock_data = fetch_stock_data(ticker)
        st.write(f"{ticker} 股票的历史数据:")
        st.line_chart(stock_data['Close'])


# 运行 Streamlit 应用
if __name__ == "__main__":
    main()
