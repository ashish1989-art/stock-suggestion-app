import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser

# Download sentiment lexicon (only once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Sidebar controls
st.sidebar.title("🔧 Settings")
symbols = st.sidebar.text_input("Enter Stock Symbols (comma separated)", "AAPL,MSFT")
show_rsi = st.sidebar.checkbox("Include RSI", value=True)
show_macd = st.sidebar.checkbox("Include MACD", value=True)
show_volume = st.sidebar.checkbox("Include Volume", value=True)
show_bbands = st.sidebar.checkbox("Include Bollinger Bands", value=True)
show_stochrsi = st.sidebar.checkbox("Include StochRSI", value=True)
use_news = st.sidebar.checkbox("Include News Sentiment", value=True)
refresh = st.sidebar.button("🔄 Refresh Now")

# Title
st.title("📈 Stock Investment Suggestion App")

for symbol in [s.strip().upper() for s in symbols.split(",")]:
    st.markdown("---")
    st.header(f"📌 Analysis for {symbol}")

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")

        if df.empty:
            st.error("No data found for this symbol.")
            continue

        # Chart
        st.subheader(f"{symbol} Candlestick Chart (1 Year)")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        score = 0

        # RSI
        if show_rsi:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            rsi = df['RSI'].iloc[-1]
            st.write(f"📊 RSI: {rsi:.2f}")
            if rsi < 30:
                st.success("RSI Suggestion: BUY (Oversold)")
                score += 1
            elif rsi > 70:
                st.error("RSI Suggestion: SELL (Overbought)")
                score -= 1
            else:
                st.info("RSI Suggestion: HOLD")

        # MACD
        if show_macd:
            macd = ta.macd(df['Close'], fast=12, slow=26)
            df['MACD'] = macd['MACD_12_26_9']
            df['Signal'] = macd['MACDs_12_26_9']
            m = df['MACD'].iloc[-1]
            s = df['Signal'].iloc[-1]
            st.write(f"📈 MACD: {m:.2f} | Signal: {s:.2f}")
            if m > s:
                st.success("MACD Suggestion: BUY")
                score += 1
            else:
                st.error("MACD Suggestion: SELL")
                score -= 1

        # Volume
        if show_volume:
            latest_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
            st.write(f"🔊 Volume: {latest_vol:.0f} | Avg (20d): {avg_vol:.0f}")
            if latest_vol > avg_vol:
                st.success("Volume: BUY (High Activity)")
                score += 1
            else:
                st.info("Volume: HOLD")

        # Bollinger Bands
        if show_bbands:
            bb = ta.bbands(df['Close'], length=20, std=2)
            upper = bb['BBU_20_2.0'].iloc[-1]
            lower = bb['BBL_20_2.0'].iloc[-1]
            close = df['Close'].iloc[-1]
            st.write(f"📉 BB → Upper: {upper:.2f} | Lower: {lower:.2f}")
            if close <= lower:
                st.success("BB Suggestion: BUY (Touching Lower)")
                score += 1
            elif close >= upper:
                st.error("BB Suggestion: SELL (Touching Upper)")
                score -= 1
            else:
                st.info("BB Suggestion: HOLD")

        # StochRSI
        if show_stochrsi:
            stoch = ta.stochrsi(df['Close'], length=14)
            srsi = stoch['STOCHRSIk_14_14_3_3'].iloc[-1]
            st.write(f"⚡ StochRSI: {srsi:.2f}")
            if srsi < 0.2:
                st.success("StochRSI: BUY")
                score += 1
            elif srsi > 0.8:
                st.error("StochRSI: SELL")
                score -= 1
            else:
                st.info("StochRSI: HOLD")

        # News Sentiment from Google News RSS
        if use_news:
            st.markdown("---")
            st.subheader("📰 News Headlines + Sentiment")
            rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)

            sid = SentimentIntensityAnalyzer()
            sentiment_score = 0
            total = 0

            for entry in feed.entries[:5]:
                title = entry.title
                link = entry.link
                st.write(f"🔗 [{title}]({link})")

                sentiment = sid.polarity_scores(title)['compound']
                sentiment_score += sentiment
                total += 1

                if sentiment >= 0.05:
                    st.success("🟢 Positive")
                elif sentiment <= -0.05:
                    st.error("🔴 Negative")
                else:
                    st.info("🟡 Neutral")

            if total:
                avg_sent = sentiment_score / total
                st.markdown("### 🧠 News Sentiment Summary")
                if avg_sent > 0.05:
                    st.success("Overall Sentiment: Positive")
                    score += 1
                elif avg_sent < -0.05:
                    st.error("Overall Sentiment: Negative")
                    score -= 1
                else:
                    st.info("Overall Sentiment: Neutral")
        # Final Suggestion
        st.markdown("---")
        st.subheader("📌 Final Smart Suggestion")
        if score >= 3:
            st.success("✅ STRONG BUY — multiple confirmations")
        elif score == 2:
            st.success("✔️ BUY")
        elif score in [1, 0]:
            st.info("⏸ HOLD")
        else:
            st.error("❌ SELL — weak or negative signals")

    except Exception as e:
        st.error(f"Error: {e}")