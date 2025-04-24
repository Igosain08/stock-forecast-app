import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
import json
from datetime import datetime

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER once
sid = SentimentIntensityAnalyzer()


def fetch_yahoo_news(ticker):
    """
    Fetch Yahoo Finance news with proper parsing of the nested structure.
    """
    try:
        print(f"Fetching news for {ticker} using Yahoo Finance API...")
        ticker_obj = yf.Ticker(ticker)
        news_data = ticker_obj.news
        
        print(f"Retrieved {len(news_data)} news items")
        
        if not news_data:
            print(f"No news found for {ticker}.")
            return pd.DataFrame(columns=['date', 'headline', 'source', 'url', 'summary'])
        
        # Process all news data
        all_news = []
        for item in news_data:
            # Handle the nested structure carefully
            content = item.get('content') if isinstance(item, dict) else None
            
            if content is None:
                continue  # Skip if content is None
            
            # Safely extract headline/title
            title = None
            if isinstance(content, dict):
                title = content.get('title')
            if title is None and isinstance(item, dict):
                title = item.get('title')
                
            # Safely extract summary
            summary = None
            if isinstance(content, dict):
                summary = content.get('summary') or content.get('description')
                
            # Safely extract provider/source
            provider = "Yahoo Finance"
            if isinstance(content, dict) and isinstance(content.get('provider'), dict):
                provider = content.get('provider').get('displayName', "Yahoo Finance")
                
            # Safely extract URL
            url = None
            if isinstance(content, dict):
                if isinstance(content.get('clickThroughUrl'), dict):
                    url = content.get('clickThroughUrl').get('url')
                elif isinstance(content.get('canonicalUrl'), dict):
                    url = content.get('canonicalUrl').get('url')
                    
            # Safely extract publication date
            pub_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(content, dict):
                pub_date = content.get('pubDate', pub_date)
                
            # Add to news list if we have at least a headline
            if title:
                all_news.append({
                    'date': pub_date,
                    'headline': title,
                    'source': provider,
                    'url': url if url else "",
                    'summary': summary if summary else ""
                })
        
        # Create DataFrame with all news
        df = pd.DataFrame(all_news)
        
        if df.empty:
            print("Failed to process any news items.")
            return pd.DataFrame(columns=['date', 'headline', 'source', 'url', 'summary'])
            
        return df
    
    except Exception as e:
        print(f"Error fetching news from Yahoo Finance for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'headline', 'source', 'url', 'summary'])


def analyze_sentiment(df):
    """
    Adds sentiment scores and labels to a DataFrame of news.
    """
    # Check if DataFrame is empty or has empty headlines
    if df.empty:
        print("Empty dataframe - no news to analyze.")
        return df
        
    # Check if all headlines are missing or empty
    if 'headline' not in df.columns or df['headline'].isna().all() or (df['headline'] == '').all():
        print("No valid headlines found for sentiment analysis.")
        return df
    
    print("Analyzing sentiment of news headlines...")
    
    def score_sentiment(text):
        if pd.isna(text) or text == '':
            return 0.0
        return sid.polarity_scores(text)['compound']
    
    # Use both headline and summary for better sentiment analysis if available
    df['analysis_text'] = df['headline'].fillna('')
    if 'summary' in df.columns:
        # Only use non-empty summaries
        for i, row in df.iterrows():
            if pd.notna(row['summary']) and row['summary'] != '':
                df.at[i, 'analysis_text'] = f"{row['headline']}. {row['summary']}"
    
    df['sentiment_score'] = df['analysis_text'].apply(score_sentiment)

    def classify(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
        
    df['sentiment'] = df['sentiment_score'].apply(classify)
    
    # Count sentiment categories
    sentiment_counts = df['sentiment'].value_counts()
    print(f"Sentiment distribution: {sentiment_counts.to_dict()}")
    
    return df

def get_sentiment_feature(ticker, days=7):
    """
    Main function to return processed news and sentiment.
    """
    news = fetch_yahoo_news(ticker)
    
    # Filter manually if you wish using pub_date (optional)
    if not news.empty and 'date' in news.columns:
        try:
            # Normalize all dates to UTC then convert to naive
            news['date'] = pd.to_datetime(news['date'], errors='coerce')
            
            # First ensure all timestamps have timezone info (use UTC if none)
            news['date'] = news['date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            
            # Then convert all to naive by removing timezone info
            news['date'] = news['date'].dt.tz_localize(None)
            
            # Use a timezone-naive cutoff date
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            news = news[news['date'] >= cutoff_date]
        except Exception as e:
            print(f"Error handling date filtering: {str(e)}")
            # If date filtering fails, continue with all news

    # Check if we have valid headlines
    has_valid_headlines = (
        not news.empty and 
        'headline' in news.columns and 
        not news['headline'].isna().all() and 
        not (news['headline'] == '').all()
    )
    
    if has_valid_headlines:
        news = analyze_sentiment(news)
        
        # Create a daily sentiment DataFrame
        try:
            if 'date' in news.columns:
                # Extract just the date part
                news['date'] = pd.to_datetime(news['date'], errors='coerce').dt.date
                daily_sentiment = news.groupby('date')['sentiment_score'].mean().reset_index()
                daily_sentiment.columns = ['date', 'sentiment_score']
                return news, daily_sentiment
        except Exception as e:
            print(f"Error creating daily sentiment: {str(e)}")
            
        return news, pd.DataFrame(columns=['date', 'sentiment_score'])
    else:
        print("No valid headlines found for sentiment analysis.")
        return pd.DataFrame(columns=['date', 'headline', 'source', 'url', 'summary', 'sentiment_score', 'sentiment']), pd.DataFrame(columns=['date', 'sentiment_score'])

if __name__ == "__main__":
    ticker = input("Enter Ticker Symbol (e.g., AAPL, MSFT): ").upper()

    print(f"\nFetching and analyzing news for {ticker}...\n")
    
    news_df, daily_sentiment = get_sentiment_feature(ticker)  # Unpack the tuple here
    
    # Check for valid results
    if not news_df.empty and 'headline' in news_df.columns and not news_df['headline'].isna().all() and not (news_df['headline'] == '').all():
        # Rest of the code remains the same

        print("\n--- News with Sentiment Analysis ---\n")
        
        # Format the output for better readability
        pd.set_option('display.max_colwidth', 100)  # Set max column width
        
        # Print headlines with sentiment
        for i, row in news_df.iterrows():
            if pd.isna(row['headline']) or row['headline'] == '':
                continue
                
            print(f"[{row['sentiment']}] ({row['sentiment_score']:.3f}) - {row['headline']}")
            print(f"Source: {row['source']}")
            
            if 'summary' in row and pd.notna(row['summary']) and row['summary'] != '':
                print(f"Summary: {row['summary'][:150]}..." if len(row['summary']) > 150 else f"Summary: {row['summary']}")
                
            if 'url' in row and pd.notna(row['url']) and row['url'] != '':
                print(f"URL: {row['url']}")
                
            print("-" * 80)
        
        # Show count of all news items with valid headlines
        valid_news = news_df[~(news_df['headline'].isna() | (news_df['headline'] == ''))]
        print(f"\nTotal valid news items: {len(valid_news)}")
        
        if not valid_news.empty:
            # Get overall sentiment stats
            avg_sentiment = valid_news['sentiment_score'].mean()
            print(f"\nOverall average sentiment for {ticker}: {avg_sentiment:.4f}")
            
            if avg_sentiment >= 0.05:
                sentiment_label = "POSITIVE"
            elif avg_sentiment <= -0.05:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            print(f"Overall sentiment classification: {sentiment_label}")
        else:
            print("\nNo valid headlines found for sentiment analysis.")
    else:
        print(f"\nNo valid news data available for {ticker}.")
        print("Try a different ticker symbol or API.")
