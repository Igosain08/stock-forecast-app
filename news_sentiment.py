import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NewsAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Add financial terms to sentiment analyzer
        financial_lexicon = {
            # Positive financial terms
            'upgrade': 3.0,
            'upgraded': 3.0,
            'beat': 2.0,
            'beats': 2.0,
            'exceeded': 2.0,
            'outperform': 3.0,
            'outperforms': 3.0,
            'growth': 2.0,
            'profitable': 2.0,
            'bullish': 3.0,
            'dividend': 1.5,
            'revenue': 1.0,
            
            # Negative financial terms
            'downgrade': -3.0,
            'downgraded': -3.0,
            'miss': -2.0,
            'misses': -2.0,
            'missed': -2.0,
            'underperform': -3.0,
            'underperforms': -3.0,
            'decline': -2.0,
            'loss': -2.0,
            'bearish': -3.0,
            'debt': -1.0,
            'investigation': -2.0,
            'lawsuit': -2.5,
            'fine': -2.0,
            'penalty': -2.0
        }
        
        # Update the sentiment analyzer's lexicon
        for word, score in financial_lexicon.items():
            self.sentiment_analyzer.lexicon[word] = score
            
    def get_company_news(self, ticker_symbol, max_items=25):
        """
        Get recent news for a company and analyze sentiment
        
        Parameters:
        -----------
        ticker_symbol : str
            The stock ticker symbol
        max_items : int, optional
            Maximum number of news items to return
            
        Returns:
        --------
        dict
            News items with sentiment analysis
        """
        try:
            # Get company info (for proper name)
            ticker = yf.Ticker(ticker_symbol)
            company_name = ticker.info.get('shortName', ticker_symbol)
            
            # Try multiple news sources
            news_items = []
            
            # Method 1: Yahoo Finance
            yahoo_news = self._get_yahoo_finance_news(ticker_symbol, max_items)
            news_items.extend(yahoo_news)
            
            # Method 2: Financial websites (using search)
            search_news = self._get_search_news(company_name, max_items)
            
            # Remove duplicates (based on title similarity)
            for item in search_news:
                if not any(self._similar_titles(item['title'], existing['title']) for existing in news_items):
                    news_items.append(item)
            
            # Limit to max_items
            news_items = news_items[:max_items]
            
            # Calculate overall sentiment
            sentiment_scores = [item['sentiment_score'] for item in news_items]
            overall_sentiment = {
                'average_score': np.mean(sentiment_scores) if sentiment_scores else 0,
                'positive_count': sum(1 for score in sentiment_scores if score > 0.2),
                'neutral_count': sum(1 for score in sentiment_scores if -0.2 <= score <= 0.2),
                'negative_count': sum(1 for score in sentiment_scores if score < -0.2)
            }
            
            # Add sentiment label
            overall_sentiment['sentiment_label'] = self._get_sentiment_label(overall_sentiment['average_score'])
            
            return {
                'company': company_name,
                'ticker': ticker_symbol,
                'news_items': news_items,
                'overall_sentiment': overall_sentiment
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _get_yahoo_finance_news(self, ticker_symbol, max_items):
        """Get news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news
            
            result = []
            for item in news[:max_items]:
                # Extract article content if possible
                article_content = self._extract_article_content(item.get('link', ''))
                
                # Calculate sentiment
                title = item.get('title', '')
                summary = item.get('summary', '')
                content_to_analyze = title + ' ' + summary + ' ' + article_content
                sentiment = self.sentiment_analyzer.polarity_scores(content_to_analyze)
                
                result.append({
                    'title': title,
                    'summary': summary,
                    'link': item.get('link', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'published_time': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment_score': sentiment['compound'],
                    'sentiment_label': self._get_sentiment_label(sentiment['compound']),
                    'sentiment_details': {
                        'positive': sentiment['pos'],
                        'neutral': sentiment['neu'],
                        'negative': sentiment['neg']
                    }
                })
            
            return result
        
        except Exception as e:
            print(f"Error getting Yahoo Finance news: {e}")
            return []
    
    def _get_search_news(self, company_name, max_items):
        """Get financial news using web search"""
        # Note: In a production system, you would use a proper news API
        # This is a simplified implementation for demonstration
        try:
            # Create search query for financial news
            search_query = f"{company_name} stock news"
            search_query = search_query.replace(' ', '+')
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            result = []
            
            # For demonstration purposes, we'll limit our search to avoid rate limiting
            # In a production environment, you would use a proper API
            sources = [
                {'url': f'https://www.google.com/search?q={search_query}&tbm=nws', 'parser': self._parse_google_news}
            ]
            
            for source in sources:
                try:
                    response = requests.get(source['url'], headers=headers, timeout=10)
                    if response.status_code == 200:
                        result.extend(source['parser'](response.text, max_items))
                except Exception as source_error:
                    print(f"Error with source {source['url']}: {source_error}")
            
            return result[:max_items]
        
        except Exception as e:
            print(f"Error getting search news: {e}")
            return []
    
    def _parse_google_news(self, html_content, max_items):
        """Parse Google News search results"""
        result = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find news articles
        news_elements = soup.select('div.xuvV6b')
        
        for element in news_elements[:max_items]:
            try:
                # Extract title
                title_element = element.select_one('div.BNeawe.vvjwJb.AP7Wnd')
                title = title_element.text if title_element else 'No title'
                
                # Extract link
                link_element = element.select_one('a')
                link = link_element['href'] if link_element and 'href' in link_element.attrs else ''
                
                # Clean link (Google prepends their own URL)
                if link.startswith('/url?q='):
                    link = link.split('/url?q=')[1].split('&sa=')[0]
                
                # Extract source and date
                source_date_element = element.select_one('div.BNeawe.UPmit.AP7Wnd')
                source_date_text = source_date_element.text if source_date_element else ''
                
                # Split source and date
                parts = source_date_text.split(' - ')
                source = parts[0] if parts else 'Unknown'
                published_time = parts[1] if len(parts) > 1 else 'Unknown date'
                
                # Extract summary
                summary_element = element.select_one('div.BNeawe.s3v9rd.AP7Wnd')
                summary = summary_element.text if summary_element else ''
                
                # Calculate sentiment
                content_to_analyze = title + ' ' + summary
                sentiment = self.sentiment_analyzer.polarity_scores(content_to_analyze)
                
                result.append({
                    'title': title,
                    'summary': summary,
                    'link': link,
                    'source': source,
                    'published_time': published_time,
                    'sentiment_score': sentiment['compound'],
                    'sentiment_label': self._get_sentiment_label(sentiment['compound']),
                    'sentiment_details': {
                        'positive': sentiment['pos'],
                        'neutral': sentiment['neu'],
                        'negative': sentiment['neg']
                    }
                })
            
            except Exception as item_error:
                print(f"Error parsing news item: {item_error}")
        
        return result
    
    
    def _extract_article_content(self, url):

        try:
            # Skip if URL is empty or invalid
            if not url or not (url.startswith('http://') or url.startswith('https://')):
                return ""
                
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit to first 5000 characters to avoid processing too much text
            return text[:5000]
        
        except Exception as e:
            # Instead of printing, just return an empty string silently
            # print(f"Error extracting article content: {e}")
            return ""

    def _get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _similar_titles(self, title1, title2):
        """Check if two titles are similar"""
        # Remove special characters and convert to lowercase
        t1 = re.sub(r'[^\w\s]', '', title1.lower())
        t2 = re.sub(r'[^\w\s]', '', title2.lower())
        
        # Split into words
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Return True if similarity is above threshold
        return intersection / union > 0.5 if union > 0 else False

def analyze_company_news(ticker_symbol, max_items=25):
    """
    Get and analyze news for a company
    
    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    max_items : int, optional
        Maximum number of news items to return
        
    Returns:
    --------
    dict
        News items with sentiment analysis
    """
    analyzer = NewsAnalyzer()
    return analyzer.get_company_news(ticker_symbol, max_items)

def analyze_sentiment_impact(ticker_symbol, days=90):
    """
    Analyze the impact of news sentiment on stock price
    
    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    days : int, optional
        Number of days to analyze
        
    Returns:
    --------
    dict
        Sentiment impact analysis
    """
    try:
        # Get company news
        analyzer = NewsAnalyzer()
        news_data = analyzer.get_company_news(ticker_symbol, max_items=100)
        
        if 'error' in news_data:
            return news_data
        
        # Get historical stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            return {'error': f"No stock data available for {ticker_symbol}"}
        
        # Calculate daily returns
        stock_data['Return'] = stock_data['Adj Close'].pct_change()
        
        # Group news by date
        news_by_date = {}
        
        for item in news_data['news_items']:
            try:
                # Parse the date
                date_str = item['published_time']
                
                # Handle different date formats
                try:
                    if isinstance(date_str, str):
                        if re.match(r'\d+ \w+ ago', date_str):
                            # Handle relative dates like "2 days ago"
                            parts = date_str.split()
                            number = int(parts[0])
                            unit = parts[1].lower()
                            
                            if 'minute' in unit:
                                date = datetime.now() - timedelta(minutes=number)
                            elif 'hour' in unit:
                                date = datetime.now() - timedelta(hours=number)
                            elif 'day' in unit:
                                date = datetime.now() - timedelta(days=number)
                            elif 'week' in unit:
                                date = datetime.now() - timedelta(weeks=number)
                            else:
                                # Default to today
                                date = datetime.now()
                        else:
                            # Try different formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%b %d, %Y', '%d %b %Y']:
                                try:
                                    date = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                # If all formats fail, default to today
                                date = datetime.now()
                    else:
                        # If not a string, default to today
                        date = datetime.now()
                except Exception:
                    # In case of any parsing error, default to today
                    date = datetime.now()
                
                # Convert to date string
                date_key = date.strftime('%Y-%m-%d')
                
                # Add to news by date
                if date_key not in news_by_date:
                    news_by_date[date_key] = []
                
                news_by_date[date_key].append(item)
            
            except Exception as e:
                print(f"Error processing news item date: {e}")
        
        # Calculate average sentiment score for each date
        sentiment_by_date = {}
        
        for date, items in news_by_date.items():
            sentiment_scores = [item['sentiment_score'] for item in items]
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            sentiment_by_date[date] = {
                'sentiment_score': avg_sentiment,
                'sentiment_label': analyzer._get_sentiment_label(avg_sentiment),
                'news_count': len(items)
            }
        
        # Merge stock data with sentiment data
        merged_data = []
        
        for date, row in stock_data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            sentiment_data = sentiment_by_date.get(date_str, {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'news_count': 0
            })
            
            merged_data.append({
                'date': date_str,
                'close': row['Adj Close'],
                'return': row['Return'] * 100 if not pd.isna(row['Return']) else 0,  # Convert to percentage
                'volume': row['Volume'],
                'sentiment_score': sentiment_data['sentiment_score'],
                'sentiment_label': sentiment_data['sentiment_label'],
                'news_count': sentiment_data['news_count']
            })
        
        # Calculate correlation between sentiment and returns
        sentiment_scores = [item['sentiment_score'] for item in merged_data if item['news_count'] > 0]
        returns = [item['return'] for item in merged_data if item['news_count'] > 0]
        
        correlation = np.corrcoef(sentiment_scores, returns)[0, 1] if len(sentiment_scores) > 1 else 0
        
        # Analyze next-day return after significant sentiment
        next_day_impact = []
        
        for i in range(len(merged_data) - 1):
            if merged_data[i]['news_count'] > 0 and abs(merged_data[i]['sentiment_score']) > 0.2:
                next_day_impact.append({
                    'date': merged_data[i]['date'],
                    'sentiment_score': merged_data[i]['sentiment_score'],
                    'sentiment_label': merged_data[i]['sentiment_label'],
                    'next_day_return': merged_data[i + 1]['return']
                })
        
        # Calculate average next-day return by sentiment
        positive_returns = [item['next_day_return'] for item in next_day_impact if item['sentiment_label'] == 'positive']
        negative_returns = [item['next_day_return'] for item in next_day_impact if item['sentiment_label'] == 'negative']
        
        avg_positive_return = np.mean(positive_returns) if positive_returns else 0
        avg_negative_return = np.mean(negative_returns) if negative_returns else 0
        
        return {
            'ticker': ticker_symbol,
            'days_analyzed': days,
            'correlation': correlation,
            'avg_positive_return': avg_positive_return,
            'avg_negative_return': avg_negative_return,
            'data': merged_data,
            'next_day_impact': next_day_impact,
            'sentiment_summary': {
                'positive_news_days': len([d for d in merged_data if d['sentiment_label'] == 'positive' and d['news_count'] > 0]),
                'negative_news_days': len([d for d in merged_data if d['sentiment_label'] == 'negative' and d['news_count'] > 0]),
                'neutral_news_days': len([d for d in merged_data if d['sentiment_label'] == 'neutral' and d['news_count'] > 0]),
                'no_news_days': len([d for d in merged_data if d['news_count'] == 0])
            }
        }
    
    except Exception as e:
        return {'error': str(e)}
