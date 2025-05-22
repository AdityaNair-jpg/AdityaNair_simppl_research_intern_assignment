import pandas as pd
from datetime import datetime, timedelta
import re
import streamlit as st # Streamlit is needed for st.warning in generate_enhanced_insights if df is empty

def generate_enhanced_insights(df):
    """Generate more detailed and context-aware insights from the DataFrame."""
    insights = []
    
    if df.empty:
        return ["No data available for analysis."]
    
    # Temporal insights
    if 'created_at' in df.columns and not df['created_at'].empty:
        # Ensure 'created_at' is datetime type before calculating date_range
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['created_at']) # Drop rows where conversion failed

        if not df['created_at'].empty:
            date_range = (df['created_at'].max() - df['created_at'].min()).days
            avg_posts_per_day = len(df) / max(date_range, 1)
            insights.append(f"📈 **Activity Pattern**: Average of {avg_posts_per_day:.1f} posts per day over {date_range} days")
            
            # Peak activity analysis
            df_temp = df.copy()
            df_temp['hour'] = df_temp['created_at'].dt.hour
            peak_hour = df_temp['hour'].mode().iloc[0] if not df_temp['hour'].empty else "Unknown"
            insights.append(f"⏰ **Peak Activity**: Most posts occur around {peak_hour}:00")
        else:
            insights.append("No valid date data for temporal insights.")
    else:
        insights.append("No 'created_at' column for temporal insights.")

    # Content insights
    if 'content' in df.columns and not df['content'].empty:
        # Ensure content is string type for .str.len()
        df['content'] = df['content'].astype(str)
        avg_content_length = df['content'].str.len().mean()
        insights.append(f"📝 **Content Length**: Average post length is {avg_content_length:.0f} characters")
        
        # Engagement proxy (longer posts might indicate more engagement)
        long_posts = df[df['content'].str.len() > avg_content_length * 1.5]
        if not long_posts.empty:
            insights.append(f"📊 **Content Quality**: {len(long_posts)} posts ({len(long_posts)/len(df)*100:.1f}%) are significantly longer than average")
    else:
        insights.append("No 'content' column for content insights.")
    
    # Author insights
    if 'author' in df.columns and not df['author'].empty:
        unique_authors = df['author'].nunique()
        posts_per_author = len(df) / unique_authors if unique_authors > 0 else 0
        top_author = df['author'].value_counts().index[0] if not df['author'].empty else "Unknown"
        top_author_posts = df['author'].value_counts().iloc[0] if not df['author'].empty else 0
        
        insights.append(f"👥 **Community Size**: {unique_authors} unique authors with average {posts_per_author:.1f} posts each")
        insights.append(f"🌟 **Top Contributor**: '{top_author}' with {top_author_posts} posts ({top_author_posts/len(df)*100:.1f}% of all content)")
    else:
        insights.append("No 'author' column for author insights.")
    
    # Sentiment insights
    if 'sentiment_label' in df.columns and not df['sentiment_label'].empty:
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True) * 100
        if not sentiment_dist.empty:
            dominant_sentiment = sentiment_dist.index[0]
            sentiment_pct = sentiment_dist.iloc[0]
            insights.append(f"😊 **Sentiment Overview**: {dominant_sentiment} sentiment dominates at {sentiment_pct:.1f}% of posts")
            
            if 'sentiment_numeric_score' in df.columns and not df['sentiment_numeric_score'].empty:
                avg_sentiment = df['sentiment_numeric_score'].mean()
                sentiment_trend = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
                insights.append(f"📊 **Sentiment Trend**: Overall sentiment leans {sentiment_trend} (score: {avg_sentiment:.2f})")
            else:
                insights.append("No 'sentiment_numeric_score' for detailed sentiment trend.")
        else:
            insights.append("No valid sentiment labels for sentiment overview.")
    else:
        insights.append("No 'sentiment_label' column for sentiment insights.")
    
    # Subreddit insights
    if 'subreddit' in df.columns and not df['subreddit'].empty:
        top_subreddits = df['subreddit'].value_counts().head(3)
        subreddit_diversity = df['subreddit'].nunique()
        insights.append(f"🏷️ **Platform Diversity**: Content spans {subreddit_diversity} different subreddits")
        insights.append(f"🔥 **Hottest Communities**: {', '.join([f'{sub} ({count} posts)' for sub, count in top_subreddits.items()])}")
    else:
        insights.append("No 'subreddit' column for community insights.")
    
    return insights

def generate_mock_insights(df):
    """Generates mock AI insights. This function can be replaced by a real LLM call."""
    # This function is kept separate as it might eventually call an external API
    # and its behavior might differ from the data-driven enhanced insights.
    insights = [
        "Identified a surge in discussions related to 'environmental policy' over the last week.",
        "Sentiment analysis shows a predominantly negative sentiment towards 'economic reforms' in recent posts.",
        "Key influencers include 'UserXYZ' and 'CommunityABC' based on engagement metrics.",
        "Detected emerging topics around 'remote work' and 'future of education' with increasing frequency.",
        "Cross-platform analysis indicates similar trends in 'Twitter' and 'Reddit' regarding 'AI ethics'."
    ]
    return insights[1:4] # Return a subset for demonstration
