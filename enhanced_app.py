import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import nltk
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
import os
import warnings
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import Network for pyvis
try:
    from pyvis.network import Network
except ImportError:
    st.warning("Pyvis not found. Network graph feature will be disabled. Please install with 'pip install pyvis'")
    Network = None # Set to None if not available

# Import transformers for sentiment analysis
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False
    
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Social Media Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Download NLTK resources on startup
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    return True


# Initialize Hugging Face sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    if transformers_available:
        try:
            # Use a lightweight model for sentiment analysis
            sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None
            )
            return sentiment_analyzer
        except Exception as e:
            st.warning(f"Failed to load Hugging Face model: {e}")
            return None
    return None


# Load data
@st.cache_data
def load_data():
    data = []
    try:
        with open('data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    # Safely extract data, handling missing 'data' key
                    data.append(json_obj.get('data', json_obj.get('data', json_obj)))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        st.error("Error: data.jsonl file not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Convert timestamp (prioritize specific columns)
    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at',
                      'date', 'datetime', 'post_timestamp']
    df['created_at'] = pd.NaT  # Initialize with NaT (Not a Time)

    for col in timestamp_cols:
        if col in df.columns:
            try:
                # Try parsing as Unix timestamp first
                df['created_at'] = pd.to_datetime(df[col], unit='s', errors='coerce')
                if df['created_at'].notna().any():
                    break  # Stop if successful
                else:
                    # If Unix conversion fails, try default parsing
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        break
                    else:
                        pass # Continue to next column if this one also fails
            except ValueError as e:
                pass # Continue to next column on error

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        print("Warning: No valid timestamp column found. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message']
    df['content'] = None  # Initialize 'content' column
    for col in text_cols:
        if col in df.columns:
            df['content'] = df[col].astype(str).fillna('')
            break
    if 'content' not in df.columns or df['content'].isnull().all():
        print("Warning: No valid text column found.")
        df['content'] = ''

    # Extract user information
    user_cols = ['user_id', 'author', 'username', 'user.screen_name']
    df['user_id'] = None  # Initialize 'user_id'
    for col in user_cols:
        if col in df.columns:
            df['user_id'] = df[col].astype(str).fillna('Unknown User')
            break
    if 'user_id' not in df.columns or df['user_id'].isnull().all():
        print("Warning: No valid user ID column found.")
        df['user_id'] = 'Unknown User'
        
    # Extract subreddit information if available
    subreddit_cols = ['subreddit', 'community', 'forum']
    df['subreddit'] = None  # Initialize 'subreddit'
    for col in subreddit_cols:
        if col in df.columns:
            df['subreddit'] = df[col].astype(str).fillna('Unknown')
            break
    if 'subreddit' not in df.columns or df['subreddit'].isnull().all():
        print("Warning: No valid subreddit column found.")
        df['subreddit'] = 'Unknown'

    # Add a media_type column if not present
    if 'media_type' not in df.columns:
        # Try to infer from other columns
        if 'subreddit' in df.columns and df['subreddit'].notna().any():
            df['media_type'] = 'reddit'
        elif 'tweet_id' in df.columns or 'twitter_id' in df.columns:
            df['media_type'] = 'twitter'
        else:
            df['media_type'] = 'unknown'

    return df


# Function to process the DataFrame
def process_data(df, search_term="", date_range=None):
    filtered_df = df.copy()

    # Date filtering
    if 'created_at' in filtered_df.columns and date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) &
            (filtered_df['created_at'].dt.date <= end_date)
        ]

    # Search filtering
    if search_term and 'content' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['content'].str.contains(
            search_term, case=False, na=False)]

    return filtered_df


# Function to generate a wordcloud
def generate_wordcloud(text_data, max_words=100, stopwords_list=None):
    text = " ".join(text_data.dropna().astype(str))
    if not text:
        return None
        
    if stopwords_list is None:
        stopwords_list = stopwords.words('english')
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=max_words, 
        contour_width=3,
        contour_color='steelblue',
        stopwords=stopwords_list
    ).generate(text)
    
    return wordcloud


# Function to process text for topic modeling
def preprocess_text(text_series):
    download_nltk_resources()  # Ensure resources are downloaded
    stop_words = set(stopwords.words('english'))
    # Add common social media terms to stopwords
    additional_stopwords = {'http', 'https', 'www', 'com', 'amp', 'rt', 'like', 'just'}
    stop_words.update(additional_stopwords)
    
    processed_texts = []

    for text in text_series:
        if isinstance(text, str) and text.strip():
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords and short words
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            # Join back to string
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        else:
            processed_texts.append("")

    return processed_texts


# Analyze sentiment using Hugging Face (primary) or simple rule-based (fallback)
def analyze_sentiment(text, sentiment_analyzer=None):
    if not isinstance(text, str) or text.strip() == "":
        return {"label": "neutral", "score": 0.5}
        
    # Use Hugging Face if available
    if sentiment_analyzer is not None:
        try:
            result = sentiment_analyzer(text)
            # Handle different output formats from different models
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
                
            # Standardize the output
            if 'label' in result:
                label = result['label'].lower()
                # Map label to positive/negative/neutral
                if 'positive' in label:
                    return {"label": "positive", "score": result.get('score', 0.8)}
                elif 'negative' in label:
                    return {"label": "negative", "score": result.get('score', 0.8)}
                else:
                    return {"label": "neutral", "score": result.get('score', 0.5)}
            else:
                # Fallback for unexpected format
                return {"label": "neutral", "score": 0.5}
        except Exception as e:
            st.warning(f"Error in sentiment analysis, using fallback: {e}")
            # Fall through to rule-based approach
    
    # Simple rule-based fallback
    positive_words = set(['good', 'great', 'excellent', 'love', 'like', 'best', 'amazing', 'awesome'])
    negative_words = set(['bad', 'terrible', 'hate', 'awful', 'worst', 'horrible', 'poor'])
    
    text = text.lower()
    words = set(re.findall(r'\b\w+\b', text))
    
    pos_matches = len(words.intersection(positive_words))
    neg_matches = len(words.intersection(negative_words))
    
    if pos_matches > neg_matches:
        return {"label": "positive", "score": 0.7}
    elif neg_matches > pos_matches:
        return {"label": "negative", "score": 0.7}
    else:
        return {"label": "neutral", "score": 0.5}


# Function to extract topics using NMF with error handling
def extract_topics(documents, n_topics=5, n_top_words=10):
    if not documents or all(doc.strip() == "" for doc in documents):
        return ["No sufficient text to extract topics."]
    
    # Filter out empty documents
    documents = [doc for doc in documents if doc.strip()]
    
    if len(documents) < n_topics:
        # Not enough documents for the requested number of topics
        n_topics = max(2, len(documents) // 2)
    
    try:
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_df=0.95,  # Ignore terms that appear in >95% of documents
            min_df=2,     # Ignore terms that appear in <2 documents
            stop_words='english',
            max_features=5000  # Limit to 5000 features to prevent memory issues
        )
        
        tfidf = vectorizer.fit_transform(documents)
        
        if tfidf.shape[1] == 0:
            return ["Not enough meaningful words found after filtering."]
            
        # Ensure we don't request more topics than features
        n_topics = min(n_topics, tfidf.shape[1], len(documents))
        
        # Apply NMF with regularization to prevent overfitting
        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            alpha=0.1,  # L2 regularization
            max_iter=300
        )
        
        nmf.fit(tfidf)
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(", ".join(top_words))
            
        return topics
        
    except Exception as e:
        st.error(f"Error during topic extraction: {str(e)}")
        return [f"Error extracting topics: {str(e)}"]


# Function to create network visualization using Pyvis
def create_network_graph(df, selected_date_range, media_type):
    if Network is None:
        st.warning("Pyvis library not found. Network graph cannot be displayed.")
        return None

    if df.empty:
        return None
        
    try:
        start_date, end_date = selected_date_range
        
        # Filter data by date range
        df_filtered = df[
            (df['created_at'].dt.date >= start_date) & 
            (df['created_at'].dt.date <= end_date)
        ]

        # Further filter by media type
        if 'media_type' in df_filtered.columns and media_type != 'All':
            df_filtered = df_filtered[df_filtered['media_type'] == media_type]

        if df_filtered.empty:
            return None

        # Create a graph
        G = nx.Graph()

        # Add nodes and edges
        user_interactions = {}
        
        for _, row in df_filtered.iterrows():
            author = str(row.get('user_id', 'Unknown User'))
            
            # Make sure the author is added as a node
            if author not in G.nodes:
                G.add_node(author)
            
            content = str(row.get('content', ''))
            
            # Look for mentions (users preceded by @)
            mentions = re.findall(r'@(\w+)', content)
            for mention in mentions:
                # Add mentioned user as a node
                if mention not in G.nodes:
                    G.add_node(mention)
                
                # Track this interaction
                interaction_key = tuple(sorted([author, mention]))
                user_interactions[interaction_key] = user_interactions.get(interaction_key, 0) + 1

            # Also check for replies/comments to build connections
            replied_to = row.get('replied_to', row.get('parent_id', None))
            if replied_to and replied_to in df_filtered['user_id'].values:
                # Find the author of the parent post
                parent_author = df_filtered[df_filtered['user_id'] == replied_to]['user_id'].iloc[0]
                parent_author = str(parent_author)
                
                if parent_author not in G.nodes:
                    G.add_node(parent_author)
                    
                # Track this interaction
                interaction_key = tuple(sorted([author, parent_author]))
                user_interactions[interaction_key] = user_interactions.get(interaction_key, 0) + 1

        # Add weighted edges based on interactions
        for (user1, user2), weight in user_interactions.items():
            G.add_edge(user1, user2, weight=weight)

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        if not G.nodes:
            st.warning("No connections to display for the selected date range and media type.")
            return None

        # Create a PyVis network
        net = Network(
            height="500px",
            width="100%",
            directed=False,
            bgcolor="#222222",
            font_color="white",
            notebook=True
        )

        # Compute centrality metrics for better visualization
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(30, len(G.nodes)))
        except Exception as e:
            st.warning(f"Error computing network metrics: {e}. Using basic visualization.")
            degree_centrality = {node: 1.0 for node in G.nodes()}
            betweenness_centrality = {node: 1.0 for node in G.nodes()}

        # Add nodes to the network with attributes
        for node in G.nodes():
            size = 10 + (degree_centrality.get(node, 0) * 50)
            net.add_node(
                node,
                size=size,
                title=f"User: {node}<br>Connections: {G.degree[node]}",
                color=f"rgba(173, 216, 230, {min(1.0, 0.3 + betweenness_centrality.get(node, 0) * 2)})"
            )

        # Add edges with weights
        for u, v, data in G.edges(data=True):
            width = 1 + data.get('weight', 1) * 0.5  # Scale edge width
            net.add_edge(u, v, width=width, title=f"Interactions: {data.get('weight', 1)}")

        # Save the graph to an HTML file and return its content
        path = 'pyvis_graph.html'
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content

    except Exception as e:
        st.error(f"Error creating network graph: {e}")
        return None


# AI-generated insights function
def generate_insights(df):
    insights = []
    
    try:
        # 1. Time-based insights
        if 'created_at' in df.columns and not df['created_at'].empty:
            daily_counts = df.groupby(df['created_at'].dt.date).size()
            if not daily_counts.empty:
                peak_day = daily_counts.idxmax()
                peak_count = daily_counts.max()
                avg_posts = daily_counts.mean()
                insights.append(f"Peak activity occurred on {peak_day} with {peak_count} posts (avg: {avg_posts:.1f}/day).")
                
                # Time trends
                if len(daily_counts) > 1:
                    first_day_count = daily_counts.iloc[0]
                    last_day_count = daily_counts.iloc[-1]
                    trend = "increasing" if last_day_count > first_day_count else "decreasing"
                    insights.append(f"Post activity shows a {trend} trend over the selected period.")
        
        # 2. Content insights
        if 'content' in df.columns and not df['content'].empty:
            # Common words analysis
            all_text = " ".join(df['content'].dropna().astype(str))
            words = word_tokenize(all_text.lower())
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
            most_common = Counter(words).most_common(5)
            if most_common:
                common_words = ", ".join([f"{word} ({count})" for word, count in most_common])
                insights.append(f"Most frequently used words: {common_words}.")
            
            # Sentiment analysis
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                most_common_sentiment = sentiment_counts.idxmax()
                sentiment_percentage = (sentiment_counts[most_common_sentiment] / len(df)) * 100
                insights.append(f"The dominant sentiment is {most_common_sentiment} ({sentiment_percentage:.1f}% of posts).")
        
        # 3. User insights
        if 'user_id' in df.columns:
            user_counts = df['user_id'].value_counts()
            top_users = user_counts.head(3)
            if not top_users.empty:
                top_users_str = ", ".join([f"{user} ({count})" for user, count in top_users.items()])
                insights.append(f"Most active users: {top_users_str}.")
            # User engagement calculation
            unique_users = df['user_id'].nunique()
            total_posts = len(df)
            avg_posts_per_user = total_posts / unique_users if unique_users > 0 else 0
            insights.append(f"Average of {avg_posts_per_user:.1f} posts per user across {unique_users} unique users.")

        # 4. Subreddit/Community insights
        if 'subreddit' in df.columns and df['subreddit'].nunique() > 1:
            subreddit_counts = df['subreddit'].value_counts()
            top_subreddits = subreddit_counts.head(3)
            if not top_subreddits.empty:
                top_subreddits_str = ", ".join([f"{sr} ({count})" for sr, count in top_subreddits.items()])
                insights.append(f"Most active communities: {top_subreddits_str}.")
    except Exception as e:
        insights.append(f"Error generating insights: {str(e)}")

    # Return at least one insight if not insights:
    if not insights:
        insights = ["Insufficient data for meaningful insights."]
    return insights



# Main Streamlit app
def main():
    st.title("📊 Enhanced Social Media Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes social media data to uncover patterns in information spread, user engagement, content trends, and sentiment analysis.
    """)

    # Initialize critical resources
    download_nltk_resources()
    sentiment_analyzer = load_sentiment_analyzer()

    # Load data with spinner
    with st.spinner("Loading data..."):
        df = load_data()

    # Sidebar for filters and options
    st.sidebar.title("Controls & Filters")
    # Check if data loaded correctly
    if df.empty or not df['created_at'].notna().any() or not df['content'].notna().any():
        st.warning("No valid data loaded or critical columns are missing/empty. Generating sample data for demonstration.")
        # Generate sample data for demonstration
        np.random.seed(42)
        sample_size = 1000
        # Create date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 1)
        date_range = (end_date - start_date).days
        # Generate random dates
        random_days = np.random.randint(0, date_range, sample_size)
        dates = [start_date + timedelta(days=day) for day in random_days]
        # Create sample content
        topics = ["politics", "technology", "health", "entertainment", "sports"]
        content_templates = [
            "Just read an article about #{}. Very interesting!",
            "Anyone else following the {} news today?",
            "Can't believe what's happening with {} right now!",
            "New developments in {} are concerning.",
            "I think {} is going to be huge this year."
        ]
        contents = []
        for _ in range(sample_size):
            topic = np.random.choice(topics)
            template = np.random.choice(content_templates)
            contents.append(template.format(topic))
        # Generate user IDs and subreddits
        user_ids = [f"user_{i}" for i in np.random.randint(1, 100, sample_size)]
        subreddits = np.random.choice(['AskReddit', 'news', 'worldnews', 'funny', 'pics', 'gaming'], sample_size)
        # Create DataFrame
        df = pd.DataFrame({
            'created_at': dates,
            'content': contents,
            'user_id': user_ids,
            'subreddit': subreddits,
'media_type': np.random.choice(['twitter', 'reddit', 'forum'], sample_size)
        })
        st.success("Generated sample data for demonstration.")

    # Date range filter
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Filter by Date Range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Search term filter
    search_term = st.sidebar.text_input("Filter by Search Term", "")
    
    # Media type filter
    media_types = ['All'] + list(df['media_type'].unique())
    selected_media_type = st.sidebar.selectbox("Filter by Media Type", media_types)

    # Process data with filters
    filtered_df = process_data(df, search_term, selected_date_range)
    
    # Apply sentiment analysis
    if not filtered_df.empty:
        with st.spinner("Analyzing sentiment..."):
            filtered_df['sentiment'] = filtered_df['content'].apply(
                lambda x: analyze_sentiment(x, sentiment_analyzer)['label']
            )
            filtered_df['sentiment_score'] = filtered_df['content'].apply(
                lambda x: analyze_sentiment(x, sentiment_analyzer)['score']
            )

    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview",
        "Time Analysis",
        "Content Analysis",
        "User Analysis",
        "Network Analysis",
        "Topic Modeling",
        "AI Insights"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Overview")
        st.write(f"Data from {min_date} to {max_date}")
        st.write(f"Total posts: {len(df)}")
        st.write(f"Unique users: {df['user_id'].nunique()}")
        st.write(f"Unique subreddits/communities: {df['subreddit'].nunique()}")

        # Display sample of the data
        st.subheader("Sample Data")
        st.dataframe(filtered_df.head(5), use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        if not filtered_df.empty:
            st.write(filtered_df.describe())
        else:
            st.write("No data to display after filtering.")

    # Tab 2: Time Analysis
    with tab2:
        st.header("Time Analysis")
        if not filtered_df.empty:
            # Daily post counts
            daily_counts = filtered_df.groupby(filtered_df['created_at'].dt.date).size()
            daily_counts = daily_counts.reset_index(name='counts')
            daily_counts.columns = ['date', 'counts']  # Rename for clarity
            
            # Weekly post counts
            weekly_counts = filtered_df.groupby(filtered_df['created_at'].dt.isocalendar().week).size()
            weekly_counts = weekly_counts.reset_index(name='counts')
            weekly_counts.columns = ['week', 'counts']
            
            # Monthly post counts
            monthly_counts = filtered_df.groupby(filtered_df['created_at'].dt.month).size()
            monthly_counts = monthly_counts.reset_index(name='counts')
            monthly_counts.columns = ['month', 'counts']
            
            # Create sub-columns
            col1, col2, col3 = st.columns(3)
            
            # Daily counts chart
            with col1:
                st.subheader("Daily Post Counts")
                fig_daily = px.line(daily_counts, x='date', y='counts', title="Daily Post Counts")
                fig_daily.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Posts",
                    xaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # Weekly counts chart
            with col2:
                st.subheader("Weekly Post Counts")
                fig_weekly = px.bar(weekly_counts, x='week', y='counts', title="Weekly Post Counts")
                fig_weekly.update_layout(
                    xaxis_title="Week",
                    yaxis_title="Number of Posts",
                    xaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Monthly counts chart
            with col3:
                st.subheader("Monthly Post Counts")
                fig_monthly = px.bar(monthly_counts, x='month', y='counts', title="Monthly Post Counts")
                fig_monthly.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Posts",
                    xaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
                
        else:
            st.warning("No data to display for the selected date range.")

    # Tab 3: Content Analysis
    with tab3:
        st.header("Content Analysis")
        
        if not filtered_df.empty:
            # Sentiment analysis
            sentiment_counts = filtered_df['sentiment'].value_counts().to_frame(name='counts')
            sentiment_counts['percentage'] = (sentiment_counts['counts'] / len(filtered_df)) * 100
            
            # Word Cloud
            text_data = filtered_df['content']
            wordcloud = generate_wordcloud(text_data)
            
            # Create sub-columns
            sentiment_col, wordcloud_col = st.columns(2)
            
            # Sentiment analysis chart
            with sentiment_col:
                st.subheader("Sentiment Analysis")
                if wordcloud is not None:
                    fig_sentiment = px.pie(
                        sentiment_counts,
                        values='counts',
                        names=sentiment_counts.index,
                        title='Sentiment Distribution',
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                    )
                    fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                    fig_sentiment.update_layout(
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0.05)',
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                else:
                    st.warning("Not enough text data to generate sentiment analysis.")
            
            # Word Cloud
            with wordcloud_col:
                st.subheader("Word Cloud")
                if wordcloud is not None:
                    plt.figure(figsize=(8, 4))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt.gcf(), use_container_width=True)
                else:
                    st.warning("Not enough text data to generate word cloud.")
        else:
            st.warning("No data to display for the selected filters.")

    # Tab 4: User Analysis
    with tab4:
        st.header("User Analysis")
        if not filtered_df.empty:
            # User post counts
            user_counts = filtered_df['user_id'].value_counts().head(10).to_frame(name='counts')
            user_counts['user_id'] = user_counts.index  # Make user_id a column
            
            # Subreddit post counts
            subreddit_counts = filtered_df['subreddit'].value_counts().head(10).to_frame(name='counts')
            subreddit_counts['subreddit'] = subreddit_counts.index
            
            # Create sub-columns
            user_col, subreddit_col = st.columns(2)
            
            # User post counts chart
            with user_col:
                st.subheader("Top 10 Most Active Users")
                fig_user_posts = px.bar(
                    user_counts,
                    x='user_id',
                    y='counts',
                    title="Top 10 Most Active Users",
                    labels={'user_id': 'User ID', 'counts': 'Number of Posts'},
                )
                fig_user_posts.update_layout(
                    xaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                )
                st.plotly_chart(fig_user_posts, use_container_width=True)
            
            # Subreddit post counts chart
            with subreddit_col:
                st.subheader("Top 10 Most Active Subreddits")
                fig_subreddit_posts = px.bar(
                    subreddit_counts,
                    x='subreddit',
                    y='counts',
                    title="Top 10 Most Active Subreddits",
                    labels={'subreddit': 'Subreddit', 'counts': 'Number of Posts'}
                )
                fig_subreddit_posts.update_layout(
                    xaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    yaxis=dict(gridcolor='rgba(200,200,200,0.2)'),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                )
                st.plotly_chart(fig_subreddit_posts, use_container_width=True)
        else:
            st.warning("No data to display for the selected filters.")

    # Tab 5: Network Analysis
    with tab5:
        st.header("Network Analysis")
        if not filtered_df.empty:
            st.subheader("User Interaction Network")
            network_html = create_network_graph(filtered_df, selected_date_range, selected_media_type)
            if network_html:
                st.components.v1.html(network_html, height=600, scrolling=True)
            else:
                st.warning("No network graph to display.  Select a wider date range, or different media type.")
        else:
            st.warning("No data to display for network analysis with the selected filters.")

    # Tab 6: Topic Modeling
    with tab6:
        st.header("Topic Modeling")
        
        if 'content' in filtered_df.columns and not filtered_df.empty:
            # Topic modeling parameters
            topic_col1, topic_col2 = st.columns(2)
            num_topics = topic_col1.slider("Number of Topics", min_value=2, max_value=10, value=5)
            num_top_words = topic_col2.slider("Number of Top Words", min_value=5, max_value=20, value=10)
            
            # Preprocess the text data
            processed_texts = preprocess_text(filtered_df['content'])
            
            # Extract topics
            topics = extract_topics(processed_texts, n_topics=num_topics, n_top_words=num_top_words)
            
            # Display topics
            if topics and not any("Error" in topic for topic in topics): # Check for errors
                st.subheader(f"Top {num_topics} Topics")
                for i, topic in enumerate(topics):
                    st.write(f"**Topic {i + 1}:** {topic}")
            else:
                st.error(topics[0]) # show the first error message
        else:
            st.warning("No data to display for topic modeling.  Make sure your data has a 'content' column, and that it is not empty.")

    # Tab 7: AI Insights
    with tab7:
        st.header("AI Insights")
        with st.spinner("Generating insights..."):
            insights = generate_insights(filtered_df)
        for insight in insights:
            st.markdown(f"- {insight}")



if __name__ == "__main__":
    main()
