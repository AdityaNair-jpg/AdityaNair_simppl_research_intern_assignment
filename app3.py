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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
import os
import random
import warnings
import plotly.graph_objects as go
import altair as alt
# Import AI-related functions from the new file
from ai_insights import generate_enhanced_insights, generate_mock_insights

# Import Network for pyvis
try:
    from pyvis.network import Network
except ImportError:
    st.warning("Pyvis not found. Network graph feature will be disabled. Please install with 'pip install pyvis'")
    Network = None # Set to None if not available

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Social Media Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources and Load VADER model on startup
@st.cache_resource
def load_sentiment_model_and_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon') # Ensure VADER lexicon is downloaded
    st.write("DEBUG: Loading NLTK VADER sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer() # Initialize VADER
    st.write("DEBUG: NLTK VADER sentiment analyzer loaded.")
    return analyzer # Return the VADER analyzer

sentiment_analyzer = load_sentiment_model_and_nltk_resources() # Renamed from sentiment_model to sentiment_analyzer for clarity

# Preprocessing for topic modeling and word cloud
@st.cache_data
def preprocess_text(text_series):
    # Ensure all elements are strings before lowercasing
    text_series = text_series.astype(str).str.lower()
    text_series = text_series.apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE)) # Remove URLs
    text_series = text_series.apply(lambda x: re.sub(r'\S*@\S*\s?', '', x)) # Remove mentions
    text_series = text_series.apply(lambda x: re.sub(r'#\w+', '', x)) # Remove hashtags
    text_series = text_series.apply(lambda x: re.sub(r'[^\w\s]', '', x)) # Remove punctuation
    tokens = text_series.apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    tokens = tokens.apply(lambda x: [word for word in x if word.isalpha() and word not in stop_words])
    return tokens

# Enhanced Topic Modeling with more detailed output
@st.cache_data
def extract_topics_enhanced(processed_texts, n_topics=5):
    # Join tokens back into strings for TF-IDF
    texts_for_tfidf = [" ".join(tokens) for tokens in processed_texts if tokens]
    if not texts_for_tfidf:
        st.warning("DEBUG: No texts available for TF-IDF. Returning empty topics and data.")
        return [], []

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        tfidf = vectorizer.fit_transform(texts_for_tfidf)
        st.write(f"DEBUG: TF-IDF matrix shape: {tfidf.shape}")
    except ValueError as e:
        st.warning(f"Could not create TF-IDF matrix. Not enough text data or all words are stop words. Error: {e}")
        return [], []

    if tfidf.shape[0] < n_topics:
        st.warning(f"DEBUG: Number of documents ({tfidf.shape[0]}) is less than number of topics requested ({n_topics}). Adjusting n_topics.")
        n_topics = tfidf.shape[0] if tfidf.shape[0] > 0 else 1
        if n_topics == 0:
            st.warning("DEBUG: No documents to create topics from.")
            return [], []

    nmf_model = NMF(n_components=n_topics, random_state=1, max_iter=100)
    nmf_model.fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    topic_data = []
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        top_scores = [topic[i] for i in top_words_idx]
        
        topic_name = f"Topic {topic_idx + 1}"
        topic_summary = ", ".join(top_words[:5])
        topics.append(f"{topic_name}: {topic_summary}")
        
        # Store detailed topic data for bubble chart
        for word, score in zip(top_words, top_scores):
            topic_data.append({
                'topic': topic_name,
                'word': word,
                'score': score,
                'topic_id': topic_idx
            })
    
    st.write(f"DEBUG: Extracted {len(topics)} topics and {len(topic_data)} topic-word entries.")
    return topics, topic_data

# Topic Modeling (backward compatibility)
@st.cache_data
def extract_topics(processed_texts, n_topics=5):
    topics, _ = extract_topics_enhanced(processed_texts, n_topics)
    return topics

# Generate sample data function with more realistic posting patterns
def generate_sample_data(num_posts=1000):
    start_date = datetime.now() - timedelta(days=365)
    
    # Create more realistic posting patterns with varying activity levels
    dates = []
    
    # Generate dates with different patterns to simulate real social media activity
    for i in range(num_posts):
        # Create clusters of activity with some random variation
        base_day = np.random.randint(0, 365)
        
        # Add some clustering effect - posts tend to cluster around certain periods
        if np.random.random() < 0.3:  # 30% chance of being in a "viral" period
            # Create clusters of posts within a few days
            cluster_offset = np.random.randint(-3, 4)
            base_day = max(0, min(364, base_day + cluster_offset))
        
        # Add some weekend/weekday patterns
        day_of_week = (start_date + timedelta(days=base_day)).weekday()
        if day_of_week < 5:  # Weekday - more likely to have posts
            if np.random.random() < 0.7:  # 70% chance to keep the date
                pass
            else:
                base_day = np.random.randint(0, 365)  # Random redistribution
        
        # Add hour variation to make it more realistic
        hour_offset = np.random.randint(0, 24) / 24.0
        post_date = start_date + timedelta(days=base_day + hour_offset)
        dates.append(post_date)
    
    # Generate more varied content
    content_templates = [
        "This is amazing news about {topic}! Really excited to see this development.",
        "Just saw the latest update on {topic}. Not sure how I feel about this...",
        "Breaking: New developments in {topic} sector. This could change everything!",
        "Discussing {topic} with colleagues today. Interesting perspectives shared.",
        "My thoughts on {topic}: We need to be more careful about implementation.",
        "Great article about {topic}. Highly recommend reading this analysis.",
        "Concerned about the recent {topic} trends. What are your thoughts?",
        "Celebrating progress in {topic}! This is what we've been working toward.",
        "Quick update on {topic} - things are moving faster than expected.",
        "Deep dive into {topic} research. The data is quite revealing."
    ]
    
    topics = [
        "artificial intelligence", "climate change", "cryptocurrency", "remote work",
        "healthcare technology", "renewable energy", "space exploration", "education reform",
        "digital privacy", "economic policy", "social media", "biotechnology",
        "urban planning", "cybersecurity", "quantum computing", "sustainable agriculture"
    ]
    
    contents = []
    for i in range(num_posts):
        template = np.random.choice(content_templates)
        topic = np.random.choice(topics)
        content = template.format(topic=topic)
        contents.append(content)
    
    # Generate authors with some having more posts than others (realistic distribution)
    author_weights = np.random.zipf(1.5, 99)  # Zipf distribution for realistic author activity
    author_weights = author_weights / author_weights.sum()
    authors = np.random.choice([f"user_{i+1}" for i in range(99)], 
                              size=num_posts, 
                              p=author_weights)
    
    # Sample subreddits with realistic distribution
    subreddit_list = [
        'Technology', 'Politics', 'Science', 'WorldNews', 'Futurology', 
        'Economics', 'History', 'Environment', 'Health', 'Education',
        'Programming', 'DataScience', 'MachineLearning', 'Cryptocurrency',
        'ClimateChange', 'SpaceX', 'Tesla', 'Investing', 'Startups', 'Innovation'
    ]
    
    # Create weighted distribution for subreddits
    subreddit_weights = np.random.dirichlet(np.ones(len(subreddit_list)) * 2)
    subreddits = np.random.choice(subreddit_list, size=num_posts, p=subreddit_weights)

    data = {
        'created_at': dates,
        'content': contents,
        'author': authors,
        'subreddit': subreddits,
        'media_type': np.random.choice(['twitter', 'reddit', 'forum'], num_posts)
    }
    return pd.DataFrame(data)

# Load data
@st.cache_data
def load_data():
    data = []
    st.write("DEBUG (load_data): Attempting to load data from 'data.jsonl'...")
    try:
        with open('data.jsonl', 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                try:
                    loaded_json = json.loads(line)
                    # Handle cases where the actual data might be nested under a 'data' key
                    data_to_append = loaded_json.get('data', loaded_json)
                    data.append(data_to_append)
                except json.JSONDecodeError as e:
                    st.warning(f"DEBUG (load_data): Skipping invalid JSON line {line_num + 1} due to error: {e}. Line starts with: {line.strip()[:100]}...")
                    continue
        st.write(f"DEBUG (load_data): Successfully read {len(data)} JSON objects from 'data.jsonl'.")
    except FileNotFoundError:
        st.error("Error: data.jsonl file not found. Please ensure it's in the same directory as app4.py.")
        st.write("DEBUG (load_data): FileNotFoundError encountered.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.write(f"DEBUG (load_data): General Exception encountered: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    st.write(f"DEBUG (load_data): DataFrame created from loaded data. Initial shape: {df.shape}")

    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at', 'date', 'datetime', 'post_timestamp']
    df['created_at'] = pd.NaT

    for col in timestamp_cols:
        if col in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df[col], unit='s', errors='coerce')
                if df['created_at'].notna().any():
                    break
                else:
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        break
            except ValueError:
                continue

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        st.warning("Warning: No valid timestamp column found or all NaT. Using current date as fallback for sample data.")
        df['created_at'] = pd.to_datetime(pd.Series([datetime.now()] * len(df))) # Fallback for ALL rows if no valid date found

    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message', 'subreddit']
    df['content'] = ''
    for col in text_cols:
        if col in df.columns and not df[col].isnull().all():
            df['content'] = df[col].astype(str).fillna('')
            break
    if df['content'].empty or df['content'].isnull().all():
         st.warning("Warning: No valid text column found or all NaN. Setting 'content' to empty strings.")
         df['content'] = ''

    def get_author(row):
        if 'author' in row and pd.notna(row['author']) and str(row['author']).lower() != 'none':
            return str(row['author'])
        if 'username' in row and pd.notna(row['username']) and str(row['username']).lower() != 'none':
            return str(row['username'])
        if 'user' in row and isinstance(row['user'], dict) and 'screen_name' in row['user'] and pd.notna(row['user']['screen_name']) and str(row['user']['screen_name']).lower() != 'none':
            return str(row['user']['screen_name'])
        if 'crosspost_parent_list' in row and isinstance(row['crosspost_parent_list'], list):
            for parent_post in row['crosspost_parent_list']:
                if isinstance(parent_post, dict) and 'author' in parent_post and pd.notna(parent_post['author']) and str(parent_post['author']).lower() != 'none':
                    return str(parent_post['author'])
        return 'Unknown User'

    df['author'] = df.apply(get_author, axis=1)

    st.write(f"DEBUG (load_data): Final DataFrame shape before return: {df.shape}")
    st.write(f"DEBUG (load_data): Non-NaT 'created_at' count: {df['created_at'].count()}")
    st.write(f"DEBUG (load_data): Non-empty 'content' count: {df['content'].astype(bool).sum()}")
    st.write(f"DEBUG (load_data): Unique authors: {df['author'].nunique()}")
    
    return df

def create_plotly_network_graph(df_for_graph, n_nodes_to_display=50):
    if df_for_graph.empty:
        st.warning("No data available to generate the network graph.")
        return None

    df_for_graph = df_for_graph[~df_for_graph['author'].isin(['None', 'Unknown User'])].copy()
    if df_for_graph.empty:
        st.warning("No valid author data available to generate the network graph after filtering 'None' or 'Unknown User'.")
        return None

    G = nx.Graph()
    all_authors = df_for_graph['author'].unique()
    all_subreddits = df_for_graph['subreddit'].unique()

    for author in all_authors:
        G.add_node(author, type='author', size=10, color='blue')
    for subreddit in all_subreddits:
        G.add_node(subreddit, type='subreddit', size=5, color='red')

    for index, row in df_for_graph.iterrows():
        author = row['author']
        subreddit = row['subreddit']
        if author in G and subreddit in G:
            G.add_edge(author, subreddit, type='posted_in')

    for index, row in df_for_graph.iterrows():
        author = row['author']
        content = row['content']
        mentions = re.findall(r'(?:u/|@)([a-zA-Z0-9_-]+)', content)
        for mentioned_user in mentions:
            if mentioned_user in all_authors and mentioned_user != author:
                if author in G and mentioned_user in G:
                    G.add_edge(author, mentioned_user, type='mentions')
    
    node_degrees = dict(G.degree())
    author_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'author']
    if author_nodes:
        author_degrees = [node_degrees[n] for n in author_nodes]
        max_author_degree = max(author_degrees) if author_degrees else 1
        min_author_degree = min(author_degrees) if author_degrees else 0
        scaled_sizes = []
        for degree in author_degrees:
            if max_author_degree == min_author_degree:
                scaled_sizes.append(20)
            else:
                scaled_size = 10 + (degree - min_author_degree) * (20 / (max_author_degree - min_author_degree))
                scaled_sizes.append(scaled_size)
        author_node_sizes_scaled = {n: s for n, s in zip(author_nodes, scaled_sizes)}
    else:
        author_node_sizes_scaled = {}

    subreddit_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'subreddit']
    subreddit_node_sizes = {n: 8 for n in subreddit_nodes}
    node_sizes = {**author_node_sizes_scaled, **subreddit_node_sizes}

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size_plot = []
    node_type = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} (Type: {G.nodes[node]['type']}, Degree: {node_degrees[node]})")
        node_color.append(G.nodes[node]['color'])
        node_size_plot.append(node_sizes.get(node, 10))
        node_type.append(G.nodes[node]['type'])

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_color,
            size=node_size_plot,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Author Interaction Network Graph',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="This graph visualizes connections between authors based on their shared activity in subreddits and direct mentions.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def fig_to_base64(fig):
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png")
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
    return img_base64

def main():
    st.title("📊 Social Media Analysis Dashboard")

    st.sidebar.header("Data Configuration")
    
    df = load_data()

    if df.empty or not df['created_at'].notna().any() or not df['content'].notna().any():
        st.warning("No valid data loaded or critical columns are missing/empty. Generating sample data for demonstration.")
        df = generate_sample_data(num_posts=1000)
        st.success("Sample data generated successfully!")
        st.write("Sample Data Head (generated):")
        st.dataframe(df.head())
    
    st.write(f"DEBUG (main): Initial DataFrame shape after load/sample: {df.shape}")

    if 'subreddit' not in df.columns:
        st.warning("'subreddit' column not found in data. Some features may be limited.")
        df['subreddit'] = 'Unknown'

    st.sidebar.header("Filter Data")

    # Initialize filtered_df immediately after df is finalized - FIXED FROM APP2
    filtered_df = df.copy()
    
    # Date filtering logic from app2.py - FIXED
    selected_date_range = None
    if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty and filtered_df['created_at'].min() is not pd.NaT:
        min_date = filtered_df['created_at'].min().date()
        max_date = filtered_df['created_at'].max().date()

        if min_date == max_date:
            st.sidebar.write(f"Data available for a single day: {min_date.strftime('%Y-%m-%d')}")
            single_date_input = st.sidebar.date_input(
                "Select Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            # Ensure selected_date_range is always a tuple (start_date, end_date)
            selected_date_range = (single_date_input, single_date_input)
        else:
            selected_date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

        if selected_date_range and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = filtered_df[(filtered_df['created_at'].dt.date >= start_date) &
                                     (filtered_df['created_at'].dt.date <= end_date)].copy()
    else:
        st.sidebar.warning("No valid date column found for filtering.")

    st.write(f"DEBUG (main): DataFrame shape after Date Range filter: {filtered_df.shape}")

    search_query = st.sidebar.text_input("Search keywords (comma-separated):")
    if search_query:
        keywords = [k.strip().lower() for k in search_query.split(',')]
        filtered_df = filtered_df[filtered_df['content'].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )].copy()
    st.write(f"DEBUG (main): DataFrame shape after Keyword filter: {filtered_df.shape}")

    sentiment_options = ['All', 'Positive', 'Neutral', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment:", sentiment_options)

    if not filtered_df.empty:
        if 'sentiment_label' not in filtered_df.columns:
            st.write("DEBUG: Running NLTK VADER sentiment analysis...")
            with st.spinner("Performing sentiment analysis..."):
                filtered_df['content'] = filtered_df['content'].astype(str)
                # Apply VADER sentiment analysis
                filtered_df['compound_score'] = filtered_df['content'].apply(lambda text: sentiment_analyzer.polarity_scores(text)['compound'])
                
                # Categorize sentiment based on compound score
                filtered_df['sentiment_label'] = filtered_df['compound_score'].apply(lambda c: 
                    'Positive' if c >= 0.05 else 
                    'Negative' if c <= -0.05 else 
                    'Neutral'
                )
                filtered_df['sentiment_numeric_score'] = filtered_df['compound_score'] # Use compound for numeric score

            st.write("DEBUG: NLTK VADER sentiment analysis complete.")
        
        df_for_sentiment_charts = filtered_df.copy() 
        
        if selected_sentiment == 'Positive':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Positive'].copy()
        elif selected_sentiment == 'Neutral':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Neutral'].copy()
        elif selected_sentiment == 'Negative':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Negative'].copy()
    else:
        st.info("No data to apply sentiment filter.")
    st.write(f"DEBUG (main): DataFrame shape after Sentiment filter: {filtered_df.shape}")

    all_subreddits_available = df['subreddit'].unique() # Use original df for all subreddits
    selected_subreddits = st.sidebar.multiselect("Filter by Subreddit:", all_subreddits_available, default=list(all_subreddits_available)) # Default to all selected
    filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)].copy()
    st.write(f"DEBUG (main): DataFrame shape after Subreddit filter: {filtered_df.shape}")

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", df.shape[0])
    with col2:
        st.metric("Filtered Posts", filtered_df.shape[0])
    with col3:
        st.metric("Unique Authors (Filtered)", filtered_df['author'].nunique())
    with col4:
        st.metric("Unique Subreddits (Filtered)", filtered_df['subreddit'].nunique())

    tab_activity, tab_sentiment, tab_entities, tab_wordcloud, tab_network, tab_topics, tab_ai = st.tabs([
        "Activity Trends", "Sentiment Analysis", "Top Entities", "Word Cloud",
        "Author Network Graph", "Topic Modeling", "AI Insights"
    ])

    with tab_activity:
        st.header("Post Activity Over Time")
        # Fixed logic from app2.py
        if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
            time_df = filtered_df.copy()
            time_df['date'] = time_df['created_at'].dt.date
            daily_counts = time_df.groupby('date').size().reset_index(name='count')
            
            # Create the line chart
            fig = px.line(daily_counts, x='date', y='count',
                          title='Post Volume Over Time',
                          labels={'date': 'Date', 'count': 'Number of Posts'},
                          markers=True, # Added markers
                          line_shape='spline', # Added spline interpolation
                          hover_data={'date': True, 'count': True} # Show date and count on hover
                          )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key Metrics
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Posts", len(filtered_df))
            if not daily_counts.empty:
                col2.metric("Start Date", daily_counts['date'].min().strftime('%Y-%m-%d'))
                col3.metric("End Date", daily_counts['date'].max().strftime('%Y-%m-%d'))
            
            if st.checkbox("Show Daily Post Counts"):
                st.dataframe(daily_counts)
        else:
            st.info("No data or 'created_at' column available for activity trend analysis after filters.")

    with tab_sentiment:
        st.header("Sentiment Analysis")
        if not df_for_sentiment_charts.empty and 'sentiment_label' in df_for_sentiment_charts.columns:
            sentiment_counts = df_for_sentiment_charts['sentiment_label'].value_counts()
            sentiment_df = pd.DataFrame({
                'Sentiment': sentiment_counts.index,
                'Count': sentiment_counts.values
            })
            
            # DEBUG: Display the DataFrame used for the pie chart
            st.write("DEBUG: DataFrame for Sentiment Pie Chart:")
            st.dataframe(sentiment_df)

            # Explicitly extract labels and values as lists
            pie_labels = sentiment_df['Sentiment'].tolist()
            pie_values = sentiment_df['Count'].tolist()

            fig = go.Figure(data=[go.Pie(
                labels=pie_labels, # Pass explicit list of labels
                values=pie_values, # Pass explicit list of values
                textinfo='label+percent+value', # Use Plotly's built-in textinfo
                insidetextfont=dict(color='white'),
                hoverinfo='label+percent+value',
                hole=.3,
                marker=dict(colors=px.colors.qualitative.Pastel)
            )])
            fig.update_layout(
                title_text='Overall Sentiment Distribution (Before Sentiment Filter)'
            )
            st.plotly_chart(fig, use_container_width=True, key="sentiment_pie_chart")

            st.subheader("Sentiment Over Time")
            if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty and 'sentiment_numeric_score' in filtered_df.columns:
                daily_sentiment = filtered_df.set_index('created_at')['sentiment_numeric_score'].resample('D').mean().reset_index()
                fig_time = px.line(
                    daily_sentiment,
                    x='created_at',
                    y='sentiment_numeric_score',
                    title='Average Sentiment Over Time (NLTK VADER)', # Updated title
                    labels={'created_at': 'Date', 'sentiment_numeric_score': 'Average Sentiment Score'},
                    markers=True, # Added markers
                    line_shape='spline', # Added spline interpolation
                    hover_data={'created_at': '|%Y-%m-%d', 'sentiment_numeric_score': ':.2f'} # Show date and score on hover
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("Date or sentiment numeric score column not available for sentiment over time analysis after filters.")
        else:
            st.info("No data to perform sentiment analysis.")

    with tab_entities:
        st.header("Top Entities/Authors/Subreddits")

        if not filtered_df.empty:
            # === TOP AUTHORS (Altair Interactive Horizontal Bar Chart) ===
            st.subheader("Top 10 Authors")
            authors_filtered = filtered_df[filtered_df['author'] != 'Unknown User']
            author_counts = authors_filtered['author'].value_counts().reset_index()
            author_counts.columns = ['Author', 'Post Count']
            top_authors = author_counts.head(10).copy()
            top_authors['Post Count'] = top_authors['Post Count'].astype(int)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_authors)

            with col2:
                if not top_authors.empty:
                    st.write("DEBUG: Data for authors chart:")
                    st.dataframe(top_authors)
                    st.write(f"DEBUG: Data types: {top_authors.dtypes}")
                    st.write(f"DEBUG: Post Count values: {top_authors['Post Count'].tolist()}")

                    # Altair interactive horizontal bar chart
                    chart_authors_alt = alt.Chart(top_authors).mark_bar().encode(
                        x=alt.X('Post Count:Q', title='Post Count'), # Quantity type for numerical axis
                        y=alt.Y('Author:N', title='Author', sort='-x'), # Nominal type for categorical axis, sorted by x-value
                        tooltip=['Author', 'Post Count'], # Tooltip on hover
                        color=alt.Color('Post Count:Q', scale=alt.Scale(scheme='viridis')), # CORRECTED: Use 'scheme' for color palette
                        text=alt.Text('Post Count:Q', format='d') # Display count on bars
                    ).properties(
                        title='Top 10 Authors by Post Count'
                    ).configure_axis(
                        labelLimit=200 # Allow longer labels if needed
                    ).interactive() # Enable interactivity like zooming and panning

                    st.altair_chart(chart_authors_alt, use_container_width=True)

            # === TOP SUBREDDITS (Altair Interactive Horizontal Bar Chart) ===
            st.subheader("Top 10 Subreddits")
            subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
            subreddit_counts.columns = ['Subreddit', 'Post Count']
            top_subreddits = subreddit_counts.head(10).copy()
            top_subreddits['Post Count'] = top_subreddits['Post Count'].astype(int)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_subreddits)

            with col2:
                if not top_subreddits.empty:
                    st.write("DEBUG: Data for subreddits chart:")
                    st.dataframe(top_subreddits)
                    st.write(f"DEBUG: Data types: {top_subreddits.dtypes}")
                    st.write(f"DEBUG: Post Count values: {top_subreddits['Post Count'].tolist()}")

                    chart_subreddits_alt = alt.Chart(top_subreddits).mark_bar().encode(
                        x=alt.X('Post Count:Q', title='Post Count'),
                        y=alt.Y('Subreddit:N', title='Subreddit', sort='-x'),
                        tooltip=['Subreddit', 'Post Count'],
                        color=alt.Color('Post Count:Q', scale=alt.Scale(scheme='plasma')), # CORRECTED: Use 'scheme'
                        text=alt.Text('Post Count:Q', format='d')
                    ).properties(
                        title='Top 10 Subreddits by Post Count'
                    ).configure_axis(
                        labelLimit=200
                    ).interactive()

                    st.altair_chart(chart_subreddits_alt, use_container_width=True)

            # === FREQUENT WORDS (Altair Interactive Horizontal Bar Chart) ===
            st.subheader("Most Frequent Words (excluding stopwords)")
            processed_texts = preprocess_text(filtered_df['content'])
            all_words = [word for sublist in processed_texts for word in sublist]
            word_freq = Counter(all_words)
            top_words_df = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(top_words_df)

            with col2:
                if not top_words_df.empty:
                    st.write("DEBUG: Data for words chart:")
                    st.dataframe(top_words_df)
                    st.write(f"DEBUG: Data types: {top_words_df.dtypes}")
                    st.write(f"DEBUG: Frequency values: {top_words_df['Frequency'].tolist()}")

                    chart_words_alt = alt.Chart(top_words_df).mark_bar().encode(
                        x=alt.X('Frequency:Q', title='Frequency'),
                        y=alt.Y('Word:N', title='Word', sort='-x'),
                        tooltip=['Word', 'Frequency'],
                        color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='cividis')), # CORRECTED: Use 'scheme'
                        text=alt.Text('Frequency:Q', format='d')
                    ).properties(
                        title='Top 15 Most Frequent Words'
                    ).configure_axis(
                        labelLimit=200
                    ).interactive()

                    st.altair_chart(chart_words_alt, use_container_width=True)

        else:
            st.info("No data available for top entities analysis after filters.")

    with tab_topics:
        st.header("Topic Modeling")
        if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
            st.write(f"DEBUG: filtered_df content head for topic modeling: {filtered_df['content'].head().to_list()}")
            processed_texts = preprocess_text(filtered_df['content'])
            st.write(f"DEBUG: processed_texts head for topic modeling: {processed_texts.head().to_list()}")
            
            # Enhanced topic modeling with bubble chart
            n_topics = st.slider("Number of Topics", min_value=3, max_value=10, value=5)
            topics, topic_data = extract_topics_enhanced(processed_texts, n_topics=n_topics)
            
            if topics and topic_data:
                st.subheader("Extracted Topics")
                for i, topic in enumerate(topics):
                    st.write(f"**{topic}**")
                
                # Create bubble chart for topics
                st.subheader("Topic Visualization - Bubble Chart")
                topic_df = pd.DataFrame(topic_data)
                
                st.write("DEBUG: topic_df head for bubble chart:")
                st.dataframe(topic_df.head())

                if not topic_df.empty:
                    # Create bubble chart
                    fig_bubble = px.scatter(
                        topic_df, 
                        x='topic_id', 
                        y='score',
                        size='score',
                        color='topic',
                        hover_name='word',
                        hover_data={'score': ':.3f'},
                        title='Topic Word Importance (Bubble Size = Importance Score)',
                        labels={'topic_id': 'Topic Number', 'score': 'Word Importance Score'},
                        size_max=60
                    )
                    
                    # Customize the layout
                    fig_bubble.update_layout(
                        height=600,
                        xaxis=dict(
                            tickmode='linear', 
                            tick0=0, 
                            dtick=1,
                            title='Topic Number'
                        ),
                        yaxis=dict(title='Word Importance Score'),
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.01
                        )
                    )
                    
                    # Update x-axis to show proper topic labels
                    topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
                    fig_bubble.update_xaxes(
                        tickvals=list(range(n_topics)),
                        ticktext=topic_labels
                    )
                    
                    st.plotly_chart(fig_bubble, use_container_width=True)
                    
                    # NEW: Alternative - Bar Charts for Each Topic
                    st.subheader("Top Words per Topic (Bar Charts)")
                    unique_topics = topic_df['topic'].unique()
                    for topic_name in unique_topics:
                        topic_words_df = topic_df[topic_df['topic'] == topic_name].sort_values(by='score', ascending=False).head(10)
                        if not topic_words_df.empty:
                            fig_topic_bar = px.bar(
                                topic_words_df,
                                x='score',
                                y='word',
                                orientation='h',
                                title=f'Top Words for {topic_name}',
                                labels={'score': 'Word Importance Score', 'word': 'Word'},
                                color='score',
                                color_continuous_scale='viridis'
                            )
                            fig_topic_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                            st.plotly_chart(fig_topic_bar, use_container_width=True)

                    # Topic word details table
                    if st.checkbox("Show Topic Word Details"):
                        st.subheader("Topic Word Scores")
                        # Group by topic and show top words
                        for topic_id, group in topic_df.groupby('topic_id'):
                            with st.expander(f"Topic {topic_id + 1} - Word Details"):
                                topic_words = group.sort_values('score', ascending=False).head(10)
                                st.dataframe(topic_words[['word', 'score']].reset_index(drop=True))
                else:
                    st.warning("Could not generate topic data for visualization.")
            else:
                st.warning("Could not extract topics. Not enough text data or all words are stop words.")
        else:
            st.warning("No content available for topic modeling after filters.")

    with tab_ai:
        st.header("AI-Generated Insights")
        
        # Enhanced insights section
        if not filtered_df.empty:
            st.subheader("📊 Automated Data Insights")
            
            with st.spinner("Analyzing data patterns..."):
                enhanced_insights = generate_enhanced_insights(filtered_df)
            
            # Display insights in an organized manner
            st.markdown("### Key Findings:")
            for insight in enhanced_insights:
                st.markdown(f"• {insight}")
            
            # Additional analysis sections
            st.markdown("---")
            
            # Content analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Content Metrics")
                if 'content' in filtered_df.columns:
                    content_lengths = filtered_df['content'].str.len()
                    fig_content = px.histogram(
                        x=content_lengths,
                        nbins=30,
                        title='Distribution of Post Lengths',
                        labels={'x': 'Character Count', 'y': 'Number of Posts'}
                    )
                    st.plotly_chart(fig_content, use_container_width=True)
            
            with col2:
                st.subheader("📅 Temporal Patterns")
                if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
                    hourly_activity = filtered_df.copy()
                    hourly_activity['hour'] = hourly_activity['created_at'].dt.hour
                    hourly_counts = hourly_activity['hour'].value_counts().sort_index()
                    
                    fig_hourly = px.bar(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        title='Posts by Hour of Day',
                        labels={'x': 'Hour', 'y': 'Number of Posts'}
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Engagement analysis
            st.subheader("🔍 Advanced Analysis")
            
            if st.button("Generate Detailed Report"):
                with st.spinner("Generating comprehensive analysis..."):
                    report = f"""
                    ## Comprehensive Social Media Analysis Report
                    
                    **Dataset Overview:**
                    - Total Posts Analyzed: {len(filtered_df):,}
                    - Date Range: {filtered_df['created_at'].min().strftime('%Y-%m-%d') if 'created_at' in filtered_df.columns else 'N/A'} to {filtered_df['created_at'].max().strftime('%Y-%m-%d') if 'created_df.columns' in filtered_df.columns else 'N/A'}
                    - Unique Authors: {filtered_df['author'].nunique():,}
                    - Communities Covered: {filtered_df['subreddit'].nunique():,}
                    
                    **Content Analysis:**
                    - Average Post Length: {filtered_df['content'].str.len().mean():.0f} characters
                    - Most Active Author: {filtered_df['author'].value_counts().index[0] if not filtered_df['author'].empty else 'N/A'}
                    - Most Popular Community: {filtered_df['subreddit'].value_counts().index[0] if not filtered_df['subreddit'].empty else 'N/A'}
                    
                    **Sentiment Overview:**
                    {filtered_df['sentiment_label'].value_counts().to_string() if 'sentiment_label' in filtered_df.columns else 'Sentiment analysis not available'}
                    
                    **Key Recommendations:**
                    1. Focus content strategy on peak activity hours
                    2. Engage with top contributors to build community
                    3. Monitor sentiment trends for early issue detection
                    4. Leverage popular topics for content planning
                    """
                    
                    st.markdown(report)
                    
                    # Download report
                    st.download_button(
                        label="Download Full Report",
                        data=report,
                        file_name=f"social_media_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        # Original OpenAI integration
        st.markdown("---")
        st.subheader("🤖 OpenAI-Powered Insights")
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:",
                                                type='password')
        if openai_api_key:
            if 'content' in filtered_df.columns and not filtered_df['content'].empty:
                with st.spinner("Generating AI insights..."):
                    insights = generate_mock_insights(filtered_df)
                st.markdown("**AI-Generated Insights:**")
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.warning("No content data available for AI analysis after filters.")
        else:
            st.info("Enter an OpenAI API key in the sidebar to enable AI-generated insights.")

    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample (after all filters)")
        st.dataframe(filtered_df.head(100))

    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()
