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
stop_words_nltk = set(stopwords.words('english'))
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
def load_nltk_resources():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        nltk.download('vader_lexicon', quiet=True)
    try:
        nltk.data.find('corpora/stopwords.zip')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt.zip')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    return SentimentIntensityAnalyzer()

sid = load_nltk_resources() # Call this to ensure downloads happen
# Download NLTK resources and Load VADER model on startup
@st.cache_resource
def load_sentiment_model_and_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon') # Ensure VADER lexicon is downloaded
    analyzer = SentimentIntensityAnalyzer() # Initialize VADER
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
    # Ensure processed_texts is a Series of lists before attempting to join
    if not isinstance(processed_texts, pd.Series):
        st.error("Expected 'processed_texts' to be a Pandas Series in extract_topics_enhanced.")
        return [], [], None, []

    # Filter out empty lists and join tokens into strings
    # Store original indices to map back later
    # Only keep indices where the list of tokens is not empty
    original_indices = processed_texts.index[processed_texts.apply(lambda x: bool(x))].tolist()
    texts_for_tfidf = [" ".join(tokens) for tokens in processed_texts if tokens]

    if not texts_for_tfidf:
        st.warning("No texts available for TF-IDF. Returning empty topics and data.")
        return [], [], None, []

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        tfidf = vectorizer.fit_transform(texts_for_tfidf)
    except ValueError as e:
        st.warning(f"Could not create TF-IDF matrix. Not enough text data or all words are stop words. Error: {e}")
        return [], [], None, []

    if tfidf.shape[0] < n_topics:
        n_topics = tfidf.shape[0] if tfidf.shape[0] > 0 else 1
        if n_topics == 0:
            st.warning("No documents to create topics from.")
            return [], [], None, []

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
    
    # Get document-topic distribution
    doc_topic_dist = nmf_model.transform(tfidf) # Each row is a document, columns are topic probabilities
    
    return topics, topic_data, doc_topic_dist, original_indices


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
    try:
        with open('data.jsonl', 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                try:
                    loaded_json = json.loads(line)
                    # Handle cases where the actual data might be nested under a 'data' key
                    data_to_append = loaded_json.get('data', loaded_json)
                    data.append(data_to_append)
                except json.JSONDecodeError as e:
                    st.warning(f"Skipping invalid JSON line {line_num + 1} due to error: {e}. Line starts with: {line.strip()[:100]}...")
                    continue
    except FileNotFoundError:
        st.error("Error: data.jsonl file not found. Please ensure it's in the same directory as app4.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

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
    
    if 'subreddit' not in df.columns:
        st.warning("'subreddit' column not found in data. Some features may be limited.")
        df['subreddit'] = 'Unknown'

    st.sidebar.header("Filter Data")

    # Date Range Slider
    # --- CORRECTED BLOCK: Uses 'created_at' directly for filtering ---
    if 'created_at' in df.columns and pd.api.types.is_datetime64_any_dtype(df['created_at']):
        # Extract just the date part for min/max values for the slider
        # .dt.date converts datetime to date objects (e.g., 2023-10-26 14:30:45 -> 2023-10-26)
        min_date_val = df['created_at'].dt.date.min()
        max_date_val = df['created_at'].dt.date.max()

        # Convert these date objects back to datetime objects for the slider's `min_value` and `max_value` arguments
        min_date_slider = pd.to_datetime(min_date_val)
        max_date_slider = pd.to_datetime(max_date_val)

        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date_slider.to_pydatetime(),
            max_value=max_date_slider.to_pydatetime(),
            value=(min_date_slider.to_pydatetime(), max_date_slider.to_pydatetime()),
            format="YYYY-MM-DD"
        )
        
        # Apply the filter using the date part of 'created_at' from the DataFrame
        # Compare df['created_at'].dt.date (a date object) with the .date() of the slider's datetime objects
        df = df[(df['created_at'].dt.date >= date_range[0].date()) & (df['created_at'].dt.date <= date_range[1].date())]
    else:
        st.sidebar.warning("Date column ('created_at') not available or not in datetime format for filtering. Please check your data loading.")
    # --- END CORRECTED BLOCK ---

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

    search_query = st.sidebar.text_input("Search keywords (comma-separated):")
    if search_query:
        keywords = [k.strip().lower() for k in search_query.split(',')]
        filtered_df = filtered_df[filtered_df['content'].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )].copy()

    sentiment_options = ['All', 'Positive', 'Neutral', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment:", sentiment_options)

    if not filtered_df.empty:
        if 'sentiment_label' not in filtered_df.columns:
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
        
        df_for_sentiment_charts = filtered_df.copy() 
        
        if selected_sentiment == 'Positive':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Positive'].copy()
        elif selected_sentiment == 'Neutral':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Neutral'].copy()
        elif selected_sentiment == 'Negative':
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'Negative'].copy()
    else:
        st.info("No data to apply sentiment filter.")

    all_subreddits_available = df['subreddit'].unique() # Use original df for all subreddits
    selected_subreddits = st.sidebar.multiselect("Filter by Subreddit:", all_subreddits_available, default=list(all_subreddits_available)) # Default to all selected
    filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)].copy()

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

    with tab_wordcloud:
        st.header("Word Cloud")
        if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
            processed_texts = preprocess_text(filtered_df['content'])
            all_words = " ".join([word for sublist in processed_texts for word in sublist])
            if all_words:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.warning("No meaningful words to generate word cloud after preprocessing and filters.")
        else:
            st.warning("No content data available for word cloud generation after filters.")

    with tab_network:
        st.header("Author Interaction Network Graph")
        if Network is None:
            st.error("Pyvis library not found. Please install it (`pip install pyvis`) to enable the network graph feature.")
        else:
            if 'author' in filtered_df.columns and not filtered_df['author'].empty and 'subreddit' in filtered_df.columns and not filtered_df['subreddit'].empty:
                df_for_graph = filtered_df[['author', 'subreddit', 'content']].copy()
                fig = create_plotly_network_graph(df_for_graph)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    img_base64 = fig_to_base64(fig)
                    st.download_button(
                        label="Download Graph as PNG",
                        data=base64.b64decode(img_base64),
                        file_name="author_network_graph.png",
                        mime="image/png"
                    )
                else:
                    st.info("Network graph could not be generated with the current filters.")
            else:
                st.warning("Author and/or Subreddit data missing or empty for graph generation after filters.")

    with tab_topics:
        st.header("Topic Modeling")
        if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
            # Ensure 'created_at' column is datetime type
            # --- MODIFIED: Check for 'created_at' instead of 'date_only' ---
            if 'created_at' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['created_at']):
                st.warning("The 'created_at' column is not available or not in datetime format. Topic prevalence over time chart cannot be generated.")
                st.stop() # Stop execution if date_only is missing or incorrect
            # --- END MODIFIED ---
            
            # Use original filtered_df to align with doc_topic_dist later
            # Filter out empty content directly from filtered_df to maintain index alignment
            filtered_df_for_topics = filtered_df[filtered_df['content'].apply(lambda x: bool(x and x.strip()))].copy()
            
            processed_texts = preprocess_text(filtered_df_for_topics['content'])
            
            n_topics = st.slider("Number of Topics", min_value=3, max_value=10, value=5, key='num_topics_slider')
            
            # --- MODIFIED CALL TO EXTRACT TOPICS ---
            topics, topic_data, doc_topic_dist, original_indices = extract_topics_enhanced(
                processed_texts, n_topics=n_topics
            )
            # ---------------------------------------

            if topics and topic_data is not None and doc_topic_dist is not None: # Check for None for doc_topic_dist
                st.subheader("Extracted Topics Overview")
                st.info("Each topic represents a cluster of words frequently appearing together. The summary provides the top 5 words for quick understanding.")
                
                # Initialize session state for custom topic names if not already present
                # or if the number of topics changed
                if 'custom_topic_names' not in st.session_state or len(st.session_state.custom_topic_names) != n_topics:
                    st.session_state.custom_topic_names = [f"Topic {i+1}: {', '.join([d['word'] for d in topic_data if d['topic_id'] == i][:5])}" for i in range(n_topics)]
                    
                st.markdown("---")
                st.subheader("Customize Topic Names")
                st.write("Review the top words for each topic below and provide a more descriptive name if you wish. This name will appear in the charts.")
                
                # Allow user to input custom names for each topic
                current_topic_names = []
                for i in range(n_topics):
                    # Ensure topic_id is correct for indexing topic_data
                    topic_summary_words = [d['word'] for d in topic_data if d['topic_id'] == i][:5]
                    default_summary = f"Topic {i+1}: {', '.join(topic_summary_words)}"
                    
                    current_topic_name = st.text_input(
                        f"Name for Topic {i+1}",
                        value=st.session_state.custom_topic_names[i],
                        key=f'topic_name_input_{i}'
                    )
                    current_topic_names.append(current_topic_name)
                st.session_state.custom_topic_names = current_topic_names
                
                # Update topic_df with the custom topic labels for word importance charts
                topic_df = pd.DataFrame(topic_data)
                topic_df['topic_label'] = topic_df['topic_id'].apply(lambda x: st.session_state.custom_topic_names[x])

                st.markdown("---")
                st.subheader("Topic Prevalence Over Time")
                st.write("This chart shows how the activity/discussion around each topic has evolved over the selected time period.")

                if doc_topic_dist is not None and not filtered_df_for_topics.empty:
                    # Create a DataFrame for document-topic distribution with original indices
                    doc_topic_df = pd.DataFrame(doc_topic_dist, index=original_indices)
                    
                    # Align doc_topic_df with filtered_df_for_topics by index
                    # This ensures correct dates are matched to topic distributions
                    doc_topic_df_aligned = filtered_df_for_topics.loc[original_indices, ['created_at']].copy()
                    
                    # Assign topic probabilities to the aligned DataFrame
                    for i in range(n_topics):
                        doc_topic_df_aligned[f'topic_{i}_prob'] = doc_topic_df[i]

                    # MODIFIED LINE: Select only topic probability columns for sum
                    topic_prob_cols = [f'topic_{i}_prob' for i in range(n_topics)]
                    topic_prevalence_daily = doc_topic_df_aligned.groupby(doc_topic_df_aligned['created_at'].dt.floor('D'))[topic_prob_cols].sum().reset_index()
                    # END MODIFIED LINE

                    topic_prevalence_daily = topic_prevalence_daily.rename(columns={'created_at': 'Date'})

                    # Rename topic probability columns to custom topic names
                    new_cols = {f'topic_{i}_prob': st.session_state.custom_topic_names[i] for i in range(n_topics)}
                    topic_prevalence_daily = topic_prevalence_daily.rename(columns=new_cols)

                    # Melt the DataFrame for Altair for multi-line chart
                    # --- MODIFIED: Use 'Date' column ---
                    topic_prevalence_melted = topic_prevalence_daily.melt(
                        id_vars=['Date'], # Use the new 'Date' column
                        var_name='Topic',
                        value_name='Prevalence'
                    )
                    # --- END MODIFIED ---

                    if not topic_prevalence_melted.empty:
                        # Altair Line Chart for Topic Prevalence Over Time
                        # --- MODIFIED: Use 'Date:T' for x-axis and tooltip ---
                        chart_prevalence_alt = alt.Chart(topic_prevalence_melted).mark_line(point=True).encode(
                            x=alt.X('Date:T', title='Date'), # Temporal type for date
                            y=alt.Y('Prevalence:Q', title='Topic Prevalence (Sum of Probabilities)'),
                            color=alt.Color('Topic:N', title='Topic', scale=alt.Scale(scheme='category20')), # Use a categorical scheme
                            tooltip=[
                                alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                                'Topic',
                                alt.Tooltip('Prevalence:Q', format='.2f')
                            ]
                        ).properties(
                            title='Topic Prevalence Over Time'
                        ).interactive()
                        # --- END MODIFIED ---

                        st.altair_chart(chart_prevalence_alt, use_container_width=True)
                    else:
                        st.warning("Could not generate topic prevalence data for chart.")
                else:
                    st.warning("Document-topic distribution data is missing or empty. Cannot generate prevalence over time chart.")

                # === NEW SECTION: POLITICAL SUBREDDIT ANALYSIS ===
                st.markdown("---")
                st.subheader("🏛️ Political Subreddit Topic Distribution")
                st.write("This analysis shows which topics are more prevalent in different political communities.")
                
                # Define political subreddit mappings
                political_subreddits = {
                    'Liberal': ['liberal', 'democrats', 'progressive', 'politics', 'politicalhumor', 'marchagainsttrump'],
                    'Conservative': ['conservative', 'republican', 'conservatives', 'the_donald', 'asktrumpsupporters', 'trump'],
                    'Neoliberal': ['neoliberal', 'centrist', 'moderatepolitics', 'politicaldiscussion'],
                    'Socialist': ['socialism', 'socialist', 'democraticsocialism', 'latestagecapitalism', 'antiwork', 'marxism']
                }
                
                # Create reverse mapping for subreddit to political lean
                subreddit_to_political = {}
                for political_lean, subs in political_subreddits.items():
                    for sub in subs:
                        subreddit_to_political[sub.lower()] = political_lean
                
                # Add political lean column to the aligned dataframe
                if 'subreddit' in filtered_df_for_topics.columns:
                    doc_topic_df_aligned['subreddit'] = filtered_df_for_topics.loc[original_indices, 'subreddit']
                    doc_topic_df_aligned['political_lean'] = doc_topic_df_aligned['subreddit'].str.lower().map(subreddit_to_political)
                    
                    # Filter for only political subreddits
                    political_posts = doc_topic_df_aligned[doc_topic_df_aligned['political_lean'].notna()].copy()
                    
                    if not political_posts.empty:
                        # Calculate average topic probabilities by political lean
                        topic_prob_cols = [f'topic_{i}_prob' for i in range(n_topics)]
                        political_topic_means = political_posts.groupby('political_lean')[topic_prob_cols].mean().reset_index()
                        
                        # Rename columns to use custom topic names
                        rename_dict = {f'topic_{i}_prob': st.session_state.custom_topic_names[i] for i in range(n_topics)}
                        political_topic_means = political_topic_means.rename(columns=rename_dict)
                        
                        # Melt for visualization
                        political_topic_melted = political_topic_means.melt(
                            id_vars=['political_lean'],
                            var_name='Topic',
                            value_name='Average_Probability'
                        )
                        
                        # Create heatmap using Altair
                        st.subheader("Political Lean × Topic Heatmap")
                        heatmap_chart = alt.Chart(political_topic_melted).mark_rect().encode(
                            x=alt.X('political_lean:N', title='Political Lean'),
                            y=alt.Y('Topic:N', title='Topic'),
                            color=alt.Color('Average_Probability:Q', 
                                          scale=alt.Scale(scheme='viridis'),
                                          title='Avg Probability'),
                            tooltip=['political_lean', 'Topic', alt.Tooltip('Average_Probability', format='.3f')]
                        ).properties(
                            title='Topic Distribution Across Political Subreddits',
                            width=400,
                            height=300
                        )
                        
                        st.altair_chart(heatmap_chart, use_container_width=True)
                        
                        # Create grouped bar chart for comparison
                        st.subheader("Topic Probability by Political Lean")
                        bar_chart = alt.Chart(political_topic_melted).mark_bar().encode(
                            x=alt.X('political_lean:N', title='Political Lean'),
                            y=alt.Y('Average_Probability:Q', title='Average Topic Probability'),
                            color=alt.Color('political_lean:N', 
                                          scale=alt.Scale(scheme='category10'),
                                          title='Political Lean'),
                            column=alt.Column('Topic:N', title='Topic'),
                            tooltip=['political_lean', 'Topic', alt.Tooltip('Average_Probability', format='.3f')]
                        ).properties(
                            width=120,
                            height=200,
                            title='Topic Probabilities Across Political Communities'
                        ).resolve_scale(
                            y='independent'  # Allow different y-scales for each topic
                        )
                        
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        # Show detailed statistics
                        if st.checkbox("Show Political Subreddit Statistics"):
                            st.subheader("Detailed Statistics")
                            
                            # Show counts by political lean
                            political_counts = political_posts['political_lean'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Posts by Political Lean:**")
                                for lean, count in political_counts.items():
                                    st.write(f"- {lean}: {count:,} posts")
                            
                            with col2:
                                st.write("**Subreddit Classification:**")
                                for lean, subs in political_subreddits.items():
                                    present_subs = [s for s in subs if s.lower() in filtered_df['subreddit'].str.lower().values]
                                    if present_subs:
                                        st.write(f"**{lean}:** {', '.join(present_subs)}")
                            
                            # Show raw data table
                            st.subheader("Average Topic Probabilities by Political Lean")
                            display_df = political_topic_means.set_index('political_lean')
                            st.dataframe(display_df.round(4))
                            
                            # Find most distinctive topics for each political lean
                            st.subheader("Most Distinctive Topics by Political Lean")
                            for lean in political_topic_means['political_lean'].unique():
                                lean_data = political_topic_means[political_topic_means['political_lean'] == lean]
                                topic_cols = [col for col in lean_data.columns if col != 'political_lean']
                                
                                # Get the topic with highest probability for this lean
                                max_topic_idx = lean_data[topic_cols].iloc[0].idxmax()
                                max_prob = lean_data[topic_cols].iloc[0].max()
                                
                                st.write(f"**{lean}**: {max_topic_idx} (probability: {max_prob:.3f})")
                    
                    else:
                        st.info("No posts found from recognized political subreddits in the current dataset. The analysis works best with data from subreddits like: r/conservative, r/liberal, r/neoliberal, r/socialism, etc.")
                        
                        # Show available subreddits for reference
                        available_subs = filtered_df['subreddit'].value_counts().head(10)
                        st.write("**Available subreddits in your data:**")
                        for sub, count in available_subs.items():
                            st.write(f"- r/{sub}: {count} posts")
                else:
                    st.warning("Subreddit information not available for political analysis.")

                st.markdown("---")
                st.subheader("Top Words per Topic (Interactive Bar Charts)")
                st.write("Explore the most significant words and their importance scores for each individual topic below.")

                # Ensure topics are sorted by their original ID for consistent display
                unique_topics_with_labels = sorted([(d['topic_id'], d['topic_label']) for d in topic_df[['topic_id', 'topic_label']].drop_duplicates().to_dict(orient='records')], key=lambda x: x[0])

                for topic_id, topic_label in unique_topics_with_labels:
                    topic_words_df = topic_df[topic_df['topic_id'] == topic_id].sort_values(by='score', ascending=False).head(10)
                    if not topic_words_df.empty:
                        chart_topic_bar_alt = alt.Chart(topic_words_df).mark_bar().encode(
                            x=alt.X('score:Q', title='Importance Score'),
                            y=alt.Y('word:N', title='Word', sort='-x'), # Sort by score descending
                            tooltip=['word', alt.Tooltip('score', format='.3f')],
                            color=alt.Color('score:Q', scale=alt.Scale(scheme='viridis')), # Use a color scheme
                            text=alt.Text('score:Q', format='.2f') # Display score on bars
                        ).properties(
                            title=f'{topic_label}' # Use the custom topic name as title
                        ).configure_axis(
                            labelLimit=200
                        ).interactive()

                        st.altair_chart(chart_topic_bar_alt, use_container_width=True)

                # Topic word details table
                if st.checkbox("Show Raw Topic Word Details"):
                    st.subheader("Topic Word Scores Table")
                    st.write("This table provides the raw importance scores for each word within its topic.")
                    # Group by topic and show top words
                    for topic_id, group in topic_df.groupby('topic_id'):
                        with st.expander(f"{st.session_state.custom_topic_names[topic_id]} - Word Details"):
                            topic_words = group.sort_values('score', ascending=False).head(10)
                            st.dataframe(topic_words[['word', 'score']].reset_index(drop=True))
            else:
                st.warning("Could not extract topics or document-topic distributions. Not enough text data or all words are stop words. Please ensure your data is clean and sufficient for topic modeling.")
        else:
            st.warning("No content available for topic modeling after filters. Please upload data and apply filters to analyze topics.")
            
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