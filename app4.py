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
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Re-added for VADER
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
# Removed: import transformers
# Removed: from transformers import pipeline 

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

# Topic Modeling
@st.cache_data
def extract_topics(processed_texts, n_topics=5):
    # Join tokens back into strings for TF-IDF
    texts_for_tfidf = [" ".join(tokens) for tokens in processed_texts if tokens]
    if not texts_for_tfidf:
        return []

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    try:
        tfidf = vectorizer.fit_transform(texts_for_tfidf)
    except ValueError:
        st.warning("Could not create TF-IDF matrix. Not enough text data or all words are stop words.")
        return []

    nmf_model = NMF(n_components=n_topics, random_state=1)
    nmf_model.fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics


# Generate sample data function (kept for fallback)
def generate_sample_data(num_posts=1000):
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_posts)]
    contents = [f"This is a sample post content {i}. It talks about politics, tech, and general news. With some positive words like amazing and good, and some negative words like terrible." for i in range(num_posts)]
    
    # Generate authors from 'user_1' to 'user_99'
    authors = [f"user_{np.random.randint(1, 100)}" for _ in range(num_posts)]
    
    # Sample subreddits
    subreddits = np.random.choice([
        'Anarchism', 'Libertarian', 'Socialism', 'Politics', 'Technology', 
        'Science', 'WorldNews', 'Futurology', 'Economics', 'History'
    ], num_posts)

    data = {
        'created_at': dates,
        'content': contents,
        'author': authors,
        'subreddit': subreddits, # Include subreddit for sample data
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

def generate_mock_insights(df):
    insights = [
        "Identified a surge in discussions related to 'environmental policy' over the last week.",
        "Sentiment analysis shows a predominantly negative sentiment towards 'economic reforms' in recent posts.",
        "Key influencers include 'UserXYZ' and 'CommunityABC' based on engagement metrics.",
        "Detected emerging topics around 'remote work' and 'future of education' with increasing frequency.",
        "Cross-platform analysis indicates similar trends in 'Twitter' and 'Reddit' regarding 'AI ethics'."
    ]
    return insights[1:4]

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

    min_date = df['created_at'].min().date() if not df['created_at'].empty and pd.notna(df['created_at'].min()) else datetime.now().date() - timedelta(days=365)
    max_date = df['created_at'].max().date() if not df['created_at'].empty and pd.notna(df['created_at'].max()) else datetime.now().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered_df = df.copy() # Start with a full copy
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Add 1 day to end_date to include the entire end_date
        filtered_df = filtered_df[(filtered_df['created_at'].dt.date >= start_date) & (filtered_df['created_at'].dt.date <= end_date)].copy()
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
        if not filtered_df.empty and 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
            daily_posts = filtered_df.set_index('created_at').resample('D').size().reset_index(name='Post Count')
            fig_activity = px.line(
                daily_posts,
                x='created_at',
                y='Post Count',
                title='Number of Posts Over Time',
                labels={'created_at': 'Date', 'Post Count': 'Number of Posts'},
                markers=True, # Added markers
                line_shape='spline', # Added spline interpolation
                hover_data={'created_at': '|%Y-%m-%d', 'Post Count': True} # Show date and count on hover
            )
            st.plotly_chart(fig_activity, use_container_width=True)
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
            st.subheader("Top 10 Authors")
            top_authors = filtered_df[filtered_df['author'] != 'Unknown User']['author'].value_counts().nlargest(10).reset_index()
            top_authors.columns = ['Author', 'Post Count']
            st.dataframe(top_authors)

            st.subheader("Top 10 Subreddits")
            top_subreddits = filtered_df['subreddit'].value_counts().nlargest(10).reset_index()
            top_subreddits.columns = ['Subreddit', 'Post Count']
            st.dataframe(top_subreddits)

            st.subheader("Most Frequent Words (excluding stopwords)")
            processed_texts = preprocess_text(filtered_df['content'])
            all_words = [word for sublist in processed_texts for word in sublist]
            word_freq = Counter(all_words)
            top_words_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
            st.dataframe(top_words_df)
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
            processed_texts = preprocess_text(filtered_df['content'])
            topics = extract_topics(processed_texts, n_topics=5)
            st.subheader("Extracted Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i + 1}: {topic}")
        else:
            st.warning("No content available for topic modeling after filters.")

    with tab_ai:
        st.header("AI-Generated Insights")
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:",
                                                type='password')
        if openai_api_key:
            if 'content' in filtered_df.columns and not filtered_df['content'].empty:
                with st.spinner("Generating AI insights..."):
                    insights = generate_mock_insights(filtered_df)
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
