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


# Download NLTK resources on startup
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon') # Ensure VADER lexicon is downloaded
    return SentimentIntensityAnalyzer() # Cache the analyzer itself

# Initialize SIA globally using the cached function
sia = download_nltk_resources()


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
                    else:
                        pass
            except ValueError as e:
                pass

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        print("Warning: No valid timestamp column found. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message', 'subreddit']
    df['content'] = None
    for col in text_cols:
        if col in df.columns:
            df['content'] = df[col].astype(str).fillna('')
            break
    if 'content' not in df.columns or df['content'].isnull().all():
        print("Warning: No valid text column found.")
        df['content'] = ''

    # Extract user information
    user_cols = ['author', 'username', 'user.screen_name']
    df['author'] = None
    for col in user_cols:
        if col in df.columns:
            df['author'] = df[col].astype(str).fillna('Unknown User')
            break
    if 'author' not in df.columns or df['author'].isnull().all():
        print("Warning: No valid user ID column found.")
        df['author'] = 'Unknown User'

    return df


# Function to process the DataFrame (this is NOT cached)
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
def generate_wordcloud(text_data):
    text = " ".join(text_data.dropna().astype(str))
    if not text:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_width=3,
                          contour_color='steelblue').generate(text)
    return wordcloud


# Function to process text for topic modeling
def preprocess_text(text_series):
    stop_words = set(stopwords.words('english'))
    processed_texts = []

    for text in text_series:
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words]
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        else:
            processed_texts.append("")

    return processed_texts

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    try:
        score = sia.polarity_scores(text)
        compound = score['compound']
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        st.error(f"Sentiment analysis error for text: '{text[:50]}...' - {e}. Returning neutral.")
        return "neutral"


# Function to extract topics using NMF
def extract_topics(documents, n_topics=5):
    if not documents or all(doc.strip() == "" for doc in documents):
        return ["No sufficient text to extract topics."]
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform(documents)

        nmf = NMF(n_components=n_topics, random_state=42)
        nmf.fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(", ".join(top_words))

        return topics
    except Exception as e:
        st.error(f"Error during topic extraction: {e}")
        return ["Error extracting topics."]


def create_plotly_network_graph(df_for_graph):
    """
    Generates an interactive network graph of author and subreddit interactions using Plotly.
    Authors are connected to subreddits they post in, and to other authors they mention.
    """
    if df_for_graph.empty:
        st.warning("No data available to build the network graph after filters.")
        return None
    
    # Ensure necessary columns are available
    if 'author' not in df_for_graph.columns or 'content' not in df_for_graph.columns or 'subreddit_name_prefixed' not in df_for_graph.columns:
        st.error("Missing 'author', 'content', or 'subreddit_name_prefixed' columns for network analysis.")
        return None

    G = nx.Graph()

    # Use sets to collect unique authors and subreddits for efficiency
    all_authors = set()
    all_subreddits = set()

    for _, row in df_for_graph.iterrows():
        author = str(row['author']).strip()
        subreddit = str(row['subreddit_name_prefixed']).strip()
        
        if author and author.lower() != 'unknown user':
            all_authors.add(author)
        if subreddit and subreddit != 'None':
            all_subreddits.add(subreddit)
    
    # Add author nodes to the graph
    for author in all_authors:
        G.add_node(author, type='author')
    
    # Add subreddit nodes to the graph
    # Prepend a distinct prefix to subreddit node IDs to avoid clashes with author names
    # and to easily identify them.
    for subreddit in all_subreddits:
        subreddit_node_id = f"r/{subreddit}" # Example: r/politics
        G.add_node(subreddit_node_id, type='subreddit')

    # --- DEBUG: Initial Node Counts ---
    st.write(f"DEBUG: Number of unique authors found in data: {len(all_authors)}")
    st.write(f"DEBUG: Number of unique subreddits found in data: {len(all_subreddits)}")
    st.write(f"DEBUG: Number of nodes in graph G after initial creation (before edges): {G.number_of_nodes()}")
    st.write(f"DEBUG: Number of author nodes in G (before edges): {len([n for n, data in G.nodes(data=True) if data.get('type') == 'author'])}")
    st.write(f"DEBUG: Number of subreddit nodes in G (before edges): {len([n for n, data in G.nodes(data=True) if data.get('type') == 'subreddit'])}")


    # Add edges: Author to Subreddit (posts in), Author to Author (mentions)
    for _, row in df_for_graph.iterrows():
        author = str(row['author']).strip()
        subreddit = str(row['subreddit_name_prefixed']).strip()
        content = str(row['content']).strip()

        if author and author.lower() != 'unknown user':
            # 1. Author posts in Subreddit edge
            subreddit_node_id = f"r/{subreddit}"
            if G.has_node(author) and G.has_node(subreddit_node_id):
                G.add_edge(author, subreddit_node_id, relationship='posts_in')

            # 2. Author to Author (mentions) edge
            mentions = re.findall(r'@(\w+)', content)
            for mention in mentions:
                mention = mention.strip()
                if mention and mention != author and mention.lower() != 'unknown user':
                    if G.has_node(mention) and G.has_node(author): # Ensure both nodes exist
                         G.add_edge(author, mention, relationship='mentions')
    
    # Remove self-loops (authors mentioning themselves, etc.)
    G.remove_edges_from(nx.selfloop_edges(G))

    # --- DEBUG: Graph G after adding edges ---
    st.write(f"DEBUG: Number of nodes in graph G after adding edges: {G.number_of_nodes()}")
    st.write(f"DEBUG: Number of edges in graph G: {G.number_of_edges()}")
    st.write(f"DEBUG: Number of author nodes in G (after edges): {len([n for n, data in G.nodes(data=True) if data.get('type') == 'author'])}")


    if not G.nodes():
        st.warning("No interactions or valid authors/subreddits to display for the selected filters.")
        return None
    
    """
    # --- Handle large graphs for better visualization performance ---
    # This block is TEMPORARILY COMMENTED OUT for debugging purposes.
    # It will be re-enabled or modified later.
    if G.number_of_nodes() > 500: # Adjust this limit as needed
        st.info(f"The network graph has {G.number_of_nodes()} nodes. Displaying the largest connected component (or a sample) for better performance.")
        components = list(nx.connected_components(G))
        if components:
            largest_component_nodes = max(components, key=len)
            G = G.subgraph(largest_component_nodes).copy()
            
            if G.number_of_nodes() > 500: # If largest component is still too big, sample
                st.info(f"Largest component ({G.number_of_nodes()} nodes) is still too large. Randomly sampling 500 nodes.")
                sampled_nodes = random.sample(list(G.nodes()), 500)
                G = G.subgraph(sampled_nodes).copy()
        else:
            st.warning("No connected components found in the graph. Graph might be too sparse.")
            return None
    """
    
    if G.number_of_nodes() == 0:
        st.warning("No connections found after processing for the network graph.")
        return None

    # Compute positions for nodes
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    # Prepare lists for Plotly traces, separating authors and subreddits
    author_node_x, author_node_y, author_node_text, author_node_degrees = [], [], [], []
    subreddit_node_x, subreddit_node_y, subreddit_node_text, subreddit_node_sizes = [], [], [], []

    edge_x = []
    edge_y = []

    # Populate node and edge data
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown') 
        x, y = pos[node]

        if node_type == 'author':
            author_node_x.append(x)
            author_node_y.append(y)
            degree = G.degree[node]
            author_node_degrees.append(degree)
            author_node_text.append(f"Author: {node}<br>Connections: {degree}")
        elif node_type == 'subreddit':
            subreddit_node_x.append(x)
            subreddit_node_y.append(y)
            subreddit_degree = G.degree[node]
            subreddit_node_sizes.append(max(15, subreddit_degree * 4 + 15)) 
            subreddit_node_text.append(f"Subreddit: {node.replace('r/', '')}<br>Connections: {subreddit_degree}")

    # --- DEBUG: Plotting Data Lists ---
    st.write(f"DEBUG: Length of author_node_x: {len(author_node_x)}")
    st.write(f"DEBUG: Length of author_node_degrees: {len(author_node_degrees)}")
    if author_node_degrees:
        st.write(f"DEBUG: Sample author degrees (first 10): {author_node_degrees[:min(10, len(author_node_degrees))]}")
    else:
        st.write("DEBUG: author_node_degrees is empty.")
    
    # Calculate author_node_sizes_scaled here to ensure it's debugged
    if author_node_degrees:
        max_author_degree = max(author_node_degrees)
    else:
        max_author_degree = 1 # Default to 1 to prevent division by zero

    author_node_sizes_scaled = [max(8, (d / max_author_degree) * 20 + 8) for d in author_node_degrees]

    st.write(f"DEBUG: Length of author_node_sizes_scaled: {len(author_node_sizes_scaled)}")
    if author_node_sizes_scaled:
        st.write(f"DEBUG: Sample author sizes (first 10): {author_node_sizes_scaled[:min(10, len(author_node_sizes_scaled))]}")
    else:
        st.write("DEBUG: author_node_sizes_scaled is empty.")

    if author_node_x:
        st.write(f"DEBUG: Sample author_node_x (first 10): {author_node_x[:min(10, len(author_node_x))]}")
        st.write(f"DEBUG: Sample author_node_y (first 10): {author_node_y[:min(10, len(author_node_y))]}")
    else:
        st.write("DEBUG: author_node_x is empty.")

    st.write(f"DEBUG: Length of subreddit_node_x: {len(subreddit_node_x)}")
    st.write(f"DEBUG: Length of subreddit_node_sizes: {len(subreddit_node_sizes)}")

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None) 
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    # --- Create Plotly Traces ---

    # Edge Trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Author Nodes Trace
    author_node_trace = go.Scatter(
        x=author_node_x, y=author_node_y,
        mode='markers',
        hoverinfo='text',
        text=author_node_text,
        marker=dict(
            showscale=True, 
            colorscale='YlGnBu', 
            reversescale=True,
            color=author_node_degrees, 
            size=author_node_sizes_scaled, 
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Author Connections',
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2,
            line_color='darkblue'
        ),
        name='Authors' 
    )

    # Subreddit Nodes Trace
    subreddit_node_trace = go.Scatter(
        x=subreddit_node_x, y=subreddit_node_y,
        mode='markers',
        hoverinfo='text',
        text=subreddit_node_text,
        marker=dict(
            color='rgb(255, 100, 100)', 
            size=subreddit_node_sizes, 
            symbol='square', 
            line_width=1,
            line_color='black'
        ),
        name='Subreddits' 
    )

    # Create the Plotly figure
    fig = go.Figure(
        data=[edge_trace, author_node_trace, subreddit_node_trace], 
        layout=go.Layout(
            title=dict(
                text='Author-Subreddit-Mention Network',
                font_size=20 
            ),
            showlegend=True, 
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Nodes are authors (circles, blue scale by connections) and subreddits (squares, red).<br>Edges connect authors to subreddits they post in, and authors to other authors they mention.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    align="left",
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

# AI-generated insights mock function (since we don't have actual OpenAI
# integration)
def generate_mock_insights(df):
    if 'created_at' in df.columns and not df['created_at'].empty:
        daily_counts = df.groupby(df['created_at'].dt.date).size()
        if not daily_counts.empty:
            peak_day = daily_counts.idxmax()
            peak_count = daily_counts.max()
        else:
            peak_day = "unknown"
            peak_count = 0
    else:
        peak_day = "unknown"
        peak_count = 0

    if 'content' in df.columns and not df['content'].empty:
        all_text = " ".join(df['content'].dropna().astype(str))
        words = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalpha() and word not in
                 stop_words and len(word) > 2]
        most_common = Counter(words).most_common(3)
        common_words = ", ".join([word for word, _ in most_common])
    else:
        common_words = "unknown"

    insights = [
        f"Peak activity occurred on {peak_day} with {peak_count} posts.",
        f"The most frequently discussed topics include {common_words}.",
        "Analysis shows a pattern of content sharing that indicates potential network effects.",
        "Information tends to spread rapidly within 24-48 hours of initial posting.",
        "Community clustering suggests distinct echo chambers in the dataset."
    ]

    return insights


# Main Streamlit app
def main():
    st.title("📊 Social Media Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes social media data to uncover patterns in information spread, focusing on potentially unreliable sources and trending topics.
    """)

    # Sidebar for filters and options
    st.sidebar.title("Controls & Filters")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    # Check if data loaded correctly
    if df.empty or not df['created_at'].notna().any() or not df['content'].notna().any():
        st.warning("No valid data loaded or critical columns are missing/empty. Generating sample data for demonstration.")

        np.random.seed(42)
        sample_size = 1000

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 1)
        date_range_days = (end_date - start_date).days

        random_days = np.random.randint(0, date_range_days, sample_size)
        dates = [start_date + timedelta(days=day) for day in random_days]

        topics = ["politics", "technology", "health", "entertainment", "sports"]
        # Modified sample content to include more potential mentions for network graph testing
        content_templates = [
            "Just read an article about #{} by @userA. Very interesting!",
            "Anyone else following the {} news today? @userB has some insights.",
            "Can't believe what's happening with {} right now! Discuss with @userC.",
            "New developments in {} are concerning, as @userD pointed out. Also @userE.",
            "I think {} is going to be huge this year. Thanks @userF!"
        ]

        contents = []
        # Create a pool of sample users for mentions
        sample_mention_users = [f"mention_user_{i}" for i in range(1, 20)] # 19 unique users for mentions
        for i in range(sample_size):
            topic = np.random.choice(topics)
            template = np.random.choice(content_templates)
            # Randomly select a user from the pool for the mention
            mention_user = np.random.choice(sample_mention_users)
            contents.append(template.format(topic, mention_user)) # Pass mention_user to format if template expects it

        authors = [f"user_{i}" for i in np.random.randint(1, 100, sample_size)]

        df = pd.DataFrame({
            'created_at': dates,
            'content': contents,
            'author': authors,
            'media_type': np.random.choice(['twitter', 'reddit', 'forum'], sample_size)
        })

        st.success("Sample data generated successfully!")
        st.write("Sample Data Head:")
        st.dataframe(df.head())

    # Initialize filtered_df immediately after df is finalized
    filtered_df = df.copy()

    # Display data info
    st.sidebar.markdown(f"**Total Posts**: {len(filtered_df)}")

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
        # If no valid date column, ensure selected_date_range has a default value for network graph
        if not df.empty and 'created_at' in df.columns and df['created_at'].notna().any():
            selected_date_range = (df['created_at'].min().date(), df['created_at'].min().date())
        else:
            selected_date_range = (datetime.now().date(), datetime.now().date()) # Fallback to today if no date data


    search_term = st.sidebar.text_input("Search in content", "")
    if search_term:
        if 'content' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['content'].str.contains(
                search_term, case=False, na=False)].copy()
            st.sidebar.markdown(
                f"Found {len(filtered_df)} posts containing '{search_term}'")
        else:
            st.sidebar.warning("Content column not found for search.")

    selected_media_type = 'All'
    if 'media_type' in df.columns and not df['media_type'].empty:
        media_types = ['All'] + list(df['media_type'].unique())
        selected_media_type = st.sidebar.selectbox("Filter by Media Type", media_types)
        if selected_media_type != 'All':
            filtered_df = filtered_df[filtered_df['media_type'] == selected_media_type].copy()
    else:
        st.sidebar.info("No 'media_type' column found in your data.")


    with st.spinner("Analyzing sentiment..."):
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            filtered_df['sentiment'] = filtered_df['content'].apply(analyze_sentiment)
        else:
            filtered_df['sentiment'] = 'neutral'
            st.warning("No content to analyze for sentiment after filtering.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Time Series Analysis", "Content Analysis",
         "Network Analysis", "Topic Modeling", "AI Insights"])

    with tab1:
        st.header("Post Activity Over Time")

        if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
            time_df = filtered_df.copy()
            time_df['date'] = time_df['created_at'].dt.date
            daily_counts = time_df.groupby('date').size().reset_index(
                name='count')

            fig = px.line(daily_counts, x='date', y='count',
                          title='Post Volume Over Time',
                          labels={'date': 'Date', 'count': 'Number of Posts'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Posts", len(filtered_df))
            col2.metric("Start Date", daily_counts['date'].min().strftime('%Y-%m-%d'))
            col3.metric("End Date", daily_counts['date'].max().strftime('%Y-%m-%d'))

            if st.checkbox("Show Daily Post Counts"):
                st.dataframe(daily_counts)
        else:
            st.warning("No valid date information available for time series analysis after filtering.")

    with tab2:
        st.header("Content Analysis")
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            st.subheader("Word Cloud")
            text_data = filtered_df['content']
            wordcloud_img = generate_wordcloud(text_data)
            if wordcloud_img:
                plt.imshow(wordcloud_img, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt.gcf())
            else:
                st.info("Not enough text data to generate a meaningful word cloud.")

            st.subheader("Most Common Words")
            all_text = " ".join(filtered_df['content'].dropna().astype(str))
            if all_text.strip():
                words = word_tokenize(all_text.lower())
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word.isalpha() and word not in
                         stop_words and len(word) > 2]
                word_counts = Counter(words)
                if word_counts:
                    most_common = pd.DataFrame(word_counts.most_common(20),
                                               columns=['word', 'count'])
                    fig_bar = px.bar(most_common, x='word', y='count',
                                      title='Top 20 Most Frequent Words')
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No common words found after processing.")
            else:
                st.info("No content available to determine most common words.")

            st.subheader("Sentiment Distribution")

            filtered_df['sentiment'] = filtered_df['sentiment'].str.strip().str.lower()

            sentiment_counts_dict = filtered_df['sentiment'].value_counts().to_dict()
            sentiment_labels = list(sentiment_counts_dict.keys())
            sentiment_values = list(sentiment_counts_dict.values())

            if sentiment_labels and sentiment_values:
                fig_sentiment = go.Figure(data=[go.Pie(
                    labels=sentiment_labels,
                    values=sentiment_values,
                    hole=.3
                )])
                fig_sentiment.update_layout(title_text='Post Sentiment Breakdown')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.warning("No sentiment data available to plot after filtering.")
        else:
            st.warning("No content available for analysis after filtering.")

    with tab3:
        st.header("User Interaction Network")
        if 'author' in filtered_df.columns and not filtered_df['author'].empty and Network is not None:
            # We now pass the already filtered_df to the graph function
            # This means the graph will be built only from the data currently in view based on sidebar filters.
            
            st.write("This graph visualizes connections between authors based on mentions (`@`) and shared subreddits.")
            
            # Add a button to trigger graph generation as it can be resource intensive
            with tab3: # Or whatever your network graph tab is named
                st.header("Author Interaction Network Graph")
        st.write("This graph visualizes connections between authors based on their shared activity in subreddits and direct mentions.")
        
        if st.button("Generate Author Network Graph (Plotly)"):
            with st.spinner("Building network graph... This might take a moment for large datasets."):
                # This call returns a Plotly Figure object
                network_fig = create_plotly_network_graph(filtered_df) 
                if network_fig:
                    # Display the Plotly Figure using st.plotly_chart
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.info("Network graph could not be generated. Please check data filters or ensure required columns are available.")
           

        elif Network is None:
            st.warning("Pyvis library is not installed. Please install it to enable network analysis (`pip install pyvis`).")
        else:
            st.warning("No user information available for network analysis after filtering. Please ensure 'author' column is populated.")


    with tab4:
        st.header("Topic Modeling")
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            processed_texts = preprocess_text(filtered_df['content'])
            topics = extract_topics(processed_texts, n_topics=5)
            st.subheader("Extracted Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i + 1}: {topic}")
        else:
            st.warning("No content available for topic modeling after filtering.")

    with tab5:
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
                st.warning("No content data available for AI analysis after filtering.")
        else:
            st.info(
                "Enter an OpenAI API key in the sidebar to enable AI-generated insights.")

    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        st.dataframe(filtered_df.head(100))

    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()