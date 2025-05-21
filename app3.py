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

# Initialize SIA globally using the cached resource
sid = download_nltk_resources()

# Preprocessing for sentiment analysis and topic modeling
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
    contents = [f"This is a sample post content {i}. It talks about politics, tech, and general news." for i in range(num_posts)]
    
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
                    json_obj = json.loads(line)
                    data.append(json_obj.get('data', json_obj))
                except json.JSONDecodeError:
                    st.warning(f"DEBUG (load_data): Skipping invalid JSON line {line_num + 1}: {line.strip()[:100]}...")
                    continue
        st.write(f"DEBUG (load_data): Successfully read {len(data)} JSON objects from 'data.jsonl'.")
    except FileNotFoundError:
        st.error("Error: data.jsonl file not found. Please ensure it's in the same directory as app2.py.")
        st.write("DEBUG (load_data): FileNotFoundError encountered.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.write(f"DEBUG (load_data): General Exception encountered: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Convert timestamp (prioritize specific columns)
    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at',
                      'date', 'datetime', 'post_timestamp']
    df['created_at'] = pd.NaT

    for col in timestamp_cols:
        if col in df.columns:
            try:
                # Try parsing as seconds since epoch first
                df['created_at'] = pd.to_datetime(df[col], unit='s', errors='coerce')
                if df['created_at'].notna().any(): # Check if any valid dates were parsed
                    break
                else:
                    # If unit='s' failed or resulted in all NaT, try inferring format
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        break
            except ValueError:
                # Continue to next column if parsing fails completely
                continue

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        st.write("DEBUG (load_data): Warning: No valid timestamp column found or all NaT. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message', 'subreddit']
    df['content'] = '' # Initialize with empty string to avoid NaN issues
    for col in text_cols:
        if col in df.columns and not df[col].isnull().all():
            df['content'] = df[col].astype(str).fillna('')
            break
    if df['content'].empty or df['content'].isnull().all(): # Check if content was truly extracted
         st.write("DEBUG (load_data): Warning: No valid text column found or all NaN. Setting 'content' to empty strings.")
         df['content'] = '' # Ensure it's not None or NaN for subsequent operations

    # --- UPDATED: Extract user information more robustly ---
    def get_author(row):
        # Try top-level 'author' first
        if 'author' in row and pd.notna(row['author']) and row['author'] != 'None':
            return str(row['author'])
        
        # Then try 'username' or 'user.screen_name'
        if 'username' in row and pd.notna(row['username']) and row['username'] != 'None':
            return str(row['username'])
        if 'user' in row and isinstance(row['user'], dict) and 'screen_name' in row['user'] and pd.notna(row['user']['screen_name']) and row['user']['screen_name'] != 'None':
            return str(row['user']['screen_name'])

        # Finally, check 'crosspost_parent_list'
        if 'crosspost_parent_list' in row and isinstance(row['crosspost_parent_list'], list):
            for parent_post in row['crosspost_parent_list']:
                if isinstance(parent_post, dict) and 'author' in parent_post and pd.notna(parent_post['author']) and parent_post['author'] != 'None':
                    return str(parent_post['author'])
        
        return 'Unknown User' # Fallback

    df['author'] = df.apply(get_author, axis=1)

    # --- DEBUG: Raw DataFrame content after loading and initial processing ---
    st.write(f"DEBUG (load_data): Raw DataFrame loaded with {len(df)} rows.")
    if 'author' in df.columns:
        unique_authors_loaded = df['author'].dropna().unique()
        st.write(f"DEBUG (load_data): Unique authors in raw loaded data: {len(unique_authors_loaded)}")
        # Only print sample if number of unique authors is manageable
        if len(unique_authors_loaded) < 50:
            st.write(f"DEBUG (load_data): Sample of unique authors loaded: {unique_authors_loaded[:min(5, len(unique_authors_loaded))]}")
    else:
        st.write("DEBUG (load_data): 'author' column not found in loaded DataFrame before return.")
    
    st.write(f"DEBUG (load_data): 'created_at' column has {df['created_at'].count()} non-NaT values (out of {len(df)}).")
    st.write(f"DEBUG (load_data): 'content' column has {df['content'].count()} non-empty values (out of {len(df)}).")
    
    return df

# Mock function for AI insights
def generate_mock_insights(df):
    insights = [
        "Identified a surge in discussions related to 'environmental policy' over the last week.",
        "Sentiment analysis shows a predominantly negative sentiment towards 'economic reforms' in recent posts.",
        "Key influencers include 'UserXYZ' and 'CommunityABC' based on engagement metrics.",
        "Detected emerging topics around 'remote work' and 'future of education' with increasing frequency.",
        "Cross-platform analysis indicates similar trends in 'Twitter' and 'Reddit' regarding 'AI ethics'."
    ]
    return insights[1:4] # Return a subset of insights for variety

# Graph creation for Plotly Network Graph
def create_plotly_network_graph(df_for_graph, n_nodes_to_display=50): # Added n_nodes_to_display parameter
    if df_for_graph.empty:
        st.warning("No data available to generate the network graph.")
        return None

    # Filter out 'None' or 'Unknown User' authors for graph visualization if they are not meaningful
    df_for_graph = df_for_graph[~df_for_graph['author'].isin(['None', 'Unknown User'])].copy()
    if df_for_graph.empty:
        st.warning("No valid author data available to generate the network graph after filtering 'None' or 'Unknown User'.")
        return None

    st.write(f"DEBUG: Number of unique authors found in data: {df_for_graph['author'].nunique()}")
    st.write(f"DEBUG: Number of unique subreddits found in data: {df_for_graph['subreddit'].nunique()}")

    G = nx.Graph()

    # Add nodes for authors and subreddits
    all_authors = df_for_graph['author'].unique()
    all_subreddits = df_for_graph['subreddit'].unique()

    for author in all_authors:
        G.add_node(author, type='author', size=10, color='blue') # Initial size, will be scaled
    for subreddit in all_subreddits:
        G.add_node(subreddit, type='subreddit', size=5, color='red') # Initial size, constant

    st.write(f"DEBUG: Number of nodes in graph G after initial creation (before edges): {G.number_of_nodes()}")
    st.write(f"DEBUG: Number of author nodes in G (before edges): {len([n for n, data in G.nodes(data=True) if data['type'] == 'author'])}")
    st.write(f"DEBUG: Number of subreddit nodes in G (before edges): {len([n for n, data in G.nodes(data=True) if data['type'] == 'subreddit'])}")

    # Add edges based on author activity in subreddits
    for index, row in df_for_graph.iterrows():
        author = row['author']
        subreddit = row['subreddit']
        if author in G and subreddit in G: # Ensure nodes exist
            G.add_edge(author, subreddit, type='posted_in')

    # Add edges based on direct mentions (simple example, need to adjust based on actual data)
    # This part needs content analysis. For demonstration, we'll assume content contains mentions.
    # In a real scenario, you'd parse `row['content']` for mentions.
    for index, row in df_for_graph.iterrows():
        author = row['author']
        content = row['content']
        # Simple regex to find potential mentions starting with u/ or @ (adjust as needed for data format)
        mentions = re.findall(r'(?:u/|@)([a-zA-Z0-9_-]+)', content)
        for mentioned_user in mentions:
            if mentioned_user in all_authors and mentioned_user != author: # Only add if mentioned user is also an author in our data
                if author in G and mentioned_user in G:
                    G.add_edge(author, mentioned_user, type='mentions')
    
    st.write(f"DEBUG: Number of nodes in graph G after adding edges: {G.number_of_nodes()}")
    st.write(f"DEBUG: Number of edges in graph G: {G.number_of_edges()}")
    st.write(f"DEBUG: Number of author nodes in G (after edges): {len([n for n, data in G.nodes(data=True) if data['type'] == 'author'])}")


    # --- Handle large graphs for better visualization performance ---
    # This block is TEMPORARILY COMMENTED OUT for debugging purposes.
    # It will be re-enabled or modified later.

    # if G.number_of_nodes() > 500: # Adjust this limit as needed
    #     st.info(f"The network graph has {G.number_of_nodes()} nodes. Displaying the largest connected component (or a sample) for better performance.")
    #     components = list(nx.connected_components(G))
    #     if components:
    #         largest_component_nodes = max(components, key=len)
    #         G = G.subgraph(largest_component_nodes).copy()
    #         if G.number_of_nodes() > n_nodes_to_display: # If largest component is still too big, sample
    #             st.info(f"Largest component ({G.number_of_nodes()} nodes) is still too large. Randomly sampling {n_nodes_to_display} nodes.")
    #             sampled_nodes = random.sample(list(G.nodes()), n_nodes_to_display)
    #             G = G.subgraph(sampled_nodes).copy()
    #     else:
    #         st.warning("No connected components found in the graph. Graph might be too sparse.")
    #         return None


    # Calculate degrees for sizing nodes
    node_degrees = dict(G.degree())

    # Scale author node sizes based on degree
    author_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'author']
    if author_nodes:
        author_degrees = [node_degrees[n] for n in author_nodes]
        # Avoid division by zero if all degrees are 0 or list is empty
        max_author_degree = max(author_degrees) if author_degrees else 1
        min_author_degree = min(author_degrees) if author_degrees else 0

        # Scale size between 10 and 30 for authors
        # Using a fixed min/max range for scaling to prevent tiny sizes with small degrees
        scaled_sizes = []
        for degree in author_degrees:
            if max_author_degree == min_author_degree: # Handle case with uniform degrees
                scaled_sizes.append(20) # Default size if no variation
            else:
                scaled_size = 10 + (degree - min_author_degree) * (20 / (max_author_degree - min_author_degree))
                scaled_sizes.append(scaled_size)
        
        author_node_sizes_scaled = {n: s for n, s in zip(author_nodes, scaled_sizes)}
    else:
        author_node_sizes_scaled = {}

    # Set subreddit node size (constant or based on a different metric)
    subreddit_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'subreddit']
    subreddit_node_sizes = {n: 8 for n in subreddit_nodes} # Fixed size for subreddits

    # Combine all node sizes
    node_sizes = {**author_node_sizes_scaled, **subreddit_node_sizes}


    # Get positions for all nodes
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42) # Adjust k and iterations for layout

    # Prepare node data for Plotly
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
        node_size_plot.append(node_sizes.get(node, 10)) # Default size if not explicitly set
        node_type.append(G.nodes[node]['type'])


    # Debugging lengths and samples for nodes
    st.write(f"DEBUG: Length of author_node_x: {len([x for i, x in enumerate(node_x) if node_type[i] == 'author'])}")
    st.write(f"DEBUG: Length of author_node_degrees: {len(author_degrees) if 'author_degrees' in locals() else 0}")
    if 'author_degrees' in locals() and author_degrees:
        st.write(f"DEBUG: Sample author degrees (first 10): {author_degrees[:10]}")
    st.write(f"DEBUG: Length of author_node_sizes_scaled: {len(author_node_sizes_scaled)}")
    if author_node_sizes_scaled:
        st.write(f"DEBUG: Sample author sizes (first 10): {list(author_node_sizes_scaled.values())[:10]}")
    if len(node_x) > 0:
        st.write(f"DEBUG: Sample author_node_x (first 10): {[x for i, x in enumerate(node_x) if node_type[i] == 'author'][:10]}")
        st.write(f"DEBUG: Sample author_node_y (first 10): {[y for i, y in enumerate(node_y) if node_type[i] == 'author'][:10]}")
    
    st.write(f"DEBUG: Length of subreddit_node_x: {len([x for i, x in enumerate(node_x) if node_type[i] == 'subreddit'])}")
    st.write(f"DEBUG: Length of subreddit_node_sizes: {len(subreddit_node_sizes)}")


    # Prepare edge data for Plotly
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
            showscale=False, # We're setting size manually
            colorscale='YlGnBu',
            reversescale=True,
            color=node_color,
            size=node_size_plot, # Use calculated sizes
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict( # Corrected: 'title' is a dictionary
                            text='Author Interaction Network Graph',
                            font=dict(size=16) # Corrected: font size within 'font' dict
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

# Function to convert plotly figure to base64 for download
def fig_to_base64(fig):
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png")
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
    return img_base64

# Main Streamlit app
def main():
    st.title("📊 Social Media Analysis Dashboard")

    st.sidebar.header("Data Configuration")
    
    # Load data
    df = load_data()

    # --- Conditional Sample Data Generation (REMAINING THE SAME) ---
    if df.empty or not df['created_at'].notna().any() or not df['content'].notna().any():
        st.warning("No valid data loaded or critical columns are missing/empty. Generating sample data for demonstration.")
        df = generate_sample_data(num_posts=1000)
        st.success("Sample data generated successfully!")
        st.write("Sample Data Head:")
        st.dataframe(df.head())
    # --- END Conditional Sample Data Generation ---


    # Ensure 'subreddit' column exists for filtering and graph, create if missing
    if 'subreddit' not in df.columns:
        st.warning("'subreddit' column not found in data. Some features may be limited.")
        df['subreddit'] = 'Unknown' # Default value if subreddit is missing


    # Sidebar filters
    st.sidebar.header("Filter Data")

    # Date range filter
    min_date = df['created_at'].min().date() if not df['created_at'].empty and pd.notna(df['created_at'].min()) else datetime.now().date() - timedelta(days=365)
    max_date = df['created_at'].max().date() if not df['created_at'].empty and pd.notna(df['created_at'].max()) else datetime.now().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['created_at'].dt.date >= start_date) & (df['created_at'].dt.date <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # Keyword search filter
    search_query = st.sidebar.text_input("Search keywords (comma-separated):")
    if search_query:
        keywords = [k.strip().lower() for k in search_query.split(',')]
        filtered_df = filtered_df[filtered_df['content'].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )].copy()

    # Sentiment filter
    sentiment_options = ['All', 'Positive', 'Neutral', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment:", sentiment_options)

    # Calculate sentiment only once for the filtered_df
    if not filtered_df.empty:
        if 'sentiment_score' not in filtered_df.columns:
            # Ensure 'content' column is string type before applying sentiment analysis
            filtered_df['content'] = filtered_df['content'].astype(str)
            filtered_df['sentiment_score'] = filtered_df['content'].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # --- DEBUG: Sentiment Analysis ---
        st.write(f"DEBUG (sentiment): filtered_df has {filtered_df.shape[0]} rows before sentiment analysis filter.")
        
        if selected_sentiment == 'Positive':
            filtered_df = filtered_df[filtered_df['sentiment_score'] >= 0.05].copy()
        elif selected_sentiment == 'Neutral':
            filtered_df = filtered_df[(filtered_df['sentiment_score'] > -0.05) & (filtered_df['sentiment_score'] < 0.05)].copy()
        elif selected_sentiment == 'Negative':
            filtered_df = filtered_df[filtered_df['sentiment_score'] <= -0.05].copy()
    else:
        st.info("No data to apply sentiment filter.")

    # Subreddit filter
    all_subreddits_available = df['subreddit'].unique()
    selected_subreddits = st.sidebar.multiselect("Filter by Subreddit:", all_subreddits_available, default=all_subreddits_available)
    filtered_df = filtered_df[filtered_df['subreddit'].isin(selected_subreddits)].copy()


    # Display metrics
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

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Sentiment Analysis", "Top Entities", "Word Cloud",
        "Author Network Graph", "Topic Modeling", "AI Insights"
    ])

    with tab1:
        st.header("Sentiment Analysis")
        if not filtered_df.empty:
            sentiment_counts = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')).value_counts()
            
            # --- DEBUG: Sentiment Counts ---
            st.write(f"DEBUG (sentiment): Sentiment counts: {sentiment_counts.to_dict()}") # Show sentiment counts
            
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title='Overall Sentiment Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sentiment Over Time")
            # Resample by day and calculate mean sentiment score
            if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
                daily_sentiment = filtered_df.set_index('created_at')['sentiment_score'].resample('D').mean().reset_index()
                fig_time = px.line(
                    daily_sentiment,
                    x='created_at',
                    y='sentiment_score',
                    title='Average Sentiment Over Time',
                    labels={'created_at': 'Date', 'sentiment_score': 'Average Sentiment Score'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("Date column not available for sentiment over time analysis.")
        else:
            st.info("No data to perform sentiment analysis.")

    with tab2:
        st.header("Top Entities/Authors/Subreddits")
        if not filtered_df.empty:
            st.subheader("Top 10 Authors")
            # Filter out 'Unknown User' from top authors list for display
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
            st.info("No data available for top entities analysis.")

    with tab3:
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
                st.warning("No meaningful words to generate word cloud after preprocessing.")
        else:
            st.warning("No content data available for word cloud generation after filtering.")

    with tab4:
        st.header("Author Interaction Network Graph")
        if Network is None:
            st.error("Pyvis library not found. Please install it (`pip install pyvis`) to enable the network graph feature.")
        else:
            if 'author' in filtered_df.columns and not filtered_df['author'].empty and 'subreddit' in filtered_df.columns and not filtered_df['subreddit'].empty:
                # Use only relevant columns for graph to avoid unnecessary data transfer
                df_for_graph = filtered_df[['author', 'subreddit', 'content']].copy()
                fig = create_plotly_network_graph(df_for_graph)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Option to download graph
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
                st.warning("Author and/or Subreddit data missing or empty for graph generation after filtering.")

    with tab5:
        st.header("Topic Modeling")
        if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
            processed_texts = preprocess_text(filtered_df['content'])
            topics = extract_topics(processed_texts, n_topics=5)
            st.subheader("Extracted Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i + 1}: {topic}")
        else:
            st.warning("No content available for topic modeling after filtering.")

    with tab6: # Renamed tab5 to tab6 due to an extra tab being added in code
        st.header("AI-Generated Insights")
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:",
                                                type='password')
        if openai_api_key:
            if 'content' in filtered_df.columns and not filtered_df['content'].empty:
                with st.spinner("Generating AI insights..."):\
                    insights = generate_mock_insights(filtered_df)
                for insight in insights:
                        st.markdown(f"- {insight}")
            else:
                st.warning("No content data available for AI analysis after filtering.")
        else:
            st.info("Enter an OpenAI API key in the sidebar to enable AI-generated insights.")

    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        st.dataframe(filtered_df.head(100))

    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()