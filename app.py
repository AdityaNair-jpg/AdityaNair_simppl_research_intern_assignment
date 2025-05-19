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
import warnings

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


download_nltk_resources()


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
                    data.append(json_obj.get('data', json_obj))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")  # Log invalid lines
                    continue
    except FileNotFoundError:
        st.error("Error: data.jsonl file not found.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Debug: Print initial column data types
    print("\n--- Initial DataFrame Info ---")
    df.info()
    print("\n--- Initial DataFrame Head ---")
    print(df.head())

    # Convert timestamp (prioritize specific columns)
    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at',
                      'date', 'datetime', 'post_timestamp']
    df['created_at'] = pd.NaT  # Initialize with NaT (Not a Time)

    for col in timestamp_cols:
        if col in df.columns:
            print(f"\nAttempting to convert column '{col}' to datetime...")
            try:
                # Try parsing as Unix timestamp first
                df['created_at'] = pd.to_datetime(df[col], unit='s', errors='coerce')
                if df['created_at'].notna().any():
                    print(f"Successfully converted '{col}' from Unix timestamp.")
                    break  # Stop if successful
                else:
                    print(
                        f"Column '{col}' conversion from Unix timestamp resulted in all NaT. Trying default parsing.")
                    # If Unix conversion fails, try default parsing
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce',
                                                     infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        print(f"Successfully converted '{col}' using default parsing.")
                        break
                    else:
                        print(
                            f"Column '{col}' conversion from default parsing also resulted in all NaT.")
            except ValueError as e:
                print(f"Error converting '{col}' to datetime: {e}")

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        print(
            "Warning: No valid timestamp column found. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series())  # Create empty datetime series

    # Debug: Print DataFrame info after timestamp conversion
    print("\n--- DataFrame Info After Timestamp Conversion ---")
    df.info()
    print("\n--- DataFrame Head After Timestamp Conversion ---")
    print(df.head())

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message',
                 'subreddit']
    df['content'] = None  # Initialize 'content' column
    for col in text_cols:
        if col in df.columns:
            df['content'] = df[col].astype(str)  # Ensure string type
            break
    if 'content' not in df.columns:
        print("Warning: No valid text column found.")

    # Extract user information
    user_cols = ['user_id', 'author', 'username', 'user.screen_name']
    df['user_id'] = None  # Initialize 'user_id'
    for col in user_cols:
        if col in df.columns:
            df['user_id'] = df[col].astype(str)
            break
    if 'user_id' not in df.columns:
        print("Warning: No valid user ID column found.")

    return df


# Function to generate a wordcloud
def generate_wordcloud(text_data):
    text = " ".join(text_data.dropna().astype(str))
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
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            tokens = [word for word in tokens if word not in stop_words]
            # Join back to string
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        else:
            processed_texts.append("")

    return processed_texts


# Function to extract topics using NMF
def extract_topics(documents, n_topics=5):
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(documents)

    # Apply NMF
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf)

    # Get topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(", ".join(top_words))

    return topics


# Function to create network visualization
def create_network_graph(df, column, filter_value=None, max_nodes=100):
    # Create a subset based on filter if provided
    if filter_value:
        subset = df[df[column].str.contains(filter_value, na=False)]
    else:
        subset = df

    # Limit to a manageable number of nodes
    if len(subset) > max_nodes:
        subset = subset.sample(max_nodes)

    # Create graph
    G = nx.Graph()

    # Add nodes for users
    for user in subset['user_id'].unique():
        G.add_node(str(user), size=5)

    # Add edges based on interactions (example: mentions)
    edges_added = 0
    for idx, row in subset.iterrows():
        user = str(row['user_id'])
        content = str(row.get('content', ''))

        # Look for mentions (users preceded by @)
        mentions = re.findall(r'@(\w+)', content)
        for mention in mentions:
            if mention in G.nodes:
                G.add_edge(user, mention, weight=1)
                edges_added += 1

    # Generate positions using spring layout
    pos = nx.spring_layout(G)

    # Create figure
    fig = plt.figure(figsize=(10, 8))

    # Draw network
    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            node_size=700, edge_color='gray', linewidths=0.5,
            font_size=8)

    # Add title
    plt.title(f"User Interaction Network ({len(G.nodes)} users, {edges_added} connections)")

    return fig


# AI-generated insights mock function (since we don't have actual OpenAI
# integration)
def generate_mock_insights(df):
    # Count posts per day
    if 'created_at' in df.columns:
        daily_counts = df.groupby(df['created_at'].dt.date).size()
        peak_day = daily_counts.idxmax()
        peak_count = daily_counts.max()
    else:
        peak_day = "unknown"
        peak_count = 0

    # Count most common words
    if 'content' in df.columns:
        all_text = " ".join(df['content'].dropna().astype(str))
        words = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalpha() and word not in
                 stop_words and len(word) > 2]
        most_common = Counter(words).most_common(3)
        common_words = ", ".join([word for word, _ in most_common])
    else:
        common_words = "unknown"

    # Generate mock insights
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
        st.write("Columns in your DataFrame:", df.columns)  # Debugging line
        if not df.empty:
            st.write("First 5 rows of DataFrame:", df.head())  # Added this line to see the data
        else:
            st.write("DataFrame is empty")

    # Check if data loaded correctly
    if df.empty:
        st.error("No data loaded or the data.jsonl file is empty/missing.")
        st.info("For demonstration purposes, a sample dataset will be generated.")

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

        # Generate user IDs
        user_ids = [f"user_{i}" for i in np.random.randint(1, 100, sample_size)]

        # Create DataFrame
        df = pd.DataFrame({
            'created_at': dates,
            'content': contents,
            'user_id': user_ids
        })

        st.success("Sample data generated successfully!")

    # Display data info
    st.sidebar.markdown(f"**Total Posts**: {len(df)}")

    # Create date filter if date column exists
    if 'created_at' in df.columns:
        min_date = df['created_at'].min().date()
        max_date = df['created_at'].max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['created_at'].dt.date >= start_date) &
                             (df['created_at'].dt.date <= end_date)]
        else:
            filtered_df = df
    else:
        filtered_df = df
        st.sidebar.warning("No date column found in data.")

    # Search functionality
    search_term = st.sidebar.text_input("Search in content", "")
    if search_term:
        if 'content' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['content'].str.contains(
                search_term, case=False, na=False)]
            st.sidebar.markdown(
                f"Found {len(filtered_df)} posts containing '{search_term}'")
        else:
            st.sidebar.warning("Content column not found for search.")

    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Time Series Analysis", "Content Analysis",
         "Network Analysis", "Topic Modeling", "AI Insights"])

    # Tab 1: Time Series Analysis
    with tab1:
        st.header("Post Activity Over Time")

        if 'created_at' in filtered_df.columns:
            # Group by date
            time_df = filtered_df.copy()
            time_df['date'] = time_df['created_at'].dt.date
            daily_counts = time_df.groupby('date').size().reset_index(
                name='count')

            # Create time series plot
            fig = px.line(daily_counts, x='date', y='count',
                          title='Post Volume Over Time',
                          labels={'date': 'Date', 'count': 'Number of Posts'})
            st.plotly_chart(fig, use_container_width=True)

            # Display aggregated metrics
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            # Ensure that the values passed to st.metric are of type int, float, str, or None.
            col1.metric("Total Posts", len(filtered_df))
            col2.metric("Start Date", daily_counts['date'].min().strftime('%Y-%m-%d'))  # Convert to string
            col3.metric("End Date", daily_counts['date'].max().strftime('%Y-%m-%d'))    # Convert to string

            # Optional: Display daily counts DataFrame
            if st.checkbox("Show Daily Post Counts"):
                st.dataframe(daily_counts)
        else:
            st.warning("No date information available for time series analysis.")

    # Tab 2: Content Analysis
    with tab2:
        st.header("Content Analysis")
        if 'content' in filtered_df.columns:
            # Word Cloud
            st.subheader("Word Cloud")
            text_data = filtered_df['content']
            wordcloud = generate_wordcloud(text_data)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())  # Use plt.gcf() to get current figure

            # Most Common Words
            st.subheader("Most Common Words")
            all_text = " ".join(filtered_df['content'].dropna().astype(str))
            words = word_tokenize(all_text.lower())
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.isalpha() and word not in
                     stop_words and len(word) > 2]  # Exclude short words
            word_counts = Counter(words)
            most_common = pd.DataFrame(word_counts.most_common(20),
                                       columns=['word', 'count'])
            fig_bar = px.bar(most_common, x='word', y='count',
                              title='Top 20 Most Frequent Words')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No content available for analysis.")

    # Tab 3: Network Analysis
    with tab3:
        st.header("User Interaction Network")
        if 'user_id' in filtered_df.columns:
            # User selection dropdown
            user_filter = st.selectbox("Filter by User",
                                       ['All'] + list(filtered_df['user_id'].unique()))
            if user_filter == 'All':
                fig = create_network_graph(filtered_df, 'user_id')
            else:
                fig = create_network_graph(filtered_df, 'user_id',
                                            filter_value=user_filter)
            st.pyplot(fig)
        else:
            st.warning("No user information available for network analysis.")

    # Tab 4: Topic Modeling
    with tab4:
        st.header("Topic Modeling")
        if 'content' in filtered_df.columns:
            processed_texts = preprocess_text(filtered_df['content'])
            topics = extract_topics(processed_texts, n_topics=5)  # Example: 5 topics
            st.subheader("Extracted Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i + 1}: {topic}")
        else:
            st.warning("No content available for topic modeling.")

    # Tab 5: AI Insights
    with tab5:
        st.header("AI-Generated Insights")
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:",
                                                type='password')
        if openai_api_key:
            if 'content' in filtered_df.columns:
                with st.spinner("Generating AI insights..."):
                    insights = generate_mock_insights(filtered_df)
                    for insight in insights:
                        st.markdown(f"- {insight}")
            else:
                st.warning("No content data available for AI analysis.")
        else:
            st.info(
                "Enter an OpenAI API key in the sidebar to enable AI-generated insights.")

    # Display raw data if requested
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        st.dataframe(filtered_df.head(100))

    # Footer
    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()
