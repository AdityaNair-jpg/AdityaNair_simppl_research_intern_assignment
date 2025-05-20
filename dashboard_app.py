import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import nltk
nltk.download('vader_lexicon') # Add this line
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Add this line
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
from sklearn.manifold import TSNE
from bertopic import BERTopic
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
import openai
from pyvis.network import Network
from IPython.display import HTML, display
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load data
@st.cache_data
def load_data():
    data = []
    with open('data.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                # Safely extract data, handling missing 'data' key
                data.append(json_obj.get('data', json_obj))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")  # Log invalid lines
                continue
    df = pd.DataFrame(data)

    # Debug: Print initial column data types
    print("\n--- Initial DataFrame Info ---")
    df.info()
    print("\n--- Initial DataFrame Head ---")
    print(df.head())

    # Convert timestamp (prioritize specific columns)
    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at', 'date', 'datetime', 'post_timestamp']
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
                    print(f"Column '{col}' conversion from Unix timestamp resulted in all NaT. Trying default parsing.")
                    # If Unix conversion fails, try default parsing
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        print(f"Successfully converted '{col}' using default parsing.")
                        break
                    else:
                        print(f"Column '{col}' conversion from default parsing also resulted in all NaT.")
            except ValueError as e:
                print(f"Error converting '{col}' to datetime: {e}")

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        print("Warning: No valid timestamp column found. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series())  # Create empty datetime series

    # Debug: Print DataFrame info after timestamp conversion
    print("\n--- DataFrame Info After Timestamp Conversion ---")
    df.info()
    print("\n--- DataFrame Head After Timestamp Conversion ---")
    print(df.head())

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message', 'subreddit']
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
                          max_words=100, contour_width=3, contour_color='steelblue').generate(text)
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

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'


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

# Function to get summary from OpenAI (if API key is provided)
def get_ai_summary(text, openai_api_key=None):
    if not openai_api_key:
        return "API key required for AI summaries"

    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes social media trends."},
                {"role": "user", "content": f"Summarize the following social media content trends in 2-3 sentences: {text}"}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to create network visualization
def create_network_graph(df, column, filter_value=None):
    # Create a subset based on filter if provided
    if filter_value:
        subset = df[df[column].str.contains(filter_value, na=False)]
    else:
        subset = df

    # Create graph
    G = nx.Graph()

    # Add nodes for users
    for user in subset['user_id'].unique():
        G.add_node(user, type='user')

    # Add edges based on interactions
    if 'in_reply_to_user_id' in subset.columns:
        for _, row in subset.iterrows():
            if pd.notna(row['in_reply_to_user_id']):
                G.add_edge(row['user_id'], row['in_reply_to_user_id'], type='reply')

    # Convert to pyvis network for visualization
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")

    # Add nodes and edges
    for node in G.nodes():
        net.add_node(node, label=str(node), title=str(node))

    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # Save and return HTML
    temp_path = "temp_network.html"
    net.save_graph(temp_path)

    with open(temp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return html_content

# Main Streamlit app
def main():
    st.set_page_config(page_title="Social Media Analysis Dashboard", layout="wide")

    st.title("Social Media Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes social media data to uncover patterns in information spread,
    focusing on potentially unreliable sources and trending topics.
    """)

    # Sidebar for filters and options
    st.sidebar.title("Filters & Options")

    # Load data
    try:
        with st.spinner("Loading data..."):
            df = load_data()
            st.write("Columns in your DataFrame:", df.columns)  # Debugging line
            if not df.empty:
                st.write("First 5 rows of DataFrame:", df.head()) #Added this line to see the data
            else:
                st.write("DataFrame is empty")

        # Check if data loaded correctly
        if df.empty:
            st.error("No data loaded. Please check if the data.jsonl file is in the correct location.")
            return

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
                filtered_df = filtered_df[filtered_df['content'].str.contains(search_term, case=False, na=False)]
                st.sidebar.markdown(f"Found {len(filtered_df)} posts containing '{search_term}'")
            else:
                st.sidebar.warning("Content column not found for search.")
        # Add sentiment column if not already present
# (Existing search and date filtering logic...)

        if 'content' in filtered_df.columns:
            with st.spinner("Analyzing sentiment..."):
                filtered_df['sentiment'] = filtered_df['content'].fillna("").apply(analyze_sentiment)
        else:
            filtered_df['sentiment'] = 'neutral' # Fallback if no content column


        # Add OpenAI API key input for AI summaries
        openai_api_key = st.sidebar.text_input("OpenAI API Key (for AI Summaries)", type="password")

        # Main dashboard content
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time Series Analysis", "Content Analysis",
                                                "Network Analysis", "Topic Modeling", "AI Insights"])

        # Tab 1: Time Series Analysis
        with tab1:
            st.header("Post Activity Over Time")

            if 'created_at' in filtered_df.columns:
                # Group by date
                time_df = filtered_df.set_index('created_at')
                # Resample by day
                daily_counts = time_df.resample('D').size().reset_index(name='count')

                # Create time series plot
                fig = px.line(daily_counts, x='created_at', y='count',
                            title='Post Volume Over Time',
                            labels={'created_at': 'Date', 'count': 'Number of Posts'})
                fig.update_layout(xaxis_title='Date', yaxis_title='Number of Posts')
                st.plotly_chart(fig, use_container_width=True)

                # Add filter for time aggregation
                time_agg = st.selectbox('Time Aggregation', ['Day', 'Week', 'Month'], index=0)

                if time_agg == 'Day':
                    resample_rule = 'D'
                    title_suffix = 'Daily'
                elif time_agg == 'Week':
                    resample_rule = 'W'
                    title_suffix = 'Weekly'
                else:
                    resample_rule = 'M'
                    title_suffix = 'Monthly'

                agg_counts = time_df.resample(resample_rule).size().reset_index(name='count')

                fig2 = px.bar(agg_counts, x='created_at', y='count',
                            title=f'{title_suffix} Post Volume',
                            labels={'created_at': 'Date', 'count': 'Number of Posts'})
                st.plotly_chart(fig2, use_container_width=True)

                # Display peaks in activity
                st.subheader("Peak Activity Periods")

                # Find peaks (days with top 10% activity)
                threshold = daily_counts['count'].quantile(0.9)
                peak_days = daily_counts[daily_counts['count'] >= threshold]

                if not peak_days.empty:
                    peak_fig = px.bar(peak_days, x='created_at', y='count',
                                    title='Days with Highest Activity',
                                    labels={'created_at': 'Date', 'count': 'Number of Posts'})
                    st.plotly_chart(peak_fig, use_container_width=True)

                    # Table of peak days
                    st.write("Peak Activity Days:")
                    peak_days_display = peak_days.sort_values('count', ascending=False)
                    peak_days_display['created_at'] = peak_days_display['created_at'].dt.strftime('%Y-%m-%d')
                    st.dataframe(peak_days_display.head(10))
                else:
                    st.write("No peak activity days identified.")
            else:
                st.error("No timestamp data available for time series analysis.")

        # Tab 2: Content Analysis
        with tab2:
            st.header("Content Analysis")

            if 'content' in filtered_df.columns:
                # Sample of filtered content
                st.subheader("Sample Posts")
                sample_size = min(5, len(filtered_df))
                for i, row in filtered_df.sample(sample_size).iterrows():
                    with st.expander(f"Post {i}"):
                        st.write(row['content'])

                # Word frequency analysis
                st.subheader("Word Frequency Analysis")

                # Process text and get word counts
                all_text = " ".join(filtered_df['content'].dropna().astype(str))
                stop_words = set(stopwords.words('english'))

                # Tokenize and clean
                words = word_tokenize(all_text.lower())
                words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]

                # Get word frequencies
                word_freq = Counter(words).most_common(20)
                word_df = pd.DataFrame(word_freq, xcolumns=['Word', 'Frequency'])

                # Bar chart of word frequencies
                fig_words = px.bar(word_df, x='Word', y='Frequency', title='Most Common Words')
                st.plotly_chart(fig_words, use_container_width=True)

                # Word cloud
                st.subheader("Word Cloud")
                wordcloud = generate_wordcloud(filtered_df['content'])

                # Display the wordcloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

                # Sentiment analysis if available from the dataset
                if 'sentiment' in filtered_df.columns:
                    st.subheader("Sentiment Analysis")
                    # Get value counts directly, which are already numeric
                    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['sentiment', 'counts'] # Rename columns for clarity

                    # Debug: Verify the DataFrame before plotting (optional, but good for confirmation)
                    st.write("Debug: Sentiment counts DataFrame before plotting", sentiment_counts)
                    st.write("Data types in sentiment_counts:", sentiment_counts.dtypes)

                    if not sentiment_counts.empty: # Ensure there's data to plot
                        # --- NEW CODE: Using plotly.graph_objects.Figure and go.Pie ---
                        fig_sentiment = go.Figure(data=[go.Pie(
                            labels=sentiment_counts['sentiment'],
                            values=sentiment_counts['counts'],
                            hole=.3 # Optional: makes it a donut chart
                        )])
                        fig_sentiment.update_layout(title_text='Post Sentiment Breakdown')
                        # --- END NEW CODE ---
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    else:
                        st.warning("No sentiment data available to plot after analysis.")
            else:
                st.error("No content data available for analysis.")

        # Tab 3: Network Analysis
        with tab3:
            st.header("Network Analysis")

            # User network visualization
            if 'user_id' in filtered_df.columns:
                st.subheader("User Interaction Network")

                # Filter options for network analysis
                network_filter = st.text_input("Filter network by keyword (optional)")

                # Network visualization
                with st.spinner("Generating network visualization..."):
                    network_html = create_network_graph(filtered_df, 'content', network_filter)
                    st.components.v1.html(network_html, height=600)

                # User activity analysis
                st.subheader("Top Active Users")
                if 'user_id' in filtered_df.columns:
                    user_counts = filtered_df['user_id'].value_counts().reset_index()
                    user_counts.columns = ['User', 'Post Count']

                    # Bar chart of top users
                    fig_users = px.bar(user_counts.head(10), x='User', y='Post Count',
                                        title='Most Active Users')
                    st.plotly_chart(fig_users, use_container_width=True)

                    # Display communities if available
                    if 'community' in filtered_df.columns:
                        st.subheader("Community Distribution")
                        community_counts = filtered_df['community'].value_counts()
                        fig_community = px.pie(names=community_counts.index, values=community_counts.values,
                                            title='Community Distribution')
                        st.plotly_chart(fig_community, use_container_width=True)
            else:
                st.error("No user data available for network analysis.")

        # Tab 4: Topic Modeling
        with tab4:
            st.header("Topic Modeling")

            if 'content' in filtered_df.columns and not filtered_df.empty:
                # Process text for topic modeling
                with st.spinner("Processing text for topic modeling..."):
                    processed_texts = preprocess_text(filtered_df['content'])

                    # Filter out empty texts
                    processed_texts = [text for text in processed_texts if text.strip()]

                    if processed_texts:
                        # Extract topics
                        num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)

                        topics = extract_topics(processed_texts, n_topics=num_topics)

                        # Display topics
                        st.subheader("Discovered Topics")
                        for i, topic in enumerate(topics):
                            st.write(f"**Topic {i+1}:** {topic}")

                        # Simple visualization of topic distribution
                        st.subheader("Topic Distribution")
                        topic_names = [f"Topic {i+1}" for i in range(len(topics))]
                        topic_counts = [100 / len(topics) for _ in range(len(topics))]  # Placeholder equal distribution

                        fig_topics = px.pie(names=topic_names, values=topic_counts,
                                        title='Topic Distribution')
                        st.plotly_chart(fig_topics, use_container_width=True)

                        # Add topic modeling visualization (TSNE for topic clustering)
                        st.subheader("Topic Embedding Visualization")
                        st.write("This visualization shows how topics cluster together semantically.")

                        # Create simple placeholder visualization
                        # In a real implementation, this would use actual embeddings
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Generate random positions for demonstration
                        np.random.seed(42)
                        x = np.random.randn(len(topic_names))
                        y = np.random.randn(len(topic_names))

                        ax.scatter(x, y, s=100)
                        for i, txt in enumerate(topic_names):
                            ax.annotate(txt, (x[i], y[i]), fontsize=9, ha='center')

                        ax.set_xlabel('Dimension 1')
                        ax.set_ylabel('Dimension 2')
                        ax.set_title('Topic Clustering (t-SNE Visualization)')
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough content for topic modeling after text processing.")
            else:
                st.error("No content data available for topic modeling.")

        # Tab 5: AI Insights
        with tab5:
            st.header("AI-Generated Insights")

            if openai_api_key:
                if 'content' in filtered_df.columns and not filtered_df.empty:
                    # Prepare sample text for AI analysis
                    sample_text = " ".join(filtered_df['content'].sample(min(50, len(filtered_df))).dropna().astype(str))

                    with st.spinner("Generating AI insights..."):
                        summary = get_ai_summary(sample_text, openai_api_key)

                        st.subheader("Summary of Content Trends")
                        st.write(summary)

                        # Generate trend analysis
                        st.subheader("AI Trend Analysis")

                        if 'created_at' in filtered_df.columns:
                            # Group by date
                            time_df = filtered_df.set_index('created_at')
                            # Resample by day
                            daily_counts = time_df.resample('D').size().reset_index(name='count')

                            trend_text = f"Analyze the following daily post counts over time and identify key trends and patterns: {daily_counts.to_string()}"
                            trend_analysis = get_ai_summary(trend_text, openai_api_key)

                            st.write(trend_analysis)

                        # Generate topic insights
                        st.subheader("AI Topic Insights")

                        if 'content' in filtered_df.columns:
                            topics_prompt = f"Identify the main topics and themes in these social media posts: {sample_text}"
                            topic_insights = get_ai_summary(topics_prompt, openai_api_key)

                            st.write(topic_insights)
                else:
                    st.error("No content data available for AI analysis.")
            else:
                st.info("Enter an OpenAI API key in the sidebar to enable AI-generated insights.")

        # Display raw data if requested
        if st.sidebar.checkbox("Show Raw Data"):
            st.subheader("Raw Data Sample")
            st.dataframe(filtered_df.head(100))

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check if the data.jsonl file is properly formatted and contains the expected columns.")

if __name__ == "__main__":
    main()
