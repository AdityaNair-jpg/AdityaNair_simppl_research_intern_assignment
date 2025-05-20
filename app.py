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
# Removed global nltk.download('vader_lexicon') as it's handled by @st.cache_resource
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
# We call this once, and it will return the cached analyzer on subsequent runs
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
    # print("\n--- Initial DataFrame Info ---") # Commented out for cleaner Streamlit output
    # df.info()
    # print("\n--- Initial DataFrame Head ---") # Commented out for cleaner Streamlit output
    # print(df.head())

    # Convert timestamp (prioritize specific columns)
    timestamp_cols = ['created_utc', 'created', 'timestamp', 'created_at',
                      'date', 'datetime', 'post_timestamp']
    df['created_at'] = pd.NaT  # Initialize with NaT (Not a Time)

    for col in timestamp_cols:
        if col in df.columns:
            # print(f"\nAttempting to convert column '{col}' to datetime...") # Commented out for cleaner Streamlit output
            try:
                # Try parsing as Unix timestamp first
                df['created_at'] = pd.to_datetime(df[col], unit='s', errors='coerce')
                if df['created_at'].notna().any():
                    # print(f"Successfully converted '{col}' from Unix timestamp.") # Commented out for cleaner Streamlit output
                    break  # Stop if successful
                else:
                    # print(f"Column '{col}' conversion from Unix timestamp resulted in all NaT. Trying default parsing.") # Commented out for cleaner Streamlit output
                    # If Unix conversion fails, try default parsing
                    df['created_at'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if df['created_at'].notna().any():
                        # print(f"Successfully converted '{col}' using default parsing.") # Commented out for cleaner Streamlit output
                        break
                    else:
                        # print(f"Column '{col}' conversion from default parsing also resulted in all NaT.") # Commented out for cleaner Streamlit output
                        pass # Continue to next column if this one also fails
            except ValueError as e:
                # print(f"Error converting '{col}' to datetime: {e}") # Commented out for cleaner Streamlit output
                pass # Continue to next column on error

    if 'created_at' not in df.columns or not df['created_at'].notna().any():
        print("Warning: No valid timestamp column found. Creating empty 'created_at' column.")
        df['created_at'] = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))  # Create empty datetime series explicitly

    # Debug: Print DataFrame info after timestamp conversion
    # print("\n--- DataFrame Info After Timestamp Conversion ---") # Commented out for cleaner Streamlit output
    # df.info()
    # print("\n--- DataFrame Head After Timestamp Conversion ---") # Commented out for cleaner Streamlit output
    # print(df.head())

    # Extract text content (prioritize 'selftext', 'title', 'content')
    text_cols = ['selftext', 'title', 'content', 'text', 'body', 'message', 'subreddit']
    df['content'] = None  # Initialize 'content' column
    for col in text_cols:
        if col in df.columns:
            df['content'] = df[col].astype(str).fillna('')  # Ensure string type and fill NaNs
            break
    if 'content' not in df.columns or df['content'].isnull().all():
        print("Warning: No valid text column found.")
        df['content'] = '' # Ensure 'content' column exists even if empty

    # Extract user information
    user_cols = ['user_id', 'author', 'username', 'user.screen_name']
    df['user_id'] = None  # Initialize 'user_id'
    for col in user_cols:
        if col in df.columns:
            df['user_id'] = df[col].astype(str).fillna('Unknown User')
            break
    if 'user_id' not in df.columns or df['user_id'].isnull().all():
        print("Warning: No valid user ID column found.")
        df['user_id'] = 'Unknown User' # Ensure 'user_id' column exists

    return df


# Function to process the DataFrame (this is NOT cached)
def process_data(df, search_term="", date_range=None):
    filtered_df = df.copy()  # Work on a copy to avoid modifying the original

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

    # Sentiment Analysis - applied directly in main() after filtering for efficiency
    # if 'content' in filtered_df.columns:
    #     filtered_df['sentiment'] = filtered_df['content'].fillna("").apply(analyze_sentiment)
    # else:
    #     filtered_df['sentiment'] = 'neutral'  # Or appropriate default

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
# This function now uses the globally cached 'sia' object
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
    except Exception as e:
        st.error(f"Error during topic extraction: {e}")
        return ["Error extracting topics."]


# Function to create network visualization using Pyvis
def create_network_graph(df, selected_date, media_type):
    if Network is None:
        st.warning("Pyvis library not found. Network graph cannot be displayed.")
        return None

    if df.empty:
        return None
    try:
        # Filter data by date
        df_filtered = df[(df['created_at'].dt.date == selected_date)]

        # Further filter by media type (assuming a 'media_type' column exists)
        if 'media_type' in df_filtered.columns and media_type != 'All':
            df_filtered = df_filtered[df_filtered['media_type'] == media_type]

        # Create a graph
        G = nx.Graph()

        # Add nodes and edges, handling missing 'user_id' and 'content' for mentions
        for _, row in df_filtered.iterrows():
            author = str(row.get('user_id', 'Unknown User'))
            G.add_node(author)
            content = str(row.get('content', ''))

            # Look for mentions (users preceded by @)
            mentions = re.findall(r'@(\w+)', content)
            for mention in mentions:
                # Add mentioned user as a node if not already present
                if mention not in G.nodes:
                    G.add_node(mention)
                # Add edge from author to mentioned user
                G.add_edge(author, mention)

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        if not G.nodes:
            st.warning("No connections to display for the selected date and media type.")
            return None

        # Create a PyVis network
        net = Network(height="500px",
                      width="100%",
                      directed=False, # Pyvis Network with 'directed=True' means arrows are drawn, False means no arrows
                      bgcolor="#222222",
                      font_color="white",
                      notebook=True) # Important for displaying in Streamlit

        net.from_nx(G)  # Use the networkx graph

        # Add degree as a node attribute (before generating the HTML)
        degree_centrality = nx.degree_centrality(G)
        for node in G.nodes():
            net.get_node(node)['size'] = degree_centrality[node] * 30 + 10 # Scale size for better visualization
            net.get_node(node)['title'] = f"Degree: {G.degree[node]}"
            net.get_node(node)['color'] = "#ADD8E6" # Light blue for nodes

        # Save the graph to an HTML file and return its content
        path = 'pyvis_graph.html'
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content

    except Exception as e:
        st.error(f"Error creating network graph: {e}")
        return None


# AI-generated insights mock function (since we don't have actual OpenAI
# integration)
def generate_mock_insights(df):
    # Count posts per day
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

    # Count most common words
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
        # st.write("Columns in your DataFrame:", df.columns)  # Debugging line - removed for cleaner UI
        # if not df.empty:
            # st.write("First 5 rows of DataFrame:", df.head())  # Added this line to see the data - removed for cleaner UI
        # else:
            # st.write("DataFrame is empty")


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

        # Generate user IDs
        user_ids = [f"user_{i}" for i in np.random.randint(1, 100, sample_size)]

        # Create DataFrame
        df = pd.DataFrame({
            'created_at': dates,
            'content': contents,
            'user_id': user_ids,
            'media_type': np.random.choice(['twitter', 'reddit', 'forum'], sample_size) # Add dummy media_type
        })

        st.success("Sample data generated successfully!")
        st.write("Sample Data Head:")
        st.dataframe(df.head())


    # Display data info
    st.sidebar.markdown(f"**Total Posts**: {len(df)}")

    # Create date filter if date column exists
    if 'created_at' in df.columns and not df['created_at'].empty and df['created_at'].min() is not pd.NaT:
        min_date = df['created_at'].min().date()
        max_date = df['created_at'].max().date()

        # Handle cases where min_date == max_date
        if min_date == max_date:
            st.sidebar.write(f"Data available for a single day: {min_date.strftime('%Y-%m-%d')}")
            selected_date_range = st.sidebar.date_input(
                "Select Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            # Ensure selected_date_range is a tuple for consistency
            if isinstance(selected_date_range, datetime):
                selected_date_range = (selected_date_range.date(), selected_date_range.date())
            else:
                selected_date_range = (selected_date_range, selected_date_range)
        else:
            selected_date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = df[(df['created_at'].dt.date >= start_date) &
                             (df['created_at'].dt.date <= end_date)].copy() # Use .copy() to avoid SettingWithCopyWarning
        else:
            filtered_df = df.copy() # Use .copy()
    else:
        filtered_df = df.copy() # Use .copy()
        st.sidebar.warning("No valid date column found for filtering.")

    # Search functionality
    search_term = st.sidebar.text_input("Search in content", "")
    if search_term:
        if 'content' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['content'].str.contains(
                search_term, case=False, na=False)].copy() # Use .copy()
            st.sidebar.markdown(
                f"Found {len(filtered_df)} posts containing '{search_term}'")
        else:
            st.sidebar.warning("Content column not found for search.")

    # Media Type Filter
    if 'media_type' in df.columns:
        media_types = ['All'] + list(df['media_type'].unique())
        selected_media_type = st.sidebar.selectbox("Filter by Media Type", media_types)
        if selected_media_type != 'All':
            filtered_df = filtered_df[filtered_df['media_type'] == selected_media_type].copy() # Use .copy()
    else:
        st.sidebar.info("No 'media_type' column found in your data.")


    # Add sentiment column if not already present, apply after all filters
    with st.spinner("Analyzing sentiment..."):
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            filtered_df['sentiment'] = filtered_df['content'].apply(analyze_sentiment)
        else:
            filtered_df['sentiment'] = 'neutral'
            st.warning("No content to analyze for sentiment after filtering.")

    # Debug: Inspect the DataFrame (optional, can be commented out for production)
    # st.subheader("Debug: DataFrame with Sentiment")
    # st.dataframe(filtered_df)

    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Time Series Analysis", "Content Analysis",
         "Network Analysis", "Topic Modeling", "AI Insights"])

    # Tab 1: Time Series Analysis
    with tab1:
        st.header("Post Activity Over Time")

        if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
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
            col2.metric("Start Date", daily_counts['date'].min().strftime('%Y-%m-%d'))
            col3.metric("End Date", daily_counts['date'].max().strftime('%Y-%m-%d'))

            # Optional: Display daily counts DataFrame
            if st.checkbox("Show Daily Post Counts"):
                st.dataframe(daily_counts)
        else:
            st.warning("No valid date information available for time series analysis after filtering.")

    # Tab 2: Content Analysis
    with tab2:
        st.header("Content Analysis")
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            # Word Cloud
            st.subheader("Word Cloud")
            text_data = filtered_df['content']
            wordcloud_img = generate_wordcloud(text_data)
            if wordcloud_img:
                plt.imshow(wordcloud_img, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt.gcf())
            else:
                st.info("Not enough text data to generate a meaningful word cloud.")


            # Most Common Words
            st.subheader("Most Common Words")
            all_text = " ".join(filtered_df['content'].dropna().astype(str))
            if all_text.strip(): # Check if there's actual text content
                words = word_tokenize(all_text.lower())
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word.isalpha() and word not in
                         stop_words and len(word) > 2]  # Exclude short words
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

            ## Sentiment Pie Chart
            st.subheader("Sentiment Distribution")

            # ✅ Clean sentiment data
            filtered_df['sentiment'] = filtered_df['sentiment'].str.strip().str.lower()

            # ✅ Count using groupby (this is correct!)
            sentiment_counts = (
                filtered_df.groupby('sentiment')
                    .size()
                    .reset_index(name='counts')
            )
            print(sentiment_counts)
            # ✅ Plot pie chart
            if not sentiment_counts.empty:
                fig_sentiment = go.Figure(data=[go.Pie(
                    labels=sentiment_counts['sentiment'],
                    values=sentiment_counts['counts'],
                    hole=.3  # Optional: makes it a donut chart
                )])
                fig_sentiment.update_layout(title_text='Post Sentiment Breakdown')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.warning("No sentiment data available to plot after filtering.")


        else:
            st.warning("No content available for analysis after filtering.")

    # Tab 3: Network Analysis
    with tab3:
        st.header("User Interaction Network")
        if 'user_id' in filtered_df.columns and not filtered_df['user_id'].empty and Network is not None:
            # For network graph, use the unfiltered df and pass date and media type filters
            # as it needs to build a network over interactions, not just current filtered posts
            # If your network graph is strictly based on posts in the current filtered_df,
            # then pass filtered_df instead of df.
            # Assuming network needs interactions across the selected date range:
            
            # Use the currently selected date from the sidebar for the network graph
            selected_date_for_network = selected_date_range[0] # Assuming start_date of the range
            if len(selected_date_range) == 2:
                selected_date_for_network = selected_date_range[0] # Use start of range for single-day graph
            else:
                # If single date input was used, selected_date_range will be a datetime.date object
                selected_date_for_network = selected_date_range

            # Ensure selected_media_type is defined
            if 'media_type' in df.columns:
                media_type_for_network = selected_media_type
            else:
                media_type_for_network = 'All' # Default if no media_type column

            network_graph_html = create_network_graph(df, selected_date_for_network, media_type_for_network) # Pass original df and filters
            
            if network_graph_html:
                st.components.v1.html(network_graph_html, height=600, scrolling=True)
            else:
                st.info("Could not generate network graph. Ensure there are interactions for the selected date and media type.")
        elif Network is None:
            st.warning("Pyvis library is not installed. Please install it to enable network analysis.")
        else:
            st.warning("No user information available for network analysis after filtering.")


    # Tab 4: Topic Modeling
    with tab4:
        st.header("Topic Modeling")
        if 'content' in filtered_df.columns and not filtered_df['content'].empty:
            processed_texts = preprocess_text(filtered_df['content'])
            topics = extract_topics(processed_texts, n_topics=5)  # Example: 5 topics
            st.subheader("Extracted Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i + 1}: {topic}")
        else:
            st.warning("No content available for topic modeling after filtering.")

    # Tab 5: AI Insights
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

    # Display raw data if requested
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        st.dataframe(filtered_df.head(100))

    # Footer
    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()

