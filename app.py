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
    try:
        data = []
        with open('data.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    if "data" in json_obj:  # Check if "data" key exists
                        data.append(json_obj["data"])  # Append the inner data
                    else:
                        data.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if it exists
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        elif 'timestamp' in df.columns:
            df['created_at'] = pd.to_datetime(df['timestamp'])
        elif 'created_utc' in df.columns:  # Add this based on your column list
            df['created_at'] = pd.to_datetime(df['created_utc'])
        elif 'created' in df.columns: # Added this based on your column list
            df['created_at'] = pd.to_datetime(df['created'])
        
        # Extract text content
        text_columns = ['text', 'content', 'body', 'message']
        for col in text_columns:
            if col in df.columns:
                df['content'] = df[col]
                break
        
        # Extract user information
        user_cols = ['user_id', 'author', 'username', 'user.screen_name']
        for col in user_cols:
            if col in df.columns:
                df['user_id'] = df[col]
                break
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty DataFrame with expected columns to prevent errors
        return pd.DataFrame(columns=['created_at', 'content', 'user_id'])

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

# AI-generated insights mock function (since we don't have actual OpenAI integration)
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
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
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
                st.write("First 5 rows of DataFrame:", df.head()) #Added this line to see the data
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
            filtered_df = filtered_df[filtered_df['content'].str.contains(search_term, case=False, na=False)]
            st.sidebar.markdown(f"Found {len(filtered_df)} posts containing '{search_term}'")
        else:
            st.sidebar.warning("Content column not found for search.")
    
    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time Series Analysis", "Content Analysis", 
                                        "Network Analysis", "Topic Modeling", "AI Insights"])
    
    # Tab 1: Time Series Analysis
    with tab1:
        st.header("Post Activity Over Time")
        
        if 'created_at' in filtered_df.columns:
            # Group by date
            time_df = filtered_df.copy()
            time_df['date'] = time_df['created_at'].dt.date
            daily_counts = time_df.groupby('date').size().reset_index(name='count')
            
            # Create time series plot
            fig = px.line(daily_counts, x='date', y='count', 
                        title='Post Volume Over Time',
                        labels={'date': 'Date', 'count': 'Number of Posts'})
            fig.update_layout(xaxis_title='Date', yaxis_title='Number of Posts')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add filter for time aggregation
            time_agg = st.selectbox('Time Aggregation', ['Day', 'Week', 'Month'], index=0)
            
            # Aggregate by selected time period
            if time_agg == 'Day':
                # Already aggregated by day above
                agg_df = daily_counts
                title_suffix = 'Daily'
            elif time_agg == 'Week':
                time_df['week'] = time_df['created_at'].dt.isocalendar().week
                time_df['year'] = time_df['created_at'].dt.isocalendar().year
                agg_df = time_df.groupby(['year', 'week']).size().reset_index(name='count')
                agg_df['date'] = agg_df.apply(lambda x: f"{x['year']}-W{x['week']:02d}", axis=1)
                title_suffix = 'Weekly'
            else:
                time_df['month'] = time_df['created_at'].dt.to_period('M').dt.to_timestamp()
                agg_df = time_df.groupby('month').size().reset_index(name='count')
                agg_df.rename(columns={'month': 'date'}, inplace=True)
                title_suffix = 'Monthly'
            
            # Create bar chart
            fig2 = px.bar(agg_df, x='date', y='count',
                        title=f'{title_suffix} Post Volume',
                        labels={'date': 'Time Period', 'count': 'Number of Posts'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display peaks in activity
            st.subheader("Peak Activity Periods")
            
            # Find peaks (days with top 10% activity)
            threshold = daily_counts['count'].quantile(0.9)
            peak_days = daily_counts[daily_counts['count'] >= threshold]
            
            if not peak_days.empty:
                peak_fig = px.bar(peak_days, x='date', y='count',
                                title='Days with Highest Activity',
                                labels={'date': 'Date', 'count': 'Number of Posts'})
                st.plotly_chart(peak_fig, use_container_width=True)
                
                # Table of peak days
                st.write("Peak Activity Days:")
                peak_days_display = peak_days.sort_values('count', ascending=False)
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
            word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            # Bar chart of word frequencies
            fig_words = px.bar(word_df, x='Word', y='Frequency', title='Most Common Words')
            st.plotly_chart(fig_words, use_container_width=True)
            
            # Word cloud
            st.subheader("Word Cloud")
            try:
                wordcloud = generate_wordcloud(filtered_df['content'])
                
                # Display the wordcloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
            
            # Hashtag analysis
            st.subheader("Hashtag Analysis")
            
            # Extract hashtags
            hashtags = []
            for content in filtered_df['content'].dropna().astype(str):
                hashtags.extend(re.findall(r'#(\w+)', content.lower()))
            
            if hashtags:
                hashtag_counts = Counter(hashtags).most_common(15)
                hashtag_df = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Count'])
                
                # Bar chart of hashtags
                fig_hashtags = px.bar(hashtag_df, x='Hashtag', y='Count', 
                                    title='Most Common Hashtags')
                st.plotly_chart(fig_hashtags, use_container_width=True)
            else:
                st.info("No hashtags found in the filtered data.")
        else:
            st.error("No content data available for analysis.")
    
    # Tab 3: Network Analysis
    with tab3:
        st.header("Network Analysis")
        
        # User network visualization
        if 'user_id' in filtered_df.columns and 'content' in filtered_df.columns:
            st.subheader("User Interaction Network")
            
            # Filter options for network analysis
            network_filter = st.text_input("Filter network by keyword (optional)", key="network_filter")
            
            # Network visualization
            with st.spinner("Generating network visualization..."):
                try:
                    network_fig = create_network_graph(filtered_df, 'content', network_filter)
                    st.pyplot(network_fig)
                except Exception as e:
                    st.error(f"Error generating network visualization: {str(e)}")
                    st.info("Creating a simplified network graph instead.")
                    
                    # Create a simplified bar chart of user activity
                    user_counts = filtered_df['user_id'].value_counts().reset_index()
                    user_counts.columns = ['User', 'Post Count']
                    
                    fig_users = px.bar(user_counts.head(20), x='User', y='Post Count',
                                    title='Most Active Users')
                    st.plotly_chart(fig_users, use_container_width=True)
            
            # User activity analysis
            st.subheader("User Activity Analysis")
            if 'user_id' in filtered_df.columns:
                user_counts = filtered_df['user_id'].value_counts().reset_index()
                user_counts.columns = ['User', 'Post Count']
                
                # Pie chart of top users
                fig_users_pie = px.pie(user_counts.head(10), names='User', values='Post Count',
                                    title='Top 10 Users by Activity')
                st.plotly_chart(fig_users_pie, use_container_width=True)
        else:
            st.error("User or content data not available for network analysis.")
    
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
                    
                    # Topic evolution over time if timestamp data is available
                    if 'created_at' in filtered_df.columns:
                        st.subheader("Topic Evolution Over Time")
                        st.write("This chart shows how topics might have evolved over the analyzed time period.")
                        
                        # Generate sample data for demonstration
                        date_range = pd.date_range(
                            start=filtered_df['created_at'].min(),
                            end=filtered_df['created_at'].max(),
                            freq='D'
                        )
                        
                        # Create placeholder data for topic evolution
                        topic_evolution_data = []
                        for date in date_range:
                            for topic_idx, topic_name in enumerate(topic_names):
                                # Create some variation in topic strength
                                strength = np.random.normal(10, 3) + (topic_idx * np.sin(date.day/15))
                                if strength < 0:
                                    strength = 0
                                
                                topic_evolution_data.append({
                                    'Date': date,
                                    'Topic': topic_name,
                                    'Strength': strength
                                })
                        
                        topic_evolution_df = pd.DataFrame(topic_evolution_data)
                        
                        # Create line chart
                        fig_evolution = px.line(
                            topic_evolution_df, 
                            x='Date', 
                            y='Strength', 
                            color='Topic',
                            title='Topic Strength Over Time'
                        )
                        st.plotly_chart(fig_evolution, use_container_width=True)
                        
                        st.info("Note: This topic evolution visualization is a simulation for demonstration purposes. In a real implementation, it would be based on actual topic modeling results over time.")
                else:
                    st.warning("Not enough content for topic modeling after text processing.")
        else:
            st.error("No content data available for topic modeling.")
    
    # Tab 5: AI Insights
    with tab5:
        st.header("AI-Generated Insights")
        
        # Generate mock insights
        insights = generate_mock_insights(filtered_df)
        
        # Display insights with fancy formatting
        st.subheader("Content Summary")
        
        for insight in insights:
            st.markdown(f"✨ {insight}")
        
        # Add a chart showing simulated sentiment analysis
        st.subheader("Sentiment Analysis")
        
        # Generate mock sentiment data
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_values = [np.random.randint(20, 50), np.random.randint(30, 60), np.random.randint(10, 30)]
        
        # Create pie chart
        fig_sentiment = px.pie(
            names=sentiment_labels, 
            values=sentiment_values,
            title='Content Sentiment Distribution',
            color=sentiment_labels,
            color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Add simulated reliability score
        st.subheader("Source Reliability Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            reliability_score = np.random.randint(30, 85)
            st.metric("Overall Reliability Score", f"{reliability_score}%")
        
        with col2:
            verified_sources = np.random.randint(10, 40)
            st.metric("Verified Sources", f"{verified_sources}%")
        
        with col3:
            known_unreliable = np.random.randint(5, 25)
            st.metric("Known Unreliable Sources", f"{known_unreliable}%")
        
        st.info("Note: These AI insights are simulated for demonstration purposes. In a real implementation, they would be generated using actual AI models analyzing the dataset.")
    
    # Display raw data if requested
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        st.dataframe(filtered_df.head(100))
    
    # Footer
    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")

if __name__ == "__main__":
    main()
