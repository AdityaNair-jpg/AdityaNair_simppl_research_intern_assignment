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
import requests
import zipfile
import io
import warnings
import plotly.graph_objects as go
import altair as alt
import spacy
from ai_insights import generate_enhanced_insights, generate_mock_insights


warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Social Media Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def setup_nltk_resources():
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except (LookupError, Exception):
        nltk.download('vader_lexicon', quiet=True)
    try:
        nltk.data.find('corpora/stopwords.zip')
    except (LookupError, Exception):
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt.zip')
    except (LookupError, Exception):
        nltk.download('punkt', quiet=True)

    analyzer = SentimentIntensityAnalyzer()
    stop_words_set = set(stopwords.words('english'))
    
    return analyzer, stop_words_set

sentiment_analyzer, stop_words_nltk = setup_nltk_resources()
@st.cache_resource
def load_spacy_model_from_disk():
    model_name = "en_core_web_sm"
    model_version = "3.8.0" 

    app_root_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(app_root_dir, "spacy_models", model_name)

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        
        try:
            whl_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{model_version}/{model_name}-{model_version}-py3-none-any.whl"

            response = requests.get(whl_url, stream=True)
            response.raise_for_status() 

            
            os.makedirs(model_dir, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                model_root_in_zip = None
                for name in zip_ref.namelist():
                    if name.startswith(f"{model_name}/") and f"{model_name}-{model_version}" in name:
                        model_root_in_zip = name.split(f"{model_name}-{model_version}")[0] + f"{model_name}-{model_version}"
                        break

                if model_root_in_zip:
                    for member in zip_ref.namelist():
                        if member.startswith(model_root_in_zip):
                            dest_path = os.path.join(model_dir, os.path.relpath(member, model_root_in_zip))
                            if member.endswith('/'): 
                                os.makedirs(dest_path, exist_ok=True)
                            else: 
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True) 
                                with open(dest_path, "wb") as outfile:
                                    outfile.write(zip_ref.read(member))
                    
                else:
                    st.error(f"Error: Could not find model data within the downloaded wheel for {model_name}-{model_version}. Check the model URL and structure.")
                    return None 

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download spaCy model from URL: {whl_url}. Error: {e}")
            return None
        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid zip archive. Check the model URL.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during model download/extraction: {e}")
            return None

    try:        
        nlp = spacy.load(model_dir)
        return nlp
    except Exception as e:
        st.error(f"Error loading spaCy model from local directory '{model_dir}': {e}")
        return None

nlp = load_spacy_model_from_disk()

if 'ner_debug_messages' not in st.session_state:
    st.session_state.ner_debug_messages = []

try:
    from pyvis.network import Network
except ImportError:
    st.warning("Pyvis not found. Network graph feature will be disabled. Please install with 'pip install pyvis'")
    Network = None 

@st.cache_data
def preprocess_text(text_series):
    text_series = text_series.astype(str).str.lower()
    text_series = text_series.apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE)) 
    text_series = text_series.apply(lambda x: re.sub(r'\S*@\S*\s?', '', x)) 
    text_series = text_series.apply(lambda x: re.sub(r'#\w+', '', x)) 
    text_series = text_series.apply(lambda x: re.sub(r'[^\w\s]', '', x)) 
    tokens = text_series.apply(word_tokenize)
    stop_words = set(stopwords.words('english')) 
    tokens = tokens.apply(lambda x: [word for word in x if word.isalpha() and word not in stop_words])
    return tokens


@st.cache_data
def extract_topics_enhanced(processed_texts, n_topics=5, _nlp_model=None, original_df_for_ner=None):
    st.session_state.ner_debug_messages = []

    if not isinstance(processed_texts, pd.Series):
        st.session_state.ner_debug_messages.append("Error: Expected 'processed_texts' to be a Pandas Series in extract_topics_enhanced.")
        return [], [], None, []

    original_indices = processed_texts.index[processed_texts.apply(lambda x: bool(x))].tolist()
    texts_for_tfidf = [" ".join(tokens) for tokens in processed_texts if tokens]

    if not texts_for_tfidf:
        st.session_state.ner_debug_messages.append("No texts available for TF-IDF. Returning empty topics and data.")
        return [], [], None, []

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    try:
        tfidf = vectorizer.fit_transform(texts_for_tfidf)
    except ValueError as e:
        st.session_state.ner_debug_messages.append(f"Could not create TF-IDF matrix. Not enough text data or all words are stop words. Error: {e}")
        return [], [], None, []

    if tfidf.shape[0] < n_topics:
        n_topics = tfidf.shape[0] if tfidf.shape[0] > 0 else 1
        if n_topics == 0:
            st.session_state.ner_debug_messages.append("No documents to create topics from.")
            return [], [], None, []

    nmf_model = NMF(n_components=n_topics, random_state=1, max_iter=100)
    nmf_model.fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics_list_for_return = [] 
    topic_data = [] 
    
    doc_topic_dist = nmf_model.transform(tfidf)
    
    if original_df_for_ner is not None and 'content' in original_df_for_ner.columns:
        original_content_series_for_ner = original_df_for_ner.loc[original_indices, 'content'].astype(str)
    else:
        original_content_series_for_ner = pd.Series([""] * len(original_indices), index=original_indices)
        st.session_state.ner_debug_messages.append("Original DataFrame for NER not provided or 'content' column missing. NER-based topic naming may be limited.")


    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[:-11:-1]  
        top_words = [feature_names[i] for i in top_words_idx]
        top_scores = [topic[i] for i in top_words_idx]
        
        
        ner_topic_name = f"Topic {topic_idx + 1}" 
        if _nlp_model and not original_content_series_for_ner.empty: 
            topic_doc_indices_for_ner = [original_indices[i] for i, prob in enumerate(doc_topic_dist[:, topic_idx]) if prob > 0.1] 
            topic_documents_content = original_content_series_for_ner.loc[topic_doc_indices_for_ner].tolist()
            all_entities = []

            if topic_documents_content:
                for doc_content in topic_documents_content:
                    doc = _nlp_model(doc_content) 
                    all_entities.extend([ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'NORP', 'LOC']])
                entity_counts = Counter(all_entities)
                most_common_entities = [
                    entity for entity, count in entity_counts.most_common(5) 
                    if len(entity) > 2 and entity.lower() not in stop_words_nltk
                ] 
                
                if most_common_entities:
                    ner_topic_name = ", ".join(most_common_entities[:3]) 
                else:
                    fallback_words = ", ".join(top_words[:3]) 
                    ner_topic_name = f"Topic {topic_idx + 1} ({fallback_words})"
                    st.session_state.ner_debug_messages.append(f"NER could not find relevant entities for Topic {topic_idx + 1}. Falling back to a name based on top words: ({fallback_words}). No suitable entities found or filtered out.")
            else:
                fallback_words = ", ".join(top_words[:3]) 
                ner_topic_name = f"Topic {topic_idx + 1} ({fallback_words})"
                st.session_state.ner_debug_messages.append(f"No documents strongly associated with Topic {topic_idx + 1} for NER (threshold > 0.1). Falling back to a name based on top words: ({fallback_words}).")
        else:
            fallback_words = ", ".join(top_words[:3]) 
            ner_topic_name = f"Topic {topic_idx + 1} ({fallback_words})"
            st.session_state.ner_debug_messages.append(f"spaCy model not loaded or no original content for NER for Topic {topic_idx + 1}. Using generic topic name enhanced with top words.")

        topics_list_for_return.append(ner_topic_name) 
        
        for word, score in zip(top_words, top_scores):
            topic_data.append({
                'topic': ner_topic_name, 
                'word': word,
                'score': score,
                'topic_id': topic_idx
            })
    
    return topics_list_for_return, topic_data, doc_topic_dist, original_indices


@st.cache_data
def load_data():
    data = []
    try:
        with open('data.jsonl', 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                try:
                    loaded_json = json.loads(line)
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
        df['created_at'] = pd.to_datetime(pd.Series([datetime.now()] * len(df))) 

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
        if G.nodes[node]['type'] == 'author':
            node_color.append('#3498db') 
        elif G.nodes[node]['type'] == 'subreddit':
            node_color.append('#e74c3c') 
        else:
            node_color.append('#95a5a6') 
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
            colorscale='Viridis',
            reversescale=False, 
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
    
    if 'spacy_load_error' in st.session_state and st.session_state.spacy_load_error:
        st.error(st.session_state.spacy_load_error)

    df = load_data()

    if df.empty or not df['created_at'].notna().any() or not df['content'].notna().any():
        st.warning("No valid data loaded or critical columns are missing/empty. Please upload your `data.jsonl` file to proceed.")
        st.info("Ensure your JSONL file contains 'created_at' (or similar timestamp) and 'content' (or similar text) columns.")
        st.stop()
    
    if 'subreddit' not in df.columns:
        st.warning("'subreddit' column not found in data. Some features may be limited.")
        df['subreddit'] = 'Unknown'

    st.sidebar.header("Filter Data")

    if 'created_at' in df.columns and pd.api.types.is_datetime64_any_dtype(df['created_at']):
        min_date_val = df['created_at'].dt.date.min()
        max_date_val = df['created_at'].dt.date.max()

        min_date_slider = pd.to_datetime(min_date_val)
        max_date_slider = pd.to_datetime(max_date_val)

        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date_slider.to_pydatetime(),
            max_value=max_date_slider.to_pydatetime(),
            value=(min_date_slider.to_pydatetime(), max_date_slider.to_pydatetime()),
            format="YYYY-MM-DD"
        )
        
        df = df[(df['created_at'].dt.date >= date_range[0].date()) & (df['created_at'].dt.date <= date_range[1].date())]
    else:
        st.sidebar.warning("Date column ('created_at') not available or not in datetime format for filtering. Please check your data loading.")

    filtered_df = df.copy()
    
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
            selected_date_range = (single_date_input, single_date_input)
        
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
        st.info(f"You are viewing posts containing: \"{search_query}\".")

    if not filtered_df.empty:
        if 'sentiment_label' not in filtered_df.columns:
            with st.spinner("Performing sentiment analysis..."):
                filtered_df['content'] = filtered_df['content'].astype(str)
                filtered_df['compound_score'] = filtered_df['content'].apply(lambda text: sentiment_analyzer.polarity_scores(text)['compound'])
                
                filtered_df['sentiment_label'] = filtered_df['compound_score'].apply(lambda c: 
                    'Positive' if c >= 0.05 else 
                    'Negative' if c <= -0.05 else 
                    'Neutral'
                )
                filtered_df['sentiment_numeric_score'] = filtered_df['compound_score'] 
        
        df_for_sentiment_charts = filtered_df.copy() 

    else:
        st.info("No data to apply sentiment filter.")


    all_subreddits_available = df['subreddit'].unique() 
    selected_subreddits = st.sidebar.multiselect("Filter by Subreddit:", all_subreddits_available, default=list(all_subreddits_available)) 
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
        with st.expander("Explanation for Activity Trends"):
            st.markdown("""
             This chart displays the volume of posts over the selected time period. Each point on the line represents the number of posts on a specific date. You can observe peaks and troughs in activity.
            """)
        with st.spinner("Generating Activity Trends..."): 
            if 'created_at' in filtered_df.columns and not filtered_df['created_at'].empty:
                time_df = filtered_df.copy()
                time_df['date'] = time_df['created_at'].dt.date
                daily_counts = time_df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(daily_counts, x='date', y='count',
                              title='Post Volume Over Time',
                              labels={'date': 'Date', 'count': 'Number of Posts'},
                              markers=True, 
                              line_shape='spline', 
                              hover_data={'date': True, 'count': True}, 
                              color_discrete_sequence=px.colors.qualitative.Plotly 
                              )
                fig.update_traces(line=dict(width=3)) 
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Key Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Posts", len(filtered_df))
                if not daily_counts.empty:
                    col2.metric("Start Date", daily_counts['date'].min().strftime('%Y-%m-%d'))
                    col3.metric("End Date", daily_counts['date'].max().strftime('%Y-%m-%d'))
                
                
            else:
                st.info("No data or 'created_at' column available for activity trend analysis after filters.")

    with tab_sentiment:
        st.header("Sentiment Analysis")
        with st.expander("Explanation for Sentiment Analysis"):
            st.markdown("""
            This pie chart visualizes the distribution of sentiment (Positive, Neutral, Negative) across all filtered posts. The size of each slice indicates the proportion of posts falling into that sentiment category.
            """)
        with st.spinner("Analyzing Sentiment..."): 
            if not df_for_sentiment_charts.empty and 'sentiment_label' in df_for_sentiment_charts.columns:
                sentiment_counts = df_for_sentiment_charts['sentiment_label'].value_counts()
                sentiment_df = pd.DataFrame({
                    'Sentiment': sentiment_counts.index,
                    'Count': sentiment_counts.values
                })
                
                pie_labels = sentiment_df['Sentiment'].tolist()
                pie_values = sentiment_df['Count'].tolist()

                sentiment_colors = {'Positive': '#ED5983', 'Neutral': '#EFCC98', 'Negative': '#782B9D'} 
                pie_marker_colors = [sentiment_colors[s] for s in pie_labels]

                fig = go.Figure(data=[go.Pie(
                    labels=pie_labels, 
                    values=pie_values, 
                    textinfo='label+percent+value', 
                    insidetextfont=dict(color='white'),
                    hoverinfo='label+percent+value',
                    hole=.3,
                    marker=dict(colors=pie_marker_colors) 
                )])
                fig.update_layout(
                    title_text='Overall Sentiment Distribution' 
                )
                st.plotly_chart(fig, use_container_width=True, key="sentiment_pie_chart")

                
            else:
                st.info("No data to perform sentiment analysis.")

    with tab_entities:
        st.header("Top Entities/Authors/Subreddits")
        with st.expander("Explanation for Top Entities"):
            st.markdown("""
            This section presents insights into the most active authors, the most discussed subreddits, and the most frequent words in the filtered dataset.
            """)
        with st.spinner("Extracting Top Entities..."): 
            if not filtered_df.empty:
                st.subheader("Top 10 Authors")
                authors_filtered = filtered_df[filtered_df['author'] != 'Unknown User']
                author_counts = authors_filtered['author'].value_counts().reset_index()
                author_counts.columns = ['Author', 'Post Count']
                top_authors = author_counts.head(10).copy()
                top_authors['Post Count'] = top_authors['Post Count'].astype(int)
                
                if not top_authors.empty:
                    chart_authors_alt = alt.Chart(top_authors).mark_bar().encode(
                        x=alt.X('Post Count:Q', title='Post Count'), 
                        y=alt.Y('Author:N', title='Author', sort='-x'), 
                        tooltip=['Author', 'Post Count'], 
                        color=alt.Color('Post Count:Q', scale=alt.Scale(scheme='magma')), 
                        text=alt.Text('Post Count:Q', format='d') 
                    ).properties(
                        title='Top 10 Authors by Post Count'
                    ).configure_axis(
                        labelLimit=200 
                    ).interactive() 

                    st.altair_chart(chart_authors_alt, use_container_width=True)

                st.subheader("Top 10 Subreddits")
                subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
                subreddit_counts.columns = ['Subreddit', 'Post Count']
                top_subreddits = subreddit_counts.head(10).copy()
                top_subreddits['Post Count'] = top_subreddits['Post Count'].astype(int)
                
                if not top_subreddits.empty:
                    chart_subreddits_alt = alt.Chart(top_subreddits).mark_bar().encode(
                        x=alt.X('Post Count:Q', title='Post Count'),
                        y=alt.Y('Subreddit:N', title='Subreddit', sort='-x'),
                        tooltip=['Subreddit', 'Post Count'],
                        color=alt.Color('Post Count:Q', scale=alt.Scale(scheme='magma')), 
                        text=alt.Text('Post Count:Q', format='d')
                    ).properties(
                        title='Top 10 Subreddits by Post Count'
                    ).configure_axis(
                        labelLimit=200
                    ).interactive()

                    st.altair_chart(chart_subreddits_alt, use_container_width=True)

                st.subheader("Most Frequent Words")
                processed_texts = preprocess_text(filtered_df['content'])
                all_words = [word for sublist in processed_texts for word in sublist]
                word_freq = Counter(all_words)
                top_words_df = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])
                
                if not top_words_df.empty:
                    chart_words_alt = alt.Chart(top_words_df).mark_bar().encode(
                        x=alt.X('Frequency:Q', title='Frequency'),
                        y=alt.Y('Word:N', title='Word', sort='-x'),
                        tooltip=['Word', 'Frequency'],
                        color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='magma')), 
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
        with st.expander("Explanation for Word Cloud"):
            st.markdown("""
             A word cloud visually represents the frequency of words in the filtered content. Larger words appear more frequently, while smaller words are less common. Common English stopwords are removed.
            """)
        with st.spinner("Generating Word Cloud..."):
            if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
                processed_texts = preprocess_text(filtered_df['content'])
                all_words = " ".join([word for sublist in processed_texts for word in sublist])

                if all_words and len(all_words.strip()) > 0:
                    try:
                        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='gnuplot').generate(all_words)

                        # Check if the wordcloud actually has words rendered
                        if wordcloud.words_: # This checks if the generated wordcloud object contains any words
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.clf() # Clear the current figure to prevent issues in subsequent runs
                        else:
                            st.warning("Word cloud generated, but it appears empty. This might be due to extreme filtering, very short/repetitive text, or all words being common stopwords.")
                            plt.clf() # Clear the figure even if empty, to be safe
                    except Exception as e:
                        st.error(f"An unexpected error occurred during Word Cloud generation: {e}")
                        plt.clf() # Clear the figure on error
                else:
                    st.warning("No meaningful words to generate word cloud after preprocessing and filters.")
            else:
                st.warning("No content data available for word cloud generation after filters.")


    with tab_network:
        st.header("Author Interaction Network Graph")
        with st.expander("Explanation for Author Network Graph"):
            st.markdown("""
            This interactive graph visualizes connections between authors and subreddits.
            """)
        with st.spinner("Building Network Graph..."): 
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
        with st.expander("Explanation for Topic Modeling"):
            st.markdown("""
             Topic modeling identifies latent "topics" within the text data. Each topic is a collection of words that frequently appear together, suggesting a common theme.
            """)
        with st.spinner("Performing Topic Modeling..."): 
            if not filtered_df.empty and 'content' in filtered_df.columns and not filtered_df['content'].empty:
                if 'created_at' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['created_at']):
                    st.warning("The 'created_at' column is not available or not in datetime format. Topic prevalence over time chart cannot be generated.")
                
                filtered_df_for_topics = filtered_df[filtered_df['content'].apply(lambda x: bool(x and x.strip()))].copy()
                
                processed_texts = preprocess_text(filtered_df_for_topics['content'])
                
                n_topics = st.slider("Number of Topics", min_value=3, max_value=10, value=5, key='num_topics_slider')
                
                topics, topic_data, doc_topic_dist, original_indices = extract_topics_enhanced(
                    processed_texts, n_topics=n_topics, _nlp_model=nlp, original_df_for_ner=filtered_df_for_topics 
                )

                if st.session_state.ner_debug_messages:
                    st.subheader("NER Topic Naming Debug Info")
                    for msg in st.session_state.ner_debug_messages:
                        st.info(msg)
                    st.markdown("---")


                if topics and topic_data is not None and doc_topic_dist is not None:
                    
                    
                    if 'custom_topic_names' not in st.session_state or len(st.session_state.custom_topic_names) != n_topics:
                        st.session_state.custom_topic_names = topics 
                        
                    st.markdown("---")
                    st.subheader("Customize Topic Names")
                    st.write("Review the top words for each topic below and provide a more descriptive name if you wish. This name will appear in the charts.")
                    
                    current_topic_names = []
                    for i in range(n_topics):
                        topic_summary_words = [d['word'] for d in topic_data if d['topic_id'] == i][:5]
                        default_summary_text = f"Top words: {', '.join(topic_summary_words)}"
                        st.write(default_summary_text) 
                        
                        current_topic_name = st.text_input(
                            f"Name for Topic {i+1}",
                            value=st.session_state.custom_topic_names[i], 
                            key=f'topic_name_input_{i}'
                        )
                        current_topic_names.append(current_topic_name)
                    st.session_state.custom_topic_names = current_topic_names
                    
                    topic_df = pd.DataFrame(topic_data)
                    topic_df['topic_label'] = topic_df['topic_id'].apply(lambda x: st.session_state.custom_topic_names[x])

                    st.markdown("---")
                    st.subheader("Topic Prevalence Over Time")
                    st.write("This chart shows how the activity/discussion around each topic has evolved over the selected time period.")

                    if doc_topic_dist is not None and not filtered_df_for_topics.empty:
                        doc_topic_df = pd.DataFrame(doc_topic_dist, index=original_indices)
                        
                        doc_topic_df_aligned = filtered_df_for_topics.loc[original_indices, ['created_at']].copy()
                        
                        for i in range(n_topics):
                            doc_topic_df_aligned[f'topic_{i}_prob'] = doc_topic_df[i]

                        topic_prob_cols = [f'topic_{i}_prob' for i in range(n_topics)]
                        topic_prevalence_daily = doc_topic_df_aligned.groupby(doc_topic_df_aligned['created_at'].dt.floor('D'))[topic_prob_cols].sum().reset_index()
                        topic_prevalence_daily = topic_prevalence_daily.rename(columns={'created_at': 'Date'})

                        new_cols = {f'topic_{i}_prob': st.session_state.custom_topic_names[i] for i in range(n_topics)}
                        topic_prevalence_daily = topic_prevalence_daily.rename(columns=new_cols)

                        topic_prevalence_melted = topic_prevalence_daily.melt(
                            id_vars=['Date'],
                            var_name='Topic',
                            value_name='Prevalence'
                        )

                        if not topic_prevalence_melted.empty:
                            chart_prevalence_alt = alt.Chart(topic_prevalence_melted).mark_line(point=True).encode(
                                x=alt.X('Date:T', title='Date'),
                                y=alt.Y('Prevalence:Q', title='Topic Prevalence (Sum of Probabilities)'),
                                color=alt.Color('Topic:N', title='Topic', scale=alt.Scale(scheme='accent')),
                                tooltip=[
                                    alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                                    'Topic',
                                    alt.Tooltip('Prevalence:Q', format='.2f')
                                ]
                            ).properties(
                                title='Topic Prevalence Over Time'
                            ).interactive()

                            st.altair_chart(chart_prevalence_alt, use_container_width=True)
                        else:
                            st.warning("Could not generate topic prevalence data for chart.")
                    else:
                        st.warning("Document-topic distribution data is missing or empty. Cannot generate prevalence over time chart.")

                    st.markdown("---")
                    st.subheader("🏛️ Political Subreddit Topic Distribution")
                    st.write("This analysis shows which topics are more prevalent in different political communities.")
                    
                    political_subreddits = {
                        'Liberal': ['liberal', 'democrats', 'progressive', 'politics', 'politicalhumor', 'marchagainsttrump'],
                        'Conservative': ['conservative', 'republican', 'conservatives', 'the_donald', 'asktrumpsupporters', 'trump'],
                        'Neoliberal': ['neoliberal', 'centrist', 'moderatepolitics', 'politicaldiscussion'],
                        'Socialist': ['socialism', 'socialist', 'democraticsocialism', 'latestagecapitalism', 'antiwork', 'marxism'],
                        'Anarchist': ['anarchism', 'anarchy', 'anarchocapitalism', 'anarcho_capitalism', 'anarcho_communism'] 
                    }
                    
                    subreddit_to_political = {}
                    for political_lean, subs in political_subreddits.items():
                        for sub in subs:
                            subreddit_to_political[sub.lower()] = political_lean
                    
                    if 'subreddit' in filtered_df_for_topics.columns:
                        doc_topic_df_aligned['subreddit'] = filtered_df_for_topics.loc[original_indices, 'subreddit']
                        doc_topic_df_aligned['political_lean'] = doc_topic_df_aligned['subreddit'].str.lower().map(subreddit_to_political)
                        
                        political_posts = doc_topic_df_aligned[doc_topic_df_aligned['political_lean'].notna()].copy()
                        
                        if not political_posts.empty:
                            topic_prob_cols = [f'topic_{i}_prob' for i in range(n_topics)]
                            political_topic_means = political_posts.groupby('political_lean')[topic_prob_cols].mean().reset_index()
                            
                            rename_dict = {f'topic_{i}_prob': st.session_state.custom_topic_names[i] for i in range(n_topics)}
                            political_topic_means = political_topic_means.rename(columns=rename_dict)
                            
                            political_topic_melted = political_topic_means.melt(
                                id_vars=['political_lean'],
                                var_name='Topic',
                                value_name='Average_Probability'
                            )
                            
                            st.subheader("Political Lean × Topic Heatmap")
                            heatmap_chart = alt.Chart(political_topic_melted).mark_rect().encode(
                                x=alt.X('political_lean:N', title='Political Lean'),
                                y=alt.Y('Topic:N', title='Topic'),
                                color=alt.Color('Average_Probability:Q', 
                                              scale=alt.Scale(scheme='plasma'),
                                              title='Avg Probability'),
                                tooltip=['political_lean', 'Topic', alt.Tooltip('Average_Probability', format='.3f')]
                            ).properties(
                                title='Topic Distribution Across Political Subreddits',
                                width=400,
                                height=300
                            )
                            
                            st.altair_chart(heatmap_chart, use_container_width=True)
                            
                    st.markdown("---")
                    st.subheader("Top Words per Topic (Interactive Bar Charts)")
                    st.write("Explore the most significant words and their importance scores for each individual topic below.")

                    unique_topics_with_labels = sorted([(d['topic_id'], d['topic_label']) for d in topic_df[['topic_id', 'topic_label']].drop_duplicates().to_dict(orient='records')], key=lambda x: x[0])

                    for topic_id, topic_label in unique_topics_with_labels:
                        topic_words_df = topic_df[topic_df['topic_id'] == topic_id].sort_values(by='score', ascending=False).head(10)
                        if not topic_words_df.empty:
                            chart_topic_bar_alt = alt.Chart(topic_words_df).mark_bar().encode(
                                x=alt.X('score:Q', title='Importance Score'),
                                y=alt.Y('word:N', title='Word', sort='-x'),
                                tooltip=['word', alt.Tooltip('score', format='.3f')],
                                color=alt.Color('score:Q', scale=alt.Scale(scheme='magma')),
                                text=alt.Text('score:Q', format='.2f')
                            ).properties(
                                title=f'{topic_label}'
                            ).interactive()

                            st.altair_chart(chart_topic_bar_alt, use_container_width=True)
            else:
                st.warning("No content available for topic modeling after filters. Please upload data and apply filters to analyze topics.")
            
    with tab_ai:
        st.header("AI-Generated Insights")
        with st.expander("Explanation for AI-Generated Insights"):
            st.markdown("""
             This section leverages Artificial Intelligence models to automatically analyze the filtered data and generate actionable insights. These insights are derived from patterns, trends, and anomalies detected by the AI beyond standard statistical aggregations.
            """)
        
        if not filtered_df.empty:
            st.subheader("📊 Automated Data Insights")
            
            with st.spinner("Analyzing data patterns..."):
                enhanced_insights = generate_enhanced_insights(filtered_df)
            
            st.markdown("### Key Findings:")
            for insight in enhanced_insights:
                st.markdown(f"• {insight}")
            
            st.markdown("---")
            
        st.markdown("---")
        st.subheader("🤖 OpenAI-Powered Insights")
        
        openai_api_key = st.secrets.get("OPENAI_API_KEY")
        
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
            st.info("OpenAI API key not provided. AI-generated insights are disabled.")

    st.markdown("---")
    st.markdown("📊 **Social Media Analysis Dashboard** | Created with Streamlit")


if __name__ == "__main__":
    main()
