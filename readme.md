📊 Social Media Analysis Dashboard
This Streamlit application provides a powerful and interactive dashboard for analyzing social media data. It enables users to upload their data, apply various filters, and gain insights through comprehensive visualizations covering activity trends, sentiment, key entities, topic modeling, and network analysis. The dashboard also integrates with AI models for enhanced insights.

✨ Features
Flexible Data Ingestion:

Loads social media data from a data.jsonl file.

Automatically identifies and parses common timestamp columns (created_utc, created, timestamp, created_at, date, datetime, post_timestamp).

Intelligently detects and uses content columns (selftext, title, content, text, body, message, subreddit) for analysis.

Extracts author information from various fields (author, username, user.screen_name).

Dynamic Data Filtering:

Date Range Slider: Filter posts by selecting a precise date range.

Keyword Search: Search for specific keywords (case-insensitive, comma-separated) within post content.

Subreddit Multiselect: Filter posts by choosing one or more subreddits.

Interactive Overview Metrics:

Displays real-time metrics for total posts, filtered posts, unique authors, and unique subreddits.

Activity Trends Visualization:

Plots post volume over time using an interactive Plotly line chart, highlighting activity peaks and troughs.

Comprehensive Sentiment Analysis:

Performs sentiment analysis on post content using NLTK's VADER lexicon.

Categorizes sentiment as Positive, Neutral, or Negative.

Visualizes the overall sentiment distribution with an interactive Plotly pie chart.

Top Entities and Frequency Analysis:

Identifies and visualizes the top 10 most active authors by post count.

Displays the top 10 most frequently posted subreddits.

Presents the top 15 most frequent words after text preprocessing (tokenization, stop word removal).

All these are rendered using interactive Altair bar charts.

Dynamic Word Cloud Generation:

Generates a visually engaging word cloud from the most frequent words in the filtered content, providing quick insights into dominant themes.

Author Interaction Network Graph:

Constructs and visualizes a network graph showing connections between authors and subreddits based on posting activity and mentions.

Allows for downloading the generated graph as a PNG image.

(Requires pyvis to be installed)

Advanced Topic Modeling:

Applies Non-negative Matrix Factorization (NMF) to uncover latent "topics" within the text data.

Intelligent Topic Naming: Attempts to name topics descriptively using Named Entity Recognition (NER) via a spaCy model, falling back to top keywords if NER entities are not prominent.

Customizable Topic Names: Provides an interface for users to review and rename automatically generated topics for better clarity.

Topic Prevalence Over Time: Visualizes how the discussion around each topic evolves over the selected date range.

Political Subreddit Topic Distribution: Analyzes and visualizes topic prevalence across predefined political communities (Liberal, Conservative, Neoliberal, Socialist, Anarchist) using an interactive heatmap.

AI-Generated Insights:

Includes a section for automated data insights derived from detected patterns and anomalies.

Integrates with OpenAI (requires API key) to provide advanced AI-generated textual insights based on the filtered content.

🚀 Installation
To set up and run the Social Media Analysis Dashboard, follow these steps:

Clone the repository:

git clone https://github.com/your-username/social-media-dashboard.git # Replace with your actual repo URL
cd social-media-dashboard

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:
Create a requirements.txt file with the following content and then install them:

streamlit
pandas
numpy
matplotlib
seaborn
plotly
networkx
nltk
scikit-learn
wordcloud
spacy
pyvis # Optional, for network graph
requests
altair

Now, install the dependencies:

pip install -r requirements.txt

Download NLTK and spaCy Resources:
The application will attempt to download the necessary NLTK (vader_lexicon, stopwords, punkt) and spaCy (en_core_web_sm) resources automatically on the first run. This might take some time. Ensure you have an active internet connection.

💾 Data Format
The dashboard expects your social media data to be in a JSON Lines (.jsonl) file named data.jsonl located in the same directory as main.py. Each line in this file must be a valid JSON object.

The application is designed to be flexible with column names, but it primarily looks for:

Content/Text: content, selftext, title, text, body, message

Timestamp: created_utc (Unix timestamp), created, timestamp, created_at, date, datetime, post_timestamp (various formats)

Author: author, username, user.screen_name (nested)

Subreddit: subreddit

Example data.jsonl structure:

{"id": "post1", "created_utc": 1678886400, "author": "userA", "subreddit": "politics", "content": "This is an interesting discussion about current political events."}
{"id": "post2", "created_at": "2023-03-15T14:30:00Z", "username": "userB", "subreddit": "news", "text": "Breaking news: Major policy changes announced today."}
{"id": "post3", "timestamp": 1678972800, "user": {"screen_name": "userC"}, "message": "Just sharing my thoughts on the latest trends."}

🔑 OpenAI API Key (Optional)
To enable the OpenAI-powered AI insights, you need to provide your OpenAI API key.

Create a directory named .streamlit in your project's root folder if it doesn't exist.

Inside .streamlit, create a file named secrets.toml.

Add your OpenAI API key to secrets.toml as follows:

OPENAI_API_KEY="your_openai_api_key_here"

If the OPENAI_API_KEY is not found, the OpenAI-powered insights will be disabled, but all other dashboard features will remain fully functional.

▶️ How to Run
After setting up the environment and preparing your data.jsonl file, run the application using Streamlit:

streamlit run main.py

This command will start the Streamlit server and open the dashboard in your default web browser.


Explanation:

User Web Browser (Client): This is where the user interacts with the dashboard. Streamlit renders the Python code into an interactive web interface, allowing users to apply filters, view charts, and explore insights.

Streamlit Application (Python/Streamlit):

Data Loading & Preprocessing: The core of the application, responsible for reading the data.jsonl file, parsing timestamps, identifying content, and performing initial text preprocessing (e.g., tokenization, stop word removal). It leverages libraries like Pandas, NLTK, and spaCy.

Analysis Modules: Dedicated sections for performing various analyses:

Sentiment Analysis: Computes sentiment scores for posts.

Topic Modeling: Identifies underlying themes in the text data.

Network Analysis: Builds and analyzes relationships between authors and subreddits.

Entity Extraction: Identifies key entities (persons, organizations, locations) within the text.

Visualization Engine: Generates interactive charts and graphs using libraries such as Plotly, Altair, WordCloud, and Pyvis, presenting the analytical results visually.

AI Insights Module (ai_insights.py): This module (expected to be implemented by the user) handles the logic for generating automated and OpenAI-powered insights, potentially interacting with external AI services.

External Resources/APIs:

NLTK Data (Local): Lexicons and corpora required for natural language processing tasks (e.g., sentiment analysis, stop word removal). These are downloaded and stored locally.

spaCy Model (Local): A pre-trained language model used for advanced text processing, particularly Named Entity Recognition (NER). This is also downloaded and stored locally.

OpenAI API (for AI Insights): An optional external service that the ai_insights.py module can call to generate more sophisticated AI-driven textual insights.

📁 Project Structure
social-media-dashboard/
├── main.py                     # Main Streamlit application script
├── data.jsonl                  # Your social media data (JSON Lines format)
├── ai_insights.py              # Module for AI insight generation (needs to be implemented)
├── requirements.txt            # List of Python dependencies
├── spacy_models/               # Directory for downloaded spaCy models (created automatically)
└── .streamlit/                 # Streamlit configuration directory
    └── secrets.toml            # Optional: Stores your OpenAI API key

📝 Notes and Troubleshooting
ai_insights.py: The main.py script imports functions (generate_enhanced_insights, generate_mock_insights) from ai_insights.py. This file is crucial for the "AI Insights" tab. You will need to create and implement these functions. generate_mock_insights suggests a placeholder for actual OpenAI integration.

Performance with Large Datasets: For very large data.jsonl files, some processing steps (especially network graph generation and topic modeling) might take time. Streamlit's caching mechanisms (@st.cache_data, @st.cache_resource) are used to optimize performance for subsequent interactions.

pyvis Requirement: The "Author Network Graph" tab requires the pyvis library. If you encounter an error or the graph doesn't display, ensure pyvis is installed (pip install pyvis).

Data Quality: The accuracy of the insights heavily depends on the quality and consistency of your input data. Ensure your data.jsonl file is well-formatted and contains relevant content and timestamp information.

NLTK/spaCy Downloads: If you face issues with NLTK or spaCy resources, try running python -m nltk.downloader all and python -m spacy download en_core_web_sm manually in your activated virtual environment.