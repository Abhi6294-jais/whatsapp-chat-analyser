# app.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import io 
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATE_TIME_PATTERN = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:AM|PM)\s-\s'
MEDIA_MESSAGE = '<Media omitted>'
LINK_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-f-F][0-9a-f-F]))+'

EMOJI_PATTERN = re.compile(
    '['
    '\U0001F600-\U0001F64F'  # Emoticons
    '\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
    '\U0001F680-\U0001F6FF'  # Transport and Map Symbols
    '\U0001F700-\U0001F77F'  # Alchemical Symbols
    '\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
    '\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
    '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    '\U0001FA00-\U0001FA6F'  # Chess Symbols
    '\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    '\U00002702-\U000027B0'  # Dingbats
    '\U000024C2-\U0001F251' 
    ']+', re.UNICODE
)

# --- Core Functions ---

@st.cache_data
def parse_chat(chat_content):
    """Parses WhatsApp chat content into a pandas DataFrame."""
    
    messages = re.split(DATE_TIME_PATTERN, chat_content)[1:]
    dates = re.findall(DATE_TIME_PATTERN, chat_content)

    if len(messages) != len(dates):
        st.error("Error: Date and message count mismatch. Please check your chat format.")
        return pd.DataFrame()

    df = pd.DataFrame({'user_message': messages, 'date_raw': dates})
    
    if df.empty:
        return pd.DataFrame()

    # Try multiple date formats
    date_formats = [
        '%m/%d/%y, %I:%M %p - ',
        '%d/%m/%y, %I:%M %p - ',
        '%m/%d/%Y, %I:%M %p - ',
        '%d/%m/%Y, %I:%M %p - '
    ]
    
    df['DateTime'] = pd.NaT
    for fmt in date_formats:
        mask = df['DateTime'].isna()
        df.loc[mask, 'DateTime'] = pd.to_datetime(
            df.loc[mask, 'date_raw'], 
            format=fmt, 
            errors='coerce'
        )

    df.dropna(subset=['DateTime'], inplace=True)
    
    if df.empty:
        return pd.DataFrame()

    users = []
    messages = []
    USER_MESSAGE_PATTERN = r'([\w\W]+?):\s' 
    
    for message in df['user_message']:
        entry = re.split(USER_MESSAGE_PATTERN, message, maxsplit=1)
        if len(entry) > 2: 
            users.append(entry[1])
            messages.append("".join(entry[2:]).strip())
        else:
            users.append('group_notification')
            messages.append(entry[0].strip())

    df['User'] = users
    df['Message'] = messages
    df.drop(columns=['user_message', 'date_raw'], inplace=True)
    df.rename(columns={'DateTime': 'date'}, inplace=True)

    # Feature Extraction
    df['Message_Length'] = df['Message'].apply(lambda x: len(str(x).split()))
    df['Media'] = df['Message'].apply(lambda x: 1 if MEDIA_MESSAGE in str(x) else 0)
    df['Links'] = df['Message'].apply(lambda x: len(re.findall(LINK_PATTERN, str(x))))
    df['Day'] = df['date'].dt.day_name()
    df['Month'] = df['date'].dt.month_name()
    df['Year'] = df['date'].dt.year
    df['Hour'] = df['date'].dt.hour
    df['Date_Only'] = df['date'].dt.date
    
    period = []
    for hour in df['Hour']:
        if 5 <= hour < 12:
            period.append('Morning')
        elif 12 <= hour < 17:
            period.append('Afternoon')
        elif 17 <= hour < 21:
            period.append('Evening')
        else:
            period.append('Night')
    
    df['day_period'] = period

    # Filter out system messages
    df = df[df['User'] != 'group_notification'].copy()
    system_messages = df[df['User'].str.contains('joined|created|left|changed the subject|changed the group icon|was added', regex=True, na=False)].index
    df.drop(system_messages, inplace=True)
    
    return df

@st.cache_data
def extract_emojis(df):
    """Extracts and counts all emojis from the chat messages."""
    all_emojis = Counter()
    df['Emojis'] = df['Message'].apply(lambda x: EMOJI_PATTERN.findall(str(x)))
    
    for emoji_list in df['Emojis']:
        for emoji in emoji_list:
            all_emojis[emoji] += 1
            
    emoji_df = pd.DataFrame(all_emojis.most_common(20), columns=['Emoji', 'Count'])
    return emoji_df

# --- Display Functions ---

def display_overall_stats(df):
    st.header("📊 Overall Chat Metrics")
    
    total_messages = len(df)
    total_words = df['Message_Length'].sum()
    total_media = df['Media'].sum()
    total_links = df['Links'].sum()
    unique_users = df['User'].nunique()
    chat_duration = (df['date'].max() - df['date'].min()).days
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Messages", f"{total_messages:,}")
    col2.metric("Total Words", f"{total_words:,}")
    col3.metric("Media Shared", f"{total_media:,}")
    col4.metric("Links Shared", f"{total_links:,}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Participants", unique_users)
    col2.metric("Chat Duration (Days)", chat_duration)
    col3.metric("Avg Messages/Day", f"{total_messages/max(1, chat_duration):.1f}")
    col4.metric("Avg Words/Message", f"{total_words/max(1, total_messages):.1f}")

def display_user_analysis(df):
    st.header("👥 User Activity Analysis")
    
    user_msg_counts = df['User'].value_counts().sort_values(ascending=False)
    user_word_counts = df.groupby('User')['Message_Length'].sum().sort_values(ascending=False)
    avg_msg_len = df.groupby('User')['Message_Length'].mean().sort_values(ascending=False).round(1)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Users by Messages")
        fig = px.bar(
            x=user_msg_counts.head(10).values,
            y=user_msg_counts.head(10).index,
            orientation='h',
            title="Top 10 Users by Message Count",
            labels={'x': 'Number of Messages', 'y': 'User'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("User Statistics")
        user_stats = pd.DataFrame({
            'Messages': user_msg_counts,
            'Total Words': user_word_counts,
            'Avg Words/Message': avg_msg_len
        })
        st.dataframe(user_stats.head(10), use_container_width=True)

def display_time_analysis(df):
    st.header("⏰ Activity Trends & Patterns")

    # Monthly timeline
    st.subheader("Monthly Message Timeline")
    monthly_activity = df.groupby(df['date'].dt.to_period("M")).size()
    monthly_activity.index = monthly_activity.index.astype(str)
    
    fig = px.line(
        x=monthly_activity.index,
        y=monthly_activity.values,
        title="Message Activity Over Time",
        labels={'x': 'Month', 'y': 'Number of Messages'}
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Activity by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['Day'].value_counts().reindex(day_order)
        
        fig = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            title="Messages by Day of Week",
            labels={'x': 'Day', 'y': 'Number of Messages'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Activity by Month")
        month_counts = df['Month'].value_counts()
        
        fig = px.bar(
            x=month_counts.index,
            y=month_counts.values,
            title="Messages by Month",
            labels={'x': 'Month', 'y': 'Number of Messages'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Activity heatmap
    st.subheader("Weekly Activity Heatmap")
    create_activity_heatmap(df)
    
    # Day period analysis
    st.subheader("Activity by Time of Day")
    period_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    period_counts = df['day_period'].value_counts().reindex(period_order)
    
    fig = px.pie(
        values=period_counts.values,
        names=period_counts.index,
        title="Message Distribution by Time of Day"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_activity_heatmap(df):
    """Creates an interactive heatmap using plotly"""
    activity_map = df.pivot_table(
        index='Day',
        columns='Hour',
        values='Message',
        aggfunc='count',
        fill_value=0
    )

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_map = activity_map.reindex(day_order)

    fig = px.imshow(
        activity_map,
        labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(title="Messages Heatmap: Day vs Hour")
    st.plotly_chart(fig, use_container_width=True)

def display_content_analysis(df):
    st.header("📝 Content & Topic Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Word Cloud")
        stop_words = set(WordCloud().stopwords)
        stop_words.update(['media', 'omitted', 'null', 'p', 'message', 'deleted', 'pm', 'am', 'ok', 'yes', 'no', 'hi', 'hello']) 
        text = " ".join(str(message).lower() for message in df['Message'])
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=stop_words, 
            min_font_size=10,
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title('Most Frequent Words')
        st.pyplot(fig)

    with col2:
        st.subheader("Top Emojis")
        emoji_df = extract_emojis(df)
        if not emoji_df.empty:
            fig = px.bar(
                emoji_df.head(10),
                x='Count',
                y='Emoji',
                orientation='h',
                title='Top 10 Most Used Emojis',
                color='Count',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emojis found in the chat.")

def display_sentiment_analysis(df):
    st.header("😊 Sentiment Analysis")
    st.info("Sentiment analysis feature coming soon!")
    # You can integrate textblob or vaderSentiment here

# --- Main Streamlit App Layout ---

def main_app():
    st.title("💬 WhatsApp Chat Analyzer")
    st.markdown("""
    Upload your **exported WhatsApp chat text file** to get deep insights into your group or individual chat activity.
    
    ### How to Export Your WhatsApp Chat:
    1. Open the WhatsApp chat you want to analyze
    2. Tap on the contact/group name → Export Chat
    3. Choose **Without Media**
    4. Upload the generated .txt file here
    """)

    st.sidebar.title("⚙️ Settings")
    uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Supported Formats:**
    - Android: `MM/DD/YY, HH:MM AM/PM - User: Message`
    - iOS: `DD/MM/YY, HH:MM AM/PM - User: Message`
    """)

    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            chat_content = bytes_data.decode("utf-8")
            
            with st.spinner("🔍 Processing chat data... This may take a few seconds."):
                df = parse_chat(chat_content)

            if df.empty:
                st.error("""
                ❌ Error: Failed to parse chat data. Please ensure:
                - The file is a standard WhatsApp chat export
                - The date format matches supported patterns
                - The file is not empty
                """)
                return

            user_list = df['User'].unique().tolist()
            user_list.sort()
            user_list.insert(0, "Overall")

            st.sidebar.markdown("---")
            selected_user = st.sidebar.selectbox("👤 Show Analysis For:", user_list)
            
            temp_df = df.copy()
            if selected_user != "Overall":
                temp_df = df[df['User'] == selected_user]

            # Display all analysis sections
            display_overall_stats(temp_df)
            
            st.markdown("---")
            display_time_analysis(temp_df)

            st.markdown("---")
            if selected_user == "Overall":
                display_user_analysis(temp_df)
                st.markdown("---")
                
            display_content_analysis(temp_df)
            
            # Optional: Add download button for processed data
            st.sidebar.markdown("---")
            if st.sidebar.button("📥 Download Processed Data"):
                csv = temp_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="whatsapp_chat_analysis.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your file format and try again.")

    else:
        st.info("📁 Waiting for file upload...")
        
        # Show sample analysis
        st.markdown("---")
        st.subheader("📸 Sample Analysis Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", "1,247")
        col2.metric("Active Users", "8")
        col3.metric("Chat Duration", "45 days")

if __name__ == '__main__':
    main_app()