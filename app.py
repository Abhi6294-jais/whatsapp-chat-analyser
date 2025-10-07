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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

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

# --- ML Functions ---

def predict_future_activity(df):
    """Use Linear Regression to predict future message volume"""
    try:
        # Ensure we have datetime data
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        if df.empty:
            return None, None, None
        
        # Prepare data: daily message counts
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'message_count']
        daily_counts = daily_counts.sort_values('date')
        
        # Ensure date column is proper datetime
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Create features: days since start
        daily_counts['days'] = (daily_counts['date'] - daily_counts['date'].min()).dt.days
        
        if len(daily_counts) < 5:
            return None, None, None
        
        # Prepare features and target
        X = daily_counts['days'].values.reshape(-1, 1)
        y = daily_counts['message_count'].values
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Future predictions (next 30 days)
        future_days = np.array(range(daily_counts['days'].max() + 1, daily_counts['days'].max() + 31)).reshape(-1, 1)
        future_predictions = model.predict(future_days)
        
        # Ensure no negative predictions
        future_predictions = np.maximum(future_predictions, 0)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        return daily_counts, future_predictions, (r2, mse, model.coef_[0], model.intercept_)
    
    except Exception as e:
        return None, None, None

def calculate_chat_health(df):
    """Calculate overall chat health score (0-100)"""
    try:
        scores = []
        
        # Activity consistency (30%)
        daily_counts = df.groupby(df['date'].dt.date).size()
        daily_var = daily_counts.std()
        consistency_score = max(0, 100 - (daily_var * 2))
        scores.append(consistency_score * 0.3)
        
        # User participation (30%)
        user_distribution = df['User'].value_counts(normalize=True)
        participation_score = 100 * (1 - user_distribution.std())
        scores.append(participation_score * 0.3)
        
        # Growth trend (20%) - Use linear regression trend
        result = predict_future_activity(df)
        if result[0] is not None:
            _, _, metrics = result
            _, _, slope, _ = metrics
            growth_score = 50 + (slope * 10)
        else:
            growth_score = 50
        scores.append(max(0, min(100, growth_score)) * 0.2)
        
        # Engagement diversity (20%)
        unique_users = df['User'].nunique()
        total_messages = len(df)
        diversity_score = min(100, (unique_users / total_messages) * 1000) if total_messages > 0 else 0
        scores.append(diversity_score * 0.2)
        
        return sum(scores)
    except:
        return 50  # Default score if calculation fails

def predict_optimal_posting_time(df):
    """Predict best time to send messages for maximum visibility"""
    try:
        hourly_activity = df.groupby('Hour').size()
        
        # Find peaks - hours with higher activity than neighbors
        peak_hours = []
        for hour in range(1, 23):
            current = hourly_activity.get(hour, 0)
            prev = hourly_activity.get(hour-1, 0)
            next_ = hourly_activity.get(hour+1, 0)
            
            if current > prev and current > next_ and current > hourly_activity.mean():
                peak_hours.append(hour)
        
        return sorted(peak_hours, key=lambda x: hourly_activity.get(x, 0), reverse=True)
    except:
        return []

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
        st.subheader("📊 Top Users by Messages")
        fig = px.bar(
            x=user_msg_counts.head(15).values,
            y=user_msg_counts.head(15).index,
            orientation='h',
            title="Top 15 Users by Message Count",
            labels={'x': 'Number of Messages', 'y': 'User'},
            color=user_msg_counts.head(15).values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 User Statistics Table")
        user_stats = pd.DataFrame({
            'Messages': user_msg_counts,
            'Total Words': user_word_counts,
            'Avg Words/Message': avg_msg_len
        })
        st.dataframe(user_stats.head(15), use_container_width=True)
        
        # Additional user metrics
        st.subheader("🏆 User Rankings")
        col1, col2, col3 = st.columns(3)
        col1.metric("Most Active User", user_msg_counts.index[0])
        col2.metric("Total Messages", user_msg_counts.iloc[0])
        col3.metric("Most Wordy User", user_word_counts.index[0])

def display_time_analysis(df):
    st.header("⏰ Activity Trends & Patterns")

    # Monthly timeline
    st.subheader("📈 Monthly Activity Timeline")
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
        st.subheader("📅 Busiest Days of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['Day'].value_counts().reindex(day_order)
        
        fig = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            title="Messages by Day of Week",
            labels={'x': 'Day', 'y': 'Number of Messages'},
            color=day_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🗓️ Busiest Months")
        month_counts = df['Month'].value_counts()
        
        fig = px.bar(
            x=month_counts.index,
            y=month_counts.values,
            title="Messages by Month",
            labels={'x': 'Month', 'y': 'Number of Messages'},
            color=month_counts.values,
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Weekly Activity Heatmap
    st.subheader("🔥 Weekly Activity Heatmap")
    create_activity_heatmap(df)
    
    # Activity by time of day
    st.subheader("🌅 Activity by Time of Day")
    col1, col2 = st.columns(2)
    
    with col1:
        period_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        period_counts = df['day_period'].value_counts().reindex(period_order)
        
        fig = px.pie(
            values=period_counts.values,
            names=period_counts.index,
            title="Message Distribution by Time of Day",
            color=period_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly activity
        hour_counts = df['Hour'].value_counts().sort_index()
        fig = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            title="Activity by Hour of Day",
            labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
            color=hour_counts.values,
            color_continuous_scale='Thermal'
        )
        fig.update_xaxes(tickvals=list(range(0, 24)))
        st.plotly_chart(fig, use_container_width=True)

def create_activity_heatmap(df):
    """Creates an interactive heatmap using plotly with error handling"""
    try:
        # Create pivot table for heatmap
        activity_map = df.pivot_table(
            index='Day',
            columns='Hour',
            values='Message',
            aggfunc='count',
            fill_value=0
        )

        # Ensure all days and hours are present
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hour_range = list(range(24))
        
        # Reindex to include all days and hours
        activity_map = activity_map.reindex(day_order, fill_value=0)
        activity_map = activity_map.reindex(columns=hour_range, fill_value=0)

        # Check if we have enough data for heatmap
        if activity_map.sum().sum() < 10:
            st.warning("Not enough data to generate a meaningful heatmap. Continue chatting! 😊")
            return

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
        
    except Exception as e:
        st.warning("Could not generate heatmap: Not enough varied activity data. Continue chatting! 📱")

def display_content_analysis(df):
    st.header("📝 Content & Topic Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("☁️ Word Cloud - Most Frequent Words")
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
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title('Top 100 Most Frequent Words')
        st.pyplot(fig)

    with col2:
        st.subheader("😊 Emoji Analysis")
        emoji_df = extract_emojis(df)
        if not emoji_df.empty:
            # Top emojis bar chart
            fig = px.bar(
                emoji_df.head(15),
                x='Count',
                y='Emoji',
                orientation='h',
                title='Top 15 Most Used Emojis',
                color='Count',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Emoji stats
            total_emojis = emoji_df['Count'].sum()
            unique_emojis = len(emoji_df)
            st.metric("Total Emojis Used", f"{total_emojis:,}")
            st.metric("Unique Emojis", unique_emojis)
        else:
            st.info("No emojis found in the chat.")

def display_ml_analysis(df):
    st.header("🤖 Machine Learning Insights")
    st.subheader("📈 Linear Regression - Message Volume Prediction")
    
    try:
        # Check if we have enough data
        if len(df) < 20:
            st.info("📊 Need at least 20 messages to generate predictions")
            return
            
        with st.spinner("Training prediction model..."):
            result = predict_future_activity(df)
        
        if result[0] is not None:
            daily_counts, future_predictions, metrics = result
            r2, mse, slope, intercept = metrics
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score", f"{r2:.3f}")
            col2.metric("Trend Slope", f"{slope:.2f}")
            col3.metric("Avg Daily Messages", f"{daily_counts['message_count'].mean():.1f}")
            col4.metric("Predicted Growth", "📈" if slope > 0 else "📉")
            
            # Create visualization
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['message_count'],
                mode='markers+lines',
                name='Actual Messages',
                marker=dict(color='blue', size=6),
                line=dict(color='blue', width=1)
            ))
            
            # Regression line
            X = daily_counts['days'].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, daily_counts['message_count'])
            regression_line = model.predict(X)
            
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=regression_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=3)
            ))
            
            # Future predictions
            last_date = daily_counts['date'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines',
                name='Future Prediction',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Message Volume Trend & Prediction (Linear Regression)",
                xaxis_title="Date",
                yaxis_title="Number of Messages",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("📊 Trend Analysis")
            if slope > 1:
                st.success(f"🚀 **Strong Growth Trend**: Messages are increasing by {slope:.2f} per day on average")
            elif slope > 0.1:
                st.info(f"📈 **Moderate Growth**: Messages are slowly increasing by {slope:.2f} per day")
            elif slope > -0.1:
                st.warning("➡️ **Stable Activity**: Message volume is consistent")
            else:
                st.error(f"📉 **Declining Trend**: Messages are decreasing by {abs(slope):.2f} per day")
                
            # Future insights
            avg_future = np.mean(future_predictions)
            current_avg = daily_counts['message_count'].mean()
            change_pct = ((avg_future - current_avg) / current_avg) * 100 if current_avg > 0 else 0
            
            st.metric(
                "Predicted Avg (Next 30 days)", 
                f"{avg_future:.1f}", 
                f"{change_pct:+.1f}%"
            )
            
        else:
            st.info("📊 Not enough daily variation in data for accurate predictions. Try with a longer chat history!")
            
    except Exception as e:
        st.error(f"❌ Error in ML analysis: {str(e)}")
        st.info("This might happen with very short or irregular chat data. Try with a longer WhatsApp export.")

def display_advanced_ml_insights(df):
    st.header("🔮 Advanced ML Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Chat Health Score")
        health_score = calculate_chat_health(df)
        
        # Health meter
        st.progress(health_score/100)
        st.metric("Overall Chat Health", f"{health_score:.1f}/100")
        
        if health_score > 80:
            st.success("💚 Excellent community engagement!")
        elif health_score > 60:
            st.info("💛 Healthy chat with good activity")
        else:
            st.warning("🧡 Room for improvement in engagement")
    
    with col2:
        st.subheader("🕒 Optimal Posting Times")
        peak_hours = predict_optimal_posting_time(df)
        
        if peak_hours:
            st.write("**Best times to message:**")
            for hour in peak_hours[:3]:  # Top 3 hours
                period = "AM" if hour < 12 else "PM"
                display_hour = hour if hour <= 12 else hour - 12
                st.write(f"• {display_hour}:00 {period}")
        else:
            st.info("Post evenly throughout the day")
    
    # User engagement tiers
    st.subheader("👥 User Engagement Tiers")
    user_engagement = df['User'].value_counts()
    
    if len(user_engagement) >= 3:
        tiers = {
            'Super Active': user_engagement[user_engagement > user_engagement.quantile(0.8)].index.tolist(),
            'Regular': user_engagement[(user_engagement <= user_engagement.quantile(0.8)) & 
                                     (user_engagement > user_engagement.quantile(0.4))].index.tolist(),
            'Occasional': user_engagement[user_engagement <= user_engagement.quantile(0.4)].index.tolist()
        }
        
        for tier, users in tiers.items():
            with st.expander(f"{tier} ({len(users)} users)"):
                if users:
                    st.write(", ".join(users[:10]))  # Show first 10 users
                else:
                    st.write("No users in this tier")
    else:
        st.info("Need more users for engagement tier analysis")
    
    # Conversation analytics
    st.subheader("💬 Conversation Analytics")
    col1, col2, col3 = st.columns(3)
    
    avg_msg_length = df['Message_Length'].mean()
    media_ratio = (df['Media'].sum() / len(df) * 100) if len(df) > 0 else 0
    active_days = df['date'].dt.date.nunique()
    
    col1.metric("Avg Message Length", f"{avg_msg_length:.1f} words")
    col2.metric("Media Share", f"{media_ratio:.1f}%")
    col3.metric("Active Days", active_days)

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
            st.markdown("---")
            display_ml_analysis(temp_df)
            st.markdown("---")
            display_advanced_ml_insights(temp_df)
            
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