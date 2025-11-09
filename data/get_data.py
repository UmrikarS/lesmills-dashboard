"""
LES MILLS YOUTUBE VIDEO ANALYSIS
Part 1: Data Collection & Initial Setup
Author: [Your Name]
Purpose: Analyze Les Mills YouTube performance for job application
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
import time
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from dotenv import load_dotenv
#
# load_dotenv() # This loads the variables from the .env file

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Libraries imported successfully!")

# ============================================================================
# YOUTUBE API SETUP
# ============================================================================

"""
Steps:
1. Create a new project
2. Enable YouTube Data API v3
3. Create credentials (API key)
4. Copy the key below
"""
API_KEY = os.getenv('YOUTUBE_API_KEY')
# Les Mills Channel ID
LES_MILLS_CHANNEL_ID =  os.environ.get("LES_MILLS_CHANNEL_ID")
# Nina's favorite video ID
NINA_VIDEO_ID = os.environ.get("NINA_VIDEO_ID")

print(f"🎯 Target Channel: Les Mills International")
print(f"⭐ Reference Video: Nina's GRIT Cardio Workout")

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def get_channel_videos(channel_id, api_key, max_results=500):
    """
    Fetch all videos from a YouTube channel

    Args:
        channel_id: YouTube channel ID
        api_key: YouTube API key
        max_results: Maximum number of videos to fetch

    Returns:
        List of video IDs
    """
    base_url = "https://www.googleapis.com/youtube/v3/search"
    video_ids = []
    next_page_token = None

    print(f"📡 Fetching videos from channel...")

    while len(video_ids) < max_results:
        params = {
            'key': api_key,
            'channelId': channel_id,
            'part': 'id',
            'order': 'date',
            'maxResults': 50,
            'type': 'video',
            'pageToken': next_page_token
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract video IDs
            for item in data.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    video_ids.append(item['id']['videoId'])

            next_page_token = data.get('nextPageToken')

            print(f"  ✓ Collected {len(video_ids)} videos so far...")

            if not next_page_token:
                break

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    print(f"✅ Total videos collected: {len(video_ids)}")
    return video_ids


def get_video_details(video_ids, api_key, batch_size=50):
    """
    Get detailed statistics for videos

    Args:
        video_ids: List of video IDs
        api_key: YouTube API key
        batch_size: Number of videos per API call

    Returns:
        DataFrame with video details
    """
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    all_video_data = []

    print(f"📊 Fetching video details for {len(video_ids)} videos...")

    # Process in batches
    for i in range(0, len(video_ids), batch_size):
        batch = video_ids[i:i + batch_size]
        video_ids_str = ','.join(batch)

        params = {
            'key': api_key,
            'id': video_ids_str,
            'part': 'snippet,statistics,contentDetails'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            for item in data.get('items', []):
                video_info = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_title': item['snippet']['channelTitle'],
                    'tags': ','.join(item['snippet'].get('tags', [])),
                    'duration': item['contentDetails']['duration'],
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url']
                }
                all_video_data.append(video_info)

            print(f"  ✓ Processed {min(i + batch_size, len(video_ids))}/{len(video_ids)} videos")
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            continue

    df = pd.DataFrame(all_video_data)
    print(f"✅ Collected details for {len(df)} videos")

    return df


def get_video_comments(video_id, api_key, max_comments=100):
    """
    Fetch comments for a specific video

    Args:
        video_id: YouTube video ID
        api_key: YouTube API key
        max_comments: Maximum comments to fetch

    Returns:
        List of comments
    """
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        params = {
            'key': api_key,
            'videoId': video_id,
            'part': 'snippet',
            'maxResults': 100,
            'order': 'relevance',
            'pageToken': next_page_token
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            for item in data.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })

            next_page_token = data.get('nextPageToken')

            if not next_page_token:
                break

            time.sleep(0.5)

        except Exception as e:
            # Comments might be disabled
            break

    return comments


def parse_duration(duration_str):
    """Convert ISO 8601 duration to seconds"""
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)

    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds


# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

print("\n" + "=" * 70)
print("🚀 STARTING DATA COLLECTION")
print("=" * 70 + "\n")

# Step 1: Get all video IDs
video_ids = get_channel_videos(LES_MILLS_CHANNEL_ID, API_KEY, max_results=500)

# Make sure Nina's video is included
if NINA_VIDEO_ID not in video_ids:
    video_ids.insert(0, NINA_VIDEO_ID)
    print(f"⭐ Added Nina's video to the list")

# Step 2: Get video details
df_videos = get_video_details(video_ids, API_KEY)

# Step 3: Data preprocessing
print(f"\n📊 Processing video data...")

# Convert published date
df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
df_videos['publish_date'] = df_videos['published_at'].dt.date
df_videos['publish_year'] = df_videos['published_at'].dt.year
df_videos['publish_month'] = df_videos['published_at'].dt.month
df_videos['publish_day_of_week'] = df_videos['published_at'].dt.day_name()

# Parse duration to seconds
df_videos['duration_seconds'] = df_videos['duration'].apply(parse_duration)
df_videos['duration_minutes'] = df_videos['duration_seconds'] / 60

# Calculate days since published
df_videos['published_at'] = pd.to_datetime(df_videos['published_at']).dt.tz_localize(None)
df_videos['days_since_published'] = (datetime.now() - df_videos['published_at']).dt.days
#df_videos['days_since_published'] = (datetime.now() - df_videos['published_at']).dt.days

# Calculate engagement metrics
df_videos['engagement_rate'] = (
                                       (df_videos['like_count'] + df_videos['comment_count']) /
                                       df_videos['view_count'].replace(0, 1)
                               ) * 100

df_videos['views_per_day'] = df_videos['view_count'] / df_videos['days_since_published'].replace(0, 1)
df_videos['like_to_view_ratio'] = df_videos['like_count'] / df_videos['view_count'].replace(0, 1)


# Categorize video length
def categorize_duration(minutes):
    if minutes < 15:
        return 'Short (< 15 min)'
    elif minutes < 30:
        return 'Medium (15-30 min)'
    elif minutes < 45:
        return 'Long (30-45 min)'
    else:
        return 'Extra Long (45+ min)'


df_videos['duration_category'] = df_videos['duration_minutes'].apply(categorize_duration)


# Extract workout type from title
def extract_workout_type(title):
    title_upper = title.upper()
    workout_types = ['GRIT', 'BODYPUMP', 'BODYCOMBAT', 'BODYBALANCE',
                     'RPM', 'SPRINT', 'BODYATTACK', 'BODYSTEP',
                     'CXWORX', 'BARRE', 'TONE', 'SH\'BAM']

    for workout in workout_types:
        if workout in title_upper:
            return workout
    return 'Other'


df_videos['workout_type'] = df_videos['title'].apply(extract_workout_type)

# Flag Nina's video
df_videos['is_nina_video'] = df_videos['video_id'] == NINA_VIDEO_ID

print(f"✅ Data preprocessing complete!")

# Display basic statistics
print(f"\n📈 DATASET OVERVIEW")
print(f"=" * 70)
print(f"Total videos: {len(df_videos)}")
print(f"Date range: {df_videos['publish_date'].min()} to {df_videos['publish_date'].max()}")
print(f"Total views: {df_videos['view_count'].sum():,}")
print(f"Total likes: {df_videos['like_count'].sum():,}")
print(f"Total comments: {df_videos['comment_count'].sum():,}")

# Save to CSV
df_videos.to_csv('./les_mills_videos.csv', index=False)
print(f"\n💾 Data saved to 'les_mills_videos.csv'")

print("\n" + "=" * 70)
print("✅ DATA COLLECTION COMPLETE!")
print("=" * 70)

# Display sample
print(f"\n📋 Sample of collected data:")
print(df_videos[['title', 'view_count', 'like_count', 'duration_minutes', 'workout_type']].head(10))

import sqlite3

print("\n" + "="*70)
print("💾 SAVING TO SQL DATABASE")
print("="*70 + "\n")

# ============================================================================
# CREATE SQLITE DATABASE
# ============================================================================

# Create database connection
conn = sqlite3.connect('les_mills_videos.db')
cursor = conn.cursor()

# Save main videos table
df_videos.to_sql('videos', conn, if_exists='replace', index=False)
print(f"✅ Saved {len(df_videos)} videos to SQLite database")

# ============================================================================
# SQL QUERIES - Demonstrate SQL Proficiency
# ============================================================================

print("\n📊 RUNNING SQL ANALYTICS QUERIES\n")

# Query 1: Workout type performance
query1 = """
SELECT 
    workout_type,
    COUNT(*) as video_count,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(AVG(engagement_rate), 2) as avg_engagement,
    MAX(view_count) as max_views
FROM videos
GROUP BY workout_type
ORDER BY avg_views DESC
"""

print("Query 1: Workout Type Performance")
print("-" * 70)
result1 = pd.read_sql(query1, conn)
print(result1.to_string(index=False))

# Query 2: Top performers by year
query2 = """
SELECT 
    publish_year,
    title,
    view_count,
    workout_type,
    ROUND(engagement_rate, 2) as engagement
FROM videos
WHERE publish_year >= 2022
ORDER BY view_count DESC
LIMIT 10
"""

print("\n\nQuery 2: Top 10 Videos Since 2022")
print("-" * 70)
result2 = pd.read_sql(query2, conn)
print(result2.to_string(index=False))

# Query 3: Monthly aggregation (time series)
query3 = """
SELECT 
    publish_year,
    publish_month,
    COUNT(*) as videos_published,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(SUM(view_count), 0) as total_views
FROM videos
GROUP BY publish_year, publish_month
ORDER BY publish_year DESC, publish_month DESC
LIMIT 12
"""

print("\n\nQuery 3: Monthly Publishing Trends (Last 12 Months)")
print("-" * 70)
result3 = pd.read_sql(query3, conn)
print(result3.to_string(index=False))

# Query 4: Nina's video comparison (window function)
query4 = """
WITH ranked_videos AS (
    SELECT 
        title,
        view_count,
        engagement_rate,
        is_nina_video,
        RANK() OVER (ORDER BY view_count DESC) as view_rank,
        ROUND(AVG(view_count) OVER (), 0) as channel_avg_views
    FROM videos
)
SELECT 
    title,
    view_count,
    engagement_rate,
    view_rank,
    channel_avg_views,
    ROUND(((view_count * 1.0 / channel_avg_views - 1) * 100), 1) as vs_avg_percent
FROM ranked_videos
WHERE is_nina_video = 1 OR view_rank <= 10
ORDER BY view_rank
"""

print("\n\nQuery 4: Nina's Video vs Top Performers (Window Functions)")
print("-" * 70)
result4 = pd.read_sql(query4, conn)
print(result4.to_string(index=False))

# Query 5: Duration analysis with CASE statements
query5 = """
SELECT 
    CASE 
        WHEN duration_minutes < 15 THEN 'Short (<15 min)'
        WHEN duration_minutes < 30 THEN 'Medium (15-30 min)'
        WHEN duration_minutes < 45 THEN 'Long (30-45 min)'
        ELSE 'Extra Long (45+ min)'
    END as duration_bucket,
    COUNT(*) as video_count,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(AVG(engagement_rate), 2) as avg_engagement
FROM videos
GROUP BY duration_bucket
ORDER BY avg_views DESC
"""

print("\n\nQuery 5: Duration Analysis (CASE Statements)")
print("-" * 70)
result5 = pd.read_sql(query5, conn)
print(result5.to_string(index=False))

# ============================================================================
# CREATE AGGREGATED TABLES (Data Warehouse Pattern)
# ============================================================================

print("\n\n💾 CREATING AGGREGATED TABLES (Data Warehouse Pattern)\n")

# Create workout summary table
cursor.execute("""
CREATE TABLE IF NOT EXISTS workout_summary AS
SELECT 
    workout_type,
    COUNT(*) as total_videos,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(AVG(like_count), 0) as avg_likes,
    ROUND(AVG(engagement_rate), 2) as avg_engagement,
    ROUND(SUM(view_count), 0) as total_views,
    MIN(view_count) as min_views,
    MAX(view_count) as max_views
FROM videos
GROUP BY workout_type
""")

print("✅ Created table: workout_summary")

# Create monthly trends table
cursor.execute("""
CREATE TABLE IF NOT EXISTS monthly_trends AS
SELECT 
    publish_year,
    publish_month,
    COUNT(*) as videos_published,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(SUM(view_count), 0) as total_views,
    ROUND(AVG(engagement_rate), 2) as avg_engagement
FROM videos
GROUP BY publish_year, publish_month
ORDER BY publish_year, publish_month
""")

print("✅ Created table: monthly_trends")

# Create top performers table
cursor.execute("""
CREATE TABLE IF NOT EXISTS top_performers AS
SELECT 
    video_id,
    title,
    workout_type,
    view_count,
    like_count,
    engagement_rate,
    duration_minutes,
    RANK() OVER (ORDER BY view_count DESC) as overall_rank,
    RANK() OVER (PARTITION BY workout_type ORDER BY view_count DESC) as type_rank
FROM videos
WHERE view_count > (SELECT AVG(view_count) FROM videos)
""")

print("✅ Created table: top_performers")

conn.commit()

# Show table schema
print("\n\n📋 DATABASE SCHEMA\n")
cursor.execute("""
SELECT name, sql 
FROM sqlite_master 
WHERE type='table' AND name NOT LIKE 'sqlite_%'
ORDER BY name
""")

tables = cursor.fetchall()
for table_name, create_sql in tables:
    print(f"Table: {table_name}")
    print("-" * 70)
    print(create_sql)
    print("\n")

# ============================================================================
# SIMULATE SNOWFLAKE/CLOUD DATA WAREHOUSE QUERIES
# ============================================================================

print("=" * 70)
print("☁️  PRODUCTION DATA WAREHOUSE QUERIES (Snowflake/BigQuery Style)")
print("=" * 70)
print("""
-- These queries demonstrate cloud data warehouse patterns
-- In production, these would run on Snowflake, BigQuery, or Redshift

-- Example 1: Incremental Load Pattern (DBT style)
-- ============================================
CREATE OR REPLACE TABLE youtube_analytics.videos_daily AS
SELECT 
    video_id,
    title,
    workout_type,
    view_count,
    engagement_rate,
    CURRENT_TIMESTAMP() as updated_at,
    DATE(published_at) as publish_date
FROM raw_data.youtube_videos
WHERE updated_at >= CURRENT_DATE() - INTERVAL '1 DAY';

-- Example 2: Slowly Changing Dimension (SCD Type 2)
-- ================================================
CREATE OR REPLACE TABLE youtube_analytics.videos_history AS
SELECT 
    video_id,
    title,
    view_count,
    engagement_rate,
    valid_from,
    COALESCE(LEAD(valid_from) OVER (
        PARTITION BY video_id 
        ORDER BY valid_from
    ), '9999-12-31') as valid_to,
    CASE 
        WHEN LEAD(valid_from) OVER (
            PARTITION BY video_id 
            ORDER BY valid_from
        ) IS NULL THEN TRUE 
        ELSE FALSE 
    END as is_current
FROM youtube_snapshots;

-- Example 3: Materialized View for Performance
-- ============================================
CREATE MATERIALIZED VIEW youtube_analytics.workout_performance_mv AS
SELECT 
    workout_type,
    DATE_TRUNC('month', published_at) as month,
    COUNT(*) as video_count,
    AVG(view_count) as avg_views,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY view_count
    ) as median_views,
    SUM(view_count) as total_views
FROM youtube_analytics.videos
GROUP BY workout_type, month;

-- Example 4: CTEs and Complex Analytics
-- ====================================
WITH video_metrics AS (
    SELECT 
        video_id,
        view_count,
        engagement_rate,
        AVG(view_count) OVER (
            PARTITION BY workout_type
        ) as workout_avg_views,
        STDDEV(view_count) OVER (
            PARTITION BY workout_type
        ) as workout_stddev
    FROM videos
),
z_scores AS (
    SELECT 
        *,
        (view_count - workout_avg_views) / 
        NULLIF(workout_stddev, 0) as z_score
    FROM video_metrics
)
SELECT * FROM z_scores WHERE ABS(z_score) > 2;

""")

print("\n✅ SQL demonstrations complete!")
print(f"💾 Database saved to: les_mills_videos.db")
print(f"📊 Tables created: videos, workout_summary, monthly_trends, top_performers")

# Close connection
conn.close()

print("\n" + "="*70)
print("✅ SQL COMPONENT COMPLETE")
print("="*70)

# ============================================================================
# SAVE SQL QUERIES TO FILE FOR DOCUMENTATION
# ============================================================================

with open('sql_queries.sql', 'w') as f:
    f.write("""
-- LES MILLS YOUTUBE ANALYTICS SQL QUERIES
-- Demonstrates SQL proficiency for job application
-- Author: [Your Name]

-- ============================================
-- QUERY 1: Workout Type Performance Analysis
-- ============================================
SELECT 
    workout_type,
    COUNT(*) as video_count,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(AVG(engagement_rate), 2) as avg_engagement,
    MAX(view_count) as max_views,
    MIN(view_count) as min_views
FROM videos
GROUP BY workout_type
ORDER BY avg_views DESC;

-- ============================================
-- QUERY 2: Top Performers with Ranking
-- ============================================
SELECT 
    title,
    workout_type,
    view_count,
    engagement_rate,
    RANK() OVER (ORDER BY view_count DESC) as overall_rank,
    RANK() OVER (PARTITION BY workout_type ORDER BY view_count DESC) as type_rank
FROM videos
ORDER BY view_count DESC
LIMIT 20;

-- ============================================
-- QUERY 3: Time Series Analysis (Monthly)
-- ============================================
SELECT 
    strftime('%Y-%m', published_at) as month,
    COUNT(*) as videos_published,
    ROUND(AVG(view_count), 0) as avg_views,
    ROUND(SUM(view_count), 0) as total_views,
    ROUND(AVG(engagement_rate), 2) as avg_engagement
FROM videos
GROUP BY month
ORDER BY month DESC;

-- ============================================
-- QUERY 4: Cohort Analysis by Duration
-- ============================================
SELECT 
    CASE 
        WHEN duration_minutes < 15 THEN 'Short'
        WHEN duration_minutes < 30 THEN 'Medium'
        WHEN duration_minutes < 45 THEN 'Long'
        ELSE 'Extra Long'
    END as duration_category,
    workout_type,
    COUNT(*) as video_count,
    ROUND(AVG(view_count), 0) as avg_views
FROM videos
GROUP BY duration_category, workout_type
ORDER BY avg_views DESC;

-- ============================================
-- QUERY 5: Nina's Video Performance Context
-- ============================================
WITH channel_stats AS (
    SELECT 
        AVG(view_count) as avg_views,
        STDDEV(view_count) as stddev_views
    FROM videos
),
video_with_stats AS (
    SELECT 
        v.*,
        c.avg_views,
        c.stddev_views,
        (v.view_count - c.avg_views) / c.stddev_views as z_score
    FROM videos v
    CROSS JOIN channel_stats c
)
SELECT 
    title,
    view_count,
    ROUND(avg_views, 0) as channel_avg,
    ROUND(z_score, 2) as z_score,
    CASE 
        WHEN z_score > 2 THEN 'Exceptional'
        WHEN z_score > 1 THEN 'Above Average'
        WHEN z_score > -1 THEN 'Average'
        ELSE 'Below Average'
    END as performance_tier
FROM video_with_stats
WHERE is_nina_video = 1 OR z_score > 1
ORDER BY view_count DESC;
""")

print("\n💾 SQL queries saved to: sql_queries.sql")
print("   Use this file to demonstrate SQL expertise!\n")