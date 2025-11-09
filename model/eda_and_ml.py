"""
LES MILLS YOUTUBE VIDEO ANALYSIS
Part 2: Exploratory Data Analysis & Machine Learning
Author: [Your Name]
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ============================================================================
# LOAD DATA
# ============================================================================

# Load the data we collected in Part 1
df = pd.read_csv('../data/les_mills_videos.csv')
df['published_at'] = pd.to_datetime(df['published_at'])
df['publish_date'] = pd.to_datetime(df['publish_date'])

print(f"📊 Loaded {len(df)} videos")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# 1. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*70 + "\n")

# Find Nina's video
nina_video = df[df['is_nina_video'] == True].iloc[0]

print(f"⭐ NINA'S VIDEO PERFORMANCE:")
print(f"Title: {nina_video['title']}")
print(f"Views: {nina_video['view_count']:,}")
print(f"Likes: {nina_video['like_count']:,}")
print(f"Comments: {nina_video['comment_count']:,}")
print(f"Engagement Rate: {nina_video['engagement_rate']:.2f}%")
print(f"Duration: {nina_video['duration_minutes']:.1f} minutes")
print(f"Workout Type: {nina_video['workout_type']}")
print(f"Days Since Published: {nina_video['days_since_published']}")

# Top 10 most viewed videos
print(f"\n🏆 TOP 10 MOST VIEWED VIDEOS:")
top_videos = df.nlargest(10, 'view_count')[['title', 'view_count', 'like_count', 'workout_type', 'duration_minutes']]
print(top_videos.to_string())

# Nina's rank
nina_rank = (df['view_count'] > nina_video['view_count']).sum() + 1
print(f"\n📊 Nina's video ranks #{nina_rank} out of {len(df)} videos by view count")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print(f"\n📈 Creating visualizations...")

# 1. View Distribution
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('View Count Distribution', 'Engagement Rate Distribution',
                   'Views by Workout Type', 'Duration vs Views'),
    specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
           [{'type': 'box'}, {'type': 'scatter'}]]
)

# View distribution
fig.add_trace(
    go.Histogram(x=df['view_count'], name='Views', nbinsx=50, marker_color='lightblue'),
    row=1, col=1
)
fig.add_vline(x=nina_video['view_count'], line_dash="dash", line_color="red",
              annotation_text="Nina's Video", row=1, col=1)

# Engagement distribution
fig.add_trace(
    go.Histogram(x=df['engagement_rate'], name='Engagement', nbinsx=50, marker_color='lightgreen'),
    row=1, col=2
)
fig.add_vline(x=nina_video['engagement_rate'], line_dash="dash", line_color="red",
              annotation_text="Nina's Video", row=1, col=2)

# Views by workout type
workout_stats = df.groupby('workout_type')['view_count'].mean().sort_values(ascending=False)
fig.add_trace(
    go.Box(x=df['workout_type'], y=df['view_count'], name='Views by Type', marker_color='orange'),
    row=2, col=1
)

# Duration vs Views
fig.add_trace(
    go.Scatter(x=df['duration_minutes'], y=df['view_count'], mode='markers',
              name='Videos', marker=dict(size=5, color='lightblue', opacity=0.6)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=[nina_video['duration_minutes']], y=[nina_video['view_count']],
              mode='markers', name="Nina's Video",
              marker=dict(size=15, color='red', symbol='star')),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Les Mills Video Performance Overview")
fig.show()

# 2. Time series analysis
fig = px.scatter(df, x='publish_date', y='view_count',
                 color='workout_type', size='engagement_rate',
                 hover_data=['title'],
                 title='Video Performance Over Time',
                 labels={'view_count': 'Views', 'publish_date': 'Publish Date'})

# Highlight Nina's video
nina_point = df[df['is_nina_video'] == True]
fig.add_trace(go.Scatter(
    x=nina_point['publish_date'],
    y=nina_point['view_count'],
    mode='markers',
    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
    name="Nina's Video",
    showlegend=True
))

fig.update_layout(height=600)
fig.show()

# 3. Workout type analysis
workout_summary = df.groupby('workout_type').agg({
    'view_count': ['mean', 'median', 'sum', 'count'],
    'like_count': 'mean',
    'engagement_rate': 'mean'
}).round(2)

print(f"\n📊 WORKOUT TYPE ANALYSIS:")
print(workout_summary)

fig = px.bar(df.groupby('workout_type')['view_count'].mean().sort_values(ascending=True),
             orientation='h',
             title='Average Views by Workout Type',
             labels={'value': 'Average Views', 'workout_type': 'Workout Type'})
fig.update_traces(marker_color='lightblue')
fig.show()

# ============================================================================
# 2. FEATURE ENGINEERING FOR ML
# ============================================================================

print(f"\n🔧 FEATURE ENGINEERING FOR MACHINE LEARNING")
print("="*70 + "\n")

# Create feature set
df_ml = df.copy()

# One-hot encode workout type
workout_dummies = pd.get_dummies(df_ml['workout_type'], prefix='workout')
df_ml = pd.concat([df_ml, workout_dummies], axis=1)

# One-hot encode duration category
duration_dummies = pd.get_dummies(df_ml['duration_category'], prefix='duration')
df_ml = pd.concat([df_ml, duration_dummies], axis=1)

# Day of week encoding
day_dummies = pd.get_dummies(df_ml['publish_day_of_week'], prefix='day')
df_ml = pd.concat([df_ml, day_dummies], axis=1)

# Title features
df_ml['title_length'] = df_ml['title'].str.len()
df_ml['title_word_count'] = df_ml['title'].str.split().str.len()
df_ml['title_has_number'] = df_ml['title'].str.contains(r'\d').astype(int)
df_ml['title_uppercase_ratio'] = df_ml['title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

# Select features for modeling
# feature_cols = [
#     'duration_seconds', 'duration_minutes', 'days_since_published',
#     'publish_year', 'publish_month',
#     'title_length', 'title_word_count', 'title_has_number', 'title_uppercase_ratio'
# ] + [col for col in df_ml.columns if col.startswith(('workout_', 'duration_', 'day_'))]

# Select ONLY numeric features
feature_cols = [
    'duration_seconds', 'duration_minutes', 'days_since_published',
    'publish_year', 'publish_month',
    'title_length', 'title_word_count', 'title_has_number', 'title_uppercase_ratio'
]

# Add dummy variables (these are automatically numeric)
feature_cols += [col for col in df_ml.columns if col.startswith('workout_')]
feature_cols += [col for col in df_ml.columns if col.startswith('duration_cat_')]  # Note: duration_cat_ not duration_
feature_cols += [col for col in df_ml.columns if col.startswith('day_')]

# Filter features that exist
feature_cols = [col for col in feature_cols if col in df_ml.columns]

print(f"✅ Created {len(feature_cols)} features for modeling")

# ============================================================================
# 3. CLUSTERING ANALYSIS
# ============================================================================

print(f"\n🎯 CLUSTERING ANALYSIS")
print("="*70 + "\n")

# Prepare data for clustering
cluster_features = ['view_count', 'like_count', 'comment_count',
                   'duration_minutes', 'engagement_rate', 'views_per_day']

X_cluster = df[cluster_features].fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal number of clusters using elbow method
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))
fig.update_layout(title='Elbow Method for Optimal K',
                 xaxis_title='Number of Clusters',
                 yaxis_title='Inertia')
fig.show()

# Use 4 clusters (can adjust based on elbow plot)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ Created {optimal_k} clusters")

# Nina's cluster
nina_cluster = nina_video['video_id']
nina_cluster_id = df[df['video_id'] == nina_cluster]['cluster'].values[0]
print(f"⭐ Nina's video belongs to Cluster {nina_cluster_id}")

# Analyze clusters
print(f"\n📊 CLUSTER CHARACTERISTICS:")
cluster_analysis = df.groupby('cluster').agg({
    'view_count': ['mean', 'count'],
    'like_count': 'mean',
    'engagement_rate': 'mean',
    'duration_minutes': 'mean',
    'views_per_day': 'mean'
}).round(2)
print(cluster_analysis)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# Plot clusters
fig = px.scatter(df, x='pca1', y='pca2', color='cluster',
                hover_data=['title', 'view_count'],
                title='Video Clusters (PCA Visualization)',
                labels={'pca1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                       'pca2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})

# Highlight Nina's video
nina_pca = df[df['is_nina_video'] == True]
fig.add_trace(go.Scatter(
    x=nina_pca['pca1'],
    y=nina_pca['pca2'],
    mode='markers',
    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
    name="Nina's Video",
    showlegend=True
))

fig.show()

# Videos in same cluster as Nina
same_cluster = df[df['cluster'] == nina_cluster_id].sort_values('view_count', ascending=False)
print(f"\n🎯 TOP VIDEOS IN NINA'S CLUSTER (Cluster {nina_cluster_id}):")
print(same_cluster[['title', 'view_count', 'workout_type', 'duration_minutes']].head(10))

# ============================================================================
# 4. PREDICTIVE MODELING
# ============================================================================

print(f"\n🤖 PREDICTIVE MODELING: View Count Prediction")
print("="*70 + "\n")



# Get current feature matrix
X_test_check = df_ml[feature_cols]
# Find any string/object columns
string_cols = X_test_check.select_dtypes(include=['object', 'string']).columns.tolist()

if string_cols:
    print(f"⚠️  Removing {len(string_cols)} string columns: {string_cols}")
    feature_cols = [col for col in feature_cols if col not in string_cols]
    print(f"✅ Remaining features: {len(feature_cols)}")
else:
    print(f"✅ All {len(feature_cols)} features are numeric!")

# Prepare data
X = df_ml[feature_cols]
y = df_ml['view_count']

print(f"Feature dtypes:\n{X.dtypes}")
print(f"\nNon-numeric columns: {X.select_dtypes(include=['object']).columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluation
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"📈 MODEL PERFORMANCE:")
print(f"  Training R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test MAE: {test_mae:,.0f} views")
print(f"  Test RMSE: {test_rmse:,.0f} views")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print(f"\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance.to_string())

# Plot feature importance
fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
            title='Top Features Driving Video Views',
            labels={'importance': 'Importance Score', 'feature': 'Feature'})
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
fig.show()

# Predict Nina's video performance
nina_features = df_ml[df_ml['is_nina_video'] == True][feature_cols].fillna(0)
nina_predicted_views = rf_model.predict(nina_features)[0]
nina_actual_views = nina_video['view_count']

print(f"\n⭐ NINA'S VIDEO PREDICTION:")
print(f"  Actual Views: {nina_actual_views:,}")
print(f"  Predicted Views: {nina_predicted_views:,.0f}")
print(f"  Difference: {nina_actual_views - nina_predicted_views:,.0f} ({((nina_actual_views/nina_predicted_views - 1) * 100):.1f}%)")

if nina_actual_views > nina_predicted_views:
    print(f"  💡 Nina's video OUTPERFORMED expectations!")
else:
    print(f"  💡 Nina's video has room for growth")

# ============================================================================
# 5. TITLE ANALYSIS (NLP)
# ============================================================================
# ============================================================================
# 5. TITLE ANALYSIS (NLP) - ENHANCED WITH PROPER PREPROCESSING
# ============================================================================

print(f"\n📝 TITLE ANALYSIS (NLP)")
print("=" * 70 + "\n")

# Import NLP libraries
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Extract top-performing videos
top_performers = df.nlargest(50, 'view_count')

# Custom stop words (including brand-specific words that aren't informative)
custom_stop_words = set(ENGLISH_STOP_WORDS) | {
    'les', 'mills', 'workout', 'min', 'minute', 'minutes',
    'video', 'full', 'class', 'new', 'official', 'youtube',
    'watch', 'subscribe', 'channel', 'https', 'www', 'com',
    'like', 'follow', 'join', 'amp'  # Added common social media words
}


def preprocess_text(text):
    """Clean and preprocess text for NLP analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


# Preprocess titles
top_performers['title_clean'] = top_performers['title'].apply(preprocess_text)

# Method 1: TF-IDF Analysis (best for finding important phrases)
print("🔤 METHOD 1: TF-IDF ANALYSIS")
print("-" * 70)

tfidf = TfidfVectorizer(
    max_features=25,
    stop_words=list(custom_stop_words),
    ngram_range=(1, 3),  # Include up to 3-word phrases
    min_df=2,  # Word must appear in at least 2 documents
    max_df=0.8  # Ignore words appearing in more than 80% of documents
)

tfidf_matrix = tfidf.fit_transform(top_performers['title_clean'])
feature_names = tfidf.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_df = pd.DataFrame({
    'term': feature_names,
    'tfidf_score': tfidf_scores
}).sort_values('tfidf_score', ascending=False)

print("Top 15 Most Important Terms/Phrases:")
print(tfidf_df.head(15).to_string(index=False))

# Visualize TF-IDF results
fig1 = px.bar(
    tfidf_df.head(15),
    x='tfidf_score',
    y='term',
    orientation='h',
    title='Most Important Terms in Top Video Titles (TF-IDF)',
    labels={'tfidf_score': 'TF-IDF Score', 'term': 'Term'},
    color='tfidf_score',
    color_continuous_scale='reds'
)
fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig1.show()

# Method 2: Word Frequency (after removing stop words)
print("\n\n🔤 METHOD 2: WORD FREQUENCY ANALYSIS")
print("-" * 70)

# Tokenize and filter
all_words = []
for title in top_performers['title_clean']:
    words = title.split()
    # Filter out stop words and short words
    filtered_words = [
        word for word in words
        if word not in custom_stop_words and len(word) > 3
    ]
    all_words.extend(filtered_words)

# Count frequencies
word_freq = Counter(all_words)
common_words_df = pd.DataFrame(
    word_freq.most_common(20),
    columns=['word', 'frequency']
)

print("Top 20 Most Frequent Meaningful Words:")
print(common_words_df.to_string(index=False))

# Visualize word frequency
fig2 = px.bar(
    common_words_df.head(15),
    x='frequency',
    y='word',
    orientation='h',
    title='Most Frequent Meaningful Words in Top Video Titles',
    labels={'frequency': 'Frequency', 'word': 'Word'},
    color='frequency',
    color_continuous_scale='blues'
)
fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig2.show()

# Method 3: Bigram Analysis (2-word phrases)
print("\n\n🔤 METHOD 3: BIGRAM (2-WORD PHRASE) ANALYSIS")
print("-" * 70)

from sklearn.feature_extraction.text import CountVectorizer

bigram_vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words=list(custom_stop_words),
    min_df=2,
    max_features=20
)

bigram_matrix = bigram_vectorizer.fit_transform(top_performers['title_clean'])
bigram_freq = bigram_matrix.sum(axis=0).A1
bigram_names = bigram_vectorizer.get_feature_names_out()

bigrams_df = pd.DataFrame({
    'bigram': bigram_names,
    'frequency': bigram_freq
}).sort_values('frequency', ascending=False)

print("Top 15 Most Common 2-Word Phrases:")
print(bigrams_df.head(15).to_string(index=False))

# Visualize bigrams
fig3 = px.bar(
    bigrams_df.head(15),
    x='frequency',
    y='bigram',
    orientation='h',
    title='Most Common 2-Word Phrases in Top Video Titles',
    labels={'frequency': 'Frequency', 'bigram': 'Phrase'},
    color='frequency',
    color_continuous_scale='greens'
)
fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
fig3.show()

# Method 4: Workout Type specific keywords
print("\n\n🔤 METHOD 4: WORKOUT-SPECIFIC KEYWORDS")
print("-" * 70)

workout_keywords = {}
for workout_type in top_performers['workout_type'].unique():
    workout_titles = top_performers[
        top_performers['workout_type'] == workout_type
        ]['title_clean']

    if len(workout_titles) > 0:
        # Get words for this workout type
        words = []
        for title in workout_titles:
            title_words = [
                w for w in title.split()
                if w not in custom_stop_words and len(w) > 3
            ]
            words.extend(title_words)

        # Get top 5 words
        word_freq = Counter(words)
        top_5 = [word for word, count in word_freq.most_common(5)]
        workout_keywords[workout_type] = top_5

print("Top Keywords by Workout Type:")
for workout, keywords in sorted(workout_keywords.items()):
    if keywords:
        print(f"  {workout:15s}: {', '.join(keywords)}")

# Summary insights
print("\n\n💡 KEY INSIGHTS FROM TITLE ANALYSIS:")
print("-" * 70)

top_terms = tfidf_df.head(5)['term'].tolist()
top_words = common_words_df.head(5)['word'].tolist()
top_bigrams = bigrams_df.head(3)['bigram'].tolist()

print(f"1. Most Important Terms: {', '.join(top_terms)}")
print(f"2. Most Frequent Words: {', '.join(top_words)}")
print(f"3. Common Phrases: {', '.join(top_bigrams)}")
print(f"\n4. Title Optimization Recommendations:")
print(f"   - Include these high-impact terms: {', '.join(top_terms[:3])}")
print(f"   - Use phrases like: {', '.join(top_bigrams[:2])}")
print(f"   - Workout-specific keywords drive engagement")
# ============================================================================
# 6. KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================

print(f"\n💡 KEY INSIGHTS & RECOMMENDATIONS")
print("="*70 + "\n")

# Calculate insights
avg_views = df['view_count'].mean()
median_views = df['view_count'].median()
top_10_avg = df.nlargest(10, 'view_count')['view_count'].mean()

best_workout_type = df.groupby('workout_type')['view_count'].mean().idxmax()
best_duration = df.groupby('duration_category')['view_count'].mean().idxmax()
best_day = df.groupby('publish_day_of_week')['view_count'].mean().idxmax()

print(f"📊 CHANNEL BENCHMARKS:")
print(f"  Average Views: {avg_views:,.0f}")
print(f"  Median Views: {median_views:,.0f}")
print(f"  Top 10 Average: {top_10_avg:,.0f}")

print(f"\n🏆 SUCCESS FACTORS:")
print(f"  Best Workout Type: {best_workout_type}")
print(f"  Optimal Duration: {best_duration}")
print(f"  Best Publishing Day: {best_day}")

print(f"\n⭐ NINA'S VIDEO ANALYSIS:")
print(f"  Performance: {((nina_actual_views/avg_views - 1) * 100):.1f}% vs. channel average")
print(f"  Cluster: Similar to other {'high' if nina_cluster_id in df.nlargest(len(df)//4, 'view_count')['cluster'].values else 'moderate'}-performing videos")

print(f"\n💼 RECOMMENDATIONS FOR LES MILLS:")
print(f"  1. Workout Type: Focus on {best_workout_type} content (highest avg views)")
print(f"  2. Video Length: {best_duration} performs best")
print(f"  3. Publishing: {best_day} is optimal for releases")
print(f"  4. Title Strategy: Include keywords like: {', '.join(tfidf_df.head(5)['word'].values)}")
print(f"  5. Nina's Format: Consider similar {nina_video['workout_type']} content at {nina_video['duration_minutes']:.0f} minutes")

print(f"\n" + "="*70)
print(f"✅ ANALYSIS COMPLETE!")
print(f"="*70)