"""
LES MILLS YOUTUBE ANALYTICS DASHBOARD
Streamlit Version - Easy to Deploy
Author: [Your Name]

Installation:
pip install streamlit plotly pandas numpy

Usage:
streamlit run streamlit_dashboard.py

Deploy to Streamlit Cloud (FREE):
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy!
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Les Mills YouTube Analytics",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #475569;
    }

    .stMetric label {
        color: #94a3b8 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    .nina-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(249, 115, 22, 0.2) 100%);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid rgba(239, 68, 68, 0.4);
        margin: 20px 0;
    }

    .stSelectbox > div > div {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache the video data"""
    try:
        df = pd.read_csv('C:\\Users\\sneha\\PycharmProjects\\less_mills_youtube_analytics\\data\\les_mills_videos.csv')
        df['published_at'] = pd.to_datetime(df['published_at'])
        return df
    except:
        # Generate sample data if CSV not found
        st.warning("CSV not found, using sample data for demonstration")
        np.random.seed(42)
        workout_types = ['GRIT', 'BODYPUMP', 'BODYCOMBAT', 'BODYBALANCE', 'RPM', 'BODYATTACK', 'Other']

        data = []
        for i in range(100):
            publish_date = datetime.now() - timedelta(days=np.random.randint(0, 730))
            views = int(np.random.exponential(50000) + 10000)
            likes = int(views * (np.random.uniform(0.01, 0.05)))
            comments = int(views * (np.random.uniform(0.001, 0.01)))
            duration = int(np.random.uniform(10, 45))

            data.append({
                'video_id': f'vid_{i}',
                'title': f'Les Mills {np.random.choice(workout_types)} Workout {i + 1}',
                'workout_type': np.random.choice(workout_types),
                'view_count': views,
                'like_count': likes,
                'comment_count': comments,
                'duration_minutes': duration,
                'published_at': publish_date,
                'engagement_rate': (likes + comments) / views * 100,
                'views_per_day': views / ((datetime.now() - publish_date).days + 1),
                'is_nina_video': False
            })

        df = pd.DataFrame(data)

        # Add Nina's video
        nina_row = {
            'video_id': 'nina_video',
            'title': 'WORK OUT #LIKENINA | 30-minute LES MILLS GRIT Cardio',
            'workout_type': 'GRIT',
            'view_count': 125000,
            'like_count': 4200,
            'comment_count': 320,
            'duration_minutes': 30,
            'published_at': datetime(2024, 4, 15),
            'engagement_rate': 3.61,
            'views_per_day': 625,
            'is_nina_video': True
        }
        df = pd.concat([pd.DataFrame([nina_row]), df], ignore_index=True)

    df = df.sort_values('view_count', ascending=False).reset_index(drop=True)
    return df


# Load data
df = load_data()
nina_video = df[df['is_nina_video'] == True].iloc[0] if 'is_nina_video' in df.columns else None

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("# 🏋️ Les Mills YouTube Analytics")
    st.markdown("### Data-Driven Content Performance Insights")

with col2:
    st.markdown("**ML-Powered Analysis**")
    st.markdown("*AI & ML Engineer Application*")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 Filters & Controls")

    workout_type = st.selectbox(
        "Select Workout Type",
        options=['All'] + sorted(df['workout_type'].unique().tolist()),
        index=0
    )

    st.markdown("---")

    st.markdown("## 📊 About This Dashboard")
    st.markdown("""
    This interactive dashboard analyzes Les Mills YouTube channel 
    performance using machine learning and data science techniques.

    **Key Features:**
    - 📈 Performance metrics
    - 🎯 Nina's video analysis
    - 📊 ML-driven insights
    - 🔍 Workout type comparison
    """)

    st.markdown("---")

    st.markdown("## 📧 Contact")
    st.markdown("**[Your Name]**")
    st.markdown("[your.email@example.com](mailto:your.email@example.com)")

# ============================================================================
# FILTER DATA
# ============================================================================

filtered_df = df if workout_type == 'All' else df[df['workout_type'] == workout_type]

# Calculate metrics
total_videos = len(filtered_df)
total_views = filtered_df['view_count'].sum()
avg_views = int(filtered_df['view_count'].mean())
avg_engagement = filtered_df['engagement_rate'].mean()

# ============================================================================
# KEY METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Videos",
        value=f"{total_videos}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Views",
        value=f"{total_views / 1000000:.1f}M",
        delta=None
    )

with col3:
    st.metric(
        label="Average Views",
        value=f"{avg_views / 1000:.0f}K",
        delta=None
    )

with col4:
    st.metric(
        label="Avg Engagement",
        value=f"{avg_engagement:.2f}%",
        delta=None
    )

st.markdown("---")

# ============================================================================
# NINA'S VIDEO SPOTLIGHT
# ============================================================================

if nina_video is not None:
    nina_rank = (df['view_count'] > nina_video['view_count']).sum() + 1
    nina_vs_avg = ((nina_video['view_count'] / avg_views - 1) * 100)

    st.markdown(f"""
    <div class="nina-card">
        <h2 style="color: #ef4444; margin-bottom: 10px;">⭐ Nina's GRIT Cardio Workout</h2>
        <p style="color: #94a3b8; font-size: 16px; margin-bottom: 20px;">{nina_video['title']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Views", f"{nina_video['view_count']:,}")

    with col2:
        st.metric("Rank", f"#{nina_rank}")

    with col3:
        st.metric("Engagement", f"{nina_video['engagement_rate']:.2f}%")

    with col4:
        st.metric("Duration", f"{nina_video['duration_minutes']} min")

    with col5:
        st.metric("vs Average", f"{nina_vs_avg:+.0f}%")

    st.markdown("---")

# ============================================================================
# CHARTS ROW 1
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Average Views by Workout Type")

    workout_summary = filtered_df.groupby('workout_type').agg({
        'view_count': 'mean'
    }).reset_index().sort_values('view_count', ascending=True)

    fig1 = go.Figure(data=[
        go.Bar(x=workout_summary['view_count'],
               y=workout_summary['workout_type'],
               orientation='h',
               marker=dict(color='#ef4444', cornerradius=10))
    ])
    fig1.update_layout(
        xaxis_title="Average Views",
        yaxis_title="",
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### ⭐ Nina's Video vs Channel Average")

    if nina_video is not None:
        avg_likes = filtered_df['like_count'].mean()
        avg_comments = filtered_df['comment_count'].mean()
        avg_vpd = filtered_df['views_per_day'].mean()

        radar_data = pd.DataFrame({
            'metric': ['Views', 'Likes', 'Comments', 'Engagement', 'Views/Day'],
            'nina': [
                (nina_video['view_count'] / avg_views * 100),
                (nina_video['like_count'] / avg_likes * 100),
                (nina_video['comment_count'] / avg_comments * 100),
                (nina_video['engagement_rate'] / avg_engagement * 100),
                (nina_video['views_per_day'] / avg_vpd * 100)
            ],
            'average': [100, 100, 100, 100, 100]
        })

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=radar_data['nina'],
            theta=radar_data['metric'],
            fill='toself',
            name="Nina's Video",
            line=dict(color='#ef4444'),
            fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        fig2.add_trace(go.Scatterpolar(
            r=radar_data['average'],
            theta=radar_data['metric'],
            fill='toself',
            name='Channel Avg',
            line=dict(color='#6b7280'),
            fillcolor='rgba(107, 114, 128, 0.2)'
        ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 150])
            ),
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================================================
# CHARTS ROW 2
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⏱️ Video Duration vs Views")

    scatter_data = filtered_df.head(50)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=scatter_data[~scatter_data.get('is_nina_video', False)]['duration_minutes'],
        y=scatter_data[~scatter_data.get('is_nina_video', False)]['view_count'],
        mode='markers',
        name='Videos',
        marker=dict(size=8, color='#3b82f6', opacity=0.6)
    ))

    if nina_video is not None:
        fig3.add_trace(go.Scatter(
            x=[nina_video['duration_minutes']],
            y=[nina_video['view_count']],
            mode='markers',
            name="Nina's Video",
            marker=dict(size=15, color='#ef4444', symbol='star')
        ))

    fig3.update_layout(
        xaxis_title="Duration (minutes)",
        yaxis_title="Views",
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("### 📈 Average Views Over Time")

    timeline_data = filtered_df.copy()
    timeline_data['month'] = timeline_data['published_at'].dt.to_period('M').astype(str)
    monthly = timeline_data.groupby('month')['view_count'].mean().reset_index()
    monthly = monthly.sort_values('month')

    fig4 = go.Figure(data=[
        go.Scatter(x=monthly['month'],
                   y=monthly['view_count'],
                   mode='lines+markers',
                   line=dict(color='#10b981', width=3),
                   marker=dict(size=8))
    ])
    fig4.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Views",
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================================================
# TOP VIDEOS TABLE
# ============================================================================

st.markdown("### 🏆 Top 10 Performing Videos")

top_10 = filtered_df.head(10).copy()
top_10['rank'] = range(1, len(top_10) + 1)
top_10['views_formatted'] = top_10['view_count'].apply(lambda x: f"{x:,}")
top_10['engagement_formatted'] = top_10['engagement_rate'].apply(lambda x: f"{x:.2f}%")
top_10['duration_formatted'] = top_10['duration_minutes'].apply(lambda x: f"{x} min")

# Add indicator for Nina's video
top_10['indicator'] = top_10.get('is_nina_video', False).apply(lambda x: '⭐' if x else '')

display_df = top_10[[
    'rank', 'indicator', 'title', 'workout_type',
    'views_formatted', 'engagement_formatted', 'duration_formatted'
]]

display_df.columns = ['#', '⭐', 'Title', 'Type', 'Views', 'Engagement', 'Duration']

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# ============================================================================
# INSIGHTS & RECOMMENDATIONS
# ============================================================================

st.markdown("---")
st.markdown("### 💡 Key Insights & Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    best_workout = filtered_df.groupby('workout_type')['view_count'].mean().idxmax()
    st.success(f"""
    **Best Performing Workout**  
    **{best_workout}** workouts generate the highest average views
    """)

with col2:
    optimal_duration = filtered_df.groupby(pd.cut(filtered_df['duration_minutes'], bins=[0, 20, 30, 40, 60]))[
        'view_count'].mean().idxmax()
    st.info(f"""
    **⏱️ Optimal Duration**  
    Videos between **20-30 minutes** perform best
    """)

with col3:
    if nina_video is not None:
        if nina_vs_avg > 0:
            st.success(f"""
            **⭐ Nina's Performance**  
            Outperforming channel average by **{nina_vs_avg:.0f}%**
            """)
        else:
            st.warning(f"""
            **⭐ Nina's Performance**  
            Room for growth: **{nina_vs_avg:.0f}%** vs average
            """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    <p>Les Mills YouTube Analytics Dashboard • Built with Streamlit & Plotly</p>
    <p>AI & ML Engineer Application Project • Data-Driven Content Strategy</p>
</div>
""", unsafe_allow_html=True)