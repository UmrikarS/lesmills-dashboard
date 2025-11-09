"""
LES MILLS YOUTUBE ANALYTICS DASHBOARD
Professional Interactive Dashboard using Plotly Dash
Author: [Your Name]

Installation:
pip install dash plotly pandas numpy

Usage:
python dashboard.py
Then open browser to http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# LOAD YOUR DATA HERE
# ============================================================================

def load_data():
    """
    Load your CSV data here
    Replace this with: df = pd.read_csv('les_mills_videos.csv')
    """
    # This is sample data - replace with your actual data
    try:
        df = pd.read_csv('C:\\Users\\sneha\\PycharmProjects\\less_mills_youtube_analytics\\data\\les_mills_videos.csv')
        df['published_at'] = pd.to_datetime(df['published_at'])
    except:
        # Generate sample data if CSV not found
        print("⚠️ CSV not found, using sample data")
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
# INITIALIZE DASH APP
# ============================================================================

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

app.title = "Les Mills YouTube Analytics"

# ============================================================================
# STYLING
# ============================================================================

colors = {
    'background': '#0f172a',
    'card_bg': '#1e293b',
    'text': '#f1f5f9',
    'primary': '#ef4444',
    'secondary': '#f97316',
    'accent': '#eab308',
    'success': '#10b981'
}

card_style = {
    'backgroundColor': colors['card_bg'],
    'padding': '20px',
    'borderRadius': '12px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
    'marginBottom': '20px',
    'border': f'1px solid #334155'
}

metric_card_style = {
    'backgroundColor': colors['card_bg'],
    'padding': '20px',
    'borderRadius': '12px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
    'textAlign': 'center',
    'border': f'1px solid #334155'
}

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🏋️ Les Mills YouTube Analytics",
                    style={'color': colors['primary'], 'marginBottom': '10px', 'fontSize': '36px',
                           'fontWeight': 'bold'}),
            html.P("Data-Driven Content Performance Insights",
                   style={'color': '#94a3b8', 'fontSize': '16px', 'marginBottom': '0'}),
        ], style={'flex': '1'}),

        html.Div([
            html.P("ML-Powered Analysis",
                   style={'color': '#94a3b8', 'fontSize': '14px', 'marginBottom': '5px', 'textAlign': 'right'}),
            html.P("AI & ML Engineer Application",
                   style={'color': '#64748b', 'fontSize': '12px', 'marginBottom': '0', 'textAlign': 'right'}),
        ]),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '30px'}),

    # Workout Type Filter
    html.Div([
        html.Label("Filter by Workout Type:",
                   style={'color': colors['text'], 'marginRight': '10px', 'fontSize': '16px', 'fontWeight': '600'}),
        dcc.Dropdown(
            id='workout-filter',
            options=[{'label': 'All Workouts', 'value': 'All'}] +
                    [{'label': wt, 'value': wt} for wt in sorted(df['workout_type'].unique())],
            value='All',
            style={'width': '250px', 'backgroundColor': colors['text'], 'color': '#303030'},
            className='dropdown'
        ),
    ], style={'marginBottom': '30px', 'display': 'flex', 'alignItems': 'center'}),

    # Key Metrics Row
    html.Div([
        html.Div([
            html.Div([
                html.H3("📹", style={'fontSize': '36px', 'margin': '0'}),
                html.H2(id='metric-videos',
                        style={'color': colors['text'], 'fontSize': '32px', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("Total Videos", style={'color': '#94a3b8', 'fontSize': '14px', 'margin': '0'}),
            ], style={**metric_card_style, 'background': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'})
        ], style={'width': '24%'}),

        html.Div([
            html.Div([
                html.H3("📈", style={'fontSize': '36px', 'margin': '0'}),
                html.H2(id='metric-total-views',
                        style={'color': colors['text'], 'fontSize': '32px', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("Total Views", style={'color': '#94a3b8', 'fontSize': '14px', 'margin': '0'}),
            ], style={**metric_card_style, 'background': 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)'})
        ], style={'width': '24%'}),

        html.Div([
            html.Div([
                html.H3("👍", style={'fontSize': '36px', 'margin': '0'}),
                html.H2(id='metric-avg-views',
                        style={'color': colors['text'], 'fontSize': '32px', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("Avg Views", style={'color': '#94a3b8', 'fontSize': '14px', 'margin': '0'}),
            ], style={**metric_card_style, 'background': 'linear-gradient(135deg, #eab308 0%, #ca8a04 100%)'})
        ], style={'width': '24%'}),

        html.Div([
            html.Div([
                html.H3("⚡", style={'fontSize': '36px', 'margin': '0'}),
                html.H2(id='metric-engagement',
                        style={'color': colors['text'], 'fontSize': '32px', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("Avg Engagement", style={'color': '#94a3b8', 'fontSize': '14px', 'margin': '0'}),
            ], style={**metric_card_style, 'background': 'linear-gradient(135deg, #10b981 0%, #059669 100%)'})
        ], style={'width': '24%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px', 'gap': '15px'}),

    # Nina's Video Spotlight
    html.Div([
        html.Div([
            html.H2("⭐ Nina's GRIT Cardio Workout",
                    style={'color': colors['primary'], 'marginBottom': '10px', 'fontSize': '24px',
                           'fontWeight': 'bold'}),
            html.P(nina_video['title'] if nina_video is not None else "Nina's workout video",
                   style={'color': '#94a3b8', 'marginBottom': '20px', 'fontSize': '16px'}),

            html.Div([
                html.Div([
                    html.P("Views", style={'color': '#94a3b8', 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.H3(id='nina-views',
                            style={'color': colors['text'], 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0'}),
                ], style={'textAlign': 'center'}),

                html.Div([
                    html.P("Rank", style={'color': '#94a3b8', 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.H3(id='nina-rank',
                            style={'color': '#eab308', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0'}),
                ], style={'textAlign': 'center'}),

                html.Div([
                    html.P("Engagement", style={'color': '#94a3b8', 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.H3(id='nina-engagement',
                            style={'color': '#10b981', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0'}),
                ], style={'textAlign': 'center'}),

                html.Div([
                    html.P("Duration", style={'color': '#94a3b8', 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.H3(id='nina-duration',
                            style={'color': '#3b82f6', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0'}),
                ], style={'textAlign': 'center'}),

                html.Div([
                    html.P("vs Average", style={'color': '#94a3b8', 'fontSize': '12px', 'marginBottom': '5px'}),
                    html.H3(id='nina-vs-avg',
                            style={'color': '#a855f7', 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0'}),
                ], style={'textAlign': 'center'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'}),
        ], style={**card_style,
                  'background': 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%)',
                  'border': '1px solid rgba(239, 68, 68, 0.3)'})
    ], style={'marginBottom': '30px'}),

    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='workout-performance-chart')
        ], style={**card_style, 'width': '49%'}),

        html.Div([
            dcc.Graph(id='nina-radar-chart')
        ], style={**card_style, 'width': '49%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),

    # Charts Row 2
    html.Div([
        html.Div([
            dcc.Graph(id='duration-scatter-chart')
        ], style={**card_style, 'width': '49%'}),

        html.Div([
            dcc.Graph(id='timeline-chart')
        ], style={**card_style, 'width': '49%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),

    # Top Videos Table
    html.Div([
        html.H2("🏆 Top 10 Performing Videos",
                style={'color': colors['text'], 'marginBottom': '20px', 'fontSize': '24px', 'fontWeight': 'bold'}),
        html.Div(id='top-videos-table')
    ], style=card_style),

    # Footer
    html.Div([
        html.P("Les Mills YouTube Analytics Dashboard • Built with Python Dash & Plotly",
               style={'color': '#64748b', 'textAlign': 'center', 'fontSize': '12px', 'marginTop': '40px'}),
        html.P("AI & ML Engineer Application Project • Data-Driven Content Strategy",
               style={'color': '#64748b', 'textAlign': 'center', 'fontSize': '12px'}),
    ])

], style={'backgroundColor': colors['background'], 'padding': '30px', 'minHeight': '100vh', 'color': colors['text']})


# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    [Output('metric-videos', 'children'),
     Output('metric-total-views', 'children'),
     Output('metric-avg-views', 'children'),
     Output('metric-engagement', 'children'),
     Output('nina-views', 'children'),
     Output('nina-rank', 'children'),
     Output('nina-engagement', 'children'),
     Output('nina-duration', 'children'),
     Output('nina-vs-avg', 'children'),
     Output('workout-performance-chart', 'figure'),
     Output('nina-radar-chart', 'figure'),
     Output('duration-scatter-chart', 'figure'),
     Output('timeline-chart', 'figure'),
     Output('top-videos-table', 'children')],
    Input('workout-filter', 'value')
)
def update_dashboard(selected_workout):
    # Filter data
    filtered_df = df if selected_workout == 'All' else df[df['workout_type'] == selected_workout]

    # Calculate metrics
    total_videos = len(filtered_df)
    total_views = filtered_df['view_count'].sum()
    avg_views = int(filtered_df['view_count'].mean())
    avg_engagement = filtered_df['engagement_rate'].mean()

    # Nina's metrics
    if nina_video is not None:
        nina_views_display = f"{nina_video['view_count']:,}"
        nina_rank = (df['view_count'] > nina_video['view_count']).sum() + 1
        nina_rank_display = f"#{nina_rank}"
        nina_engagement_display = f"{nina_video['engagement_rate']:.2f}%"
        nina_duration_display = f"{nina_video['duration_minutes']} min"
        nina_vs_avg = ((nina_video['view_count'] / avg_views - 1) * 100)
        nina_vs_avg_display = f"{nina_vs_avg:+.0f}%"
    else:
        nina_views_display = "N/A"
        nina_rank_display = "N/A"
        nina_engagement_display = "N/A"
        nina_duration_display = "N/A"
        nina_vs_avg_display = "N/A"

    # Chart 1: Workout Performance
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
        title="Average Views by Workout Type",
        title_font=dict(size=18, color=colors['text']),
        xaxis_title="Average Views",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig1.update_xaxes(showgrid=True, gridcolor='#334155')
    fig1.update_yaxes(showgrid=False)

    # Chart 2: Nina Radar
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
            title="Nina's Video vs Channel Average",
            title_font=dict(size=18, color=colors['text']),
            polar=dict(
                radialaxis=dict(visible=True, showgrid=True, gridcolor='#334155', color=colors['text']),
                angularaxis=dict(showgrid=True, gridcolor='#334155', color=colors['text'])
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text']),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
    else:
        fig2 = go.Figure()

    # Chart 3: Duration vs Views
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
        title="Video Duration vs Views",
        title_font=dict(size=18, color=colors['text']),
        xaxis_title="Duration (minutes)",
        yaxis_title="Views",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    fig3.update_xaxes(showgrid=True, gridcolor='#334155')
    fig3.update_yaxes(showgrid=True, gridcolor='#334155')

    # Chart 4: Timeline
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
        title="Average Views Over Time",
        title_font=dict(size=18, color=colors['text']),
        xaxis_title="Month",
        yaxis_title="Average Views",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig4.update_xaxes(showgrid=True, gridcolor='#334155')
    fig4.update_yaxes(showgrid=True, gridcolor='#334155')

    # Top Videos Table
    top_10 = filtered_df.head(10)

    table_header = html.Thead(html.Tr([
        html.Th("Rank", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'left'}),
        html.Th("Title", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'left'}),
        html.Th("Type", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'left'}),
        html.Th("Views", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'right'}),
        html.Th("Engagement", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'right'}),
        html.Th("Duration", style={'color': '#94a3b8', 'padding': '12px', 'textAlign': 'right'}),
    ]))

    table_rows = []
    for idx, row in top_10.iterrows():
        is_nina = row.get('is_nina_video', False)
        row_style = {'backgroundColor': 'rgba(239, 68, 68, 0.1)' if is_nina else 'transparent'}

        table_rows.append(html.Tr([
            html.Td(f"#{idx + 1}",
                    style={'padding': '12px', 'color': '#ef4444' if is_nina else colors['text'], 'fontWeight': 'bold'}),
            html.Td([
                "⭐ " if is_nina else "",
                row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            ], style={'padding': '12px', 'color': '#ef4444' if is_nina else colors['text']}),
            html.Td(row['workout_type'], style={'padding': '12px', 'color': '#94a3b8'}),
            html.Td(f"{row['view_count']:,}",
                    style={'padding': '12px', 'textAlign': 'right', 'fontWeight': '600', 'color': colors['text']}),
            html.Td(f"{row['engagement_rate']:.2f}%",
                    style={'padding': '12px', 'textAlign': 'right', 'color': '#10b981'}),
            html.Td(f"{row['duration_minutes']} min",
                    style={'padding': '12px', 'textAlign': 'right', 'color': '#94a3b8'}),
        ], style=row_style))

    table = html.Table([table_header, html.Tbody(table_rows)],
                       style={'width': '100%', 'borderCollapse': 'collapse'})

    return (
        str(total_videos),
        f"{total_views / 1000000:.1f}M",
        f"{avg_views / 1000:.0f}K",
        f"{avg_engagement:.2f}%",
        nina_views_display,
        nina_rank_display,
        nina_engagement_display,
        nina_duration_display,
        nina_vs_avg_display,
        fig1, fig2, fig3, fig4, table
    )


# ============================================================================
# RUN APP
# ============================================================================

# Expose server for deployment
server = app.server

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 LES MILLS YOUTUBE ANALYTICS DASHBOARD")
    print("=" * 70)
    print("\n📊 Starting dashboard server...")
    print("🌐 Open your browser to: http://localhost:8050")
    print("⚠️  Press Ctrl+C to stop the server\n")

    #app.run(debug=True, host='0.0.0.0', port=8050)

    app.run_server(debug=False, host='0.0.0.0', port=8050)