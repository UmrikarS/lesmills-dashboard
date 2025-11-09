
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
