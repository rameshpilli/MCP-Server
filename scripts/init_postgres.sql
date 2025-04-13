-- Create metadata schema
CREATE SCHEMA IF NOT EXISTS metadata;

-- Create data schema
CREATE SCHEMA IF NOT EXISTS jobs_data;
CREATE SCHEMA IF NOT EXISTS news_data;

-- Metadata tables
CREATE TABLE IF NOT EXISTS metadata.data_sources (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS metadata.tables (
    id VARCHAR(255) PRIMARY KEY,
    data_source_id VARCHAR(255) REFERENCES metadata.data_sources(id),
    name VARCHAR(255) NOT NULL,
    schema VARCHAR(255),
    description TEXT,
    topic VARCHAR(255),
    columns JSONB,
    sample_queries JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata.query_history (
    id VARCHAR(255) PRIMARY KEY,
    api_key VARCHAR(255) NOT NULL,
    natural_query TEXT,
    sql_query TEXT,
    data_source_id VARCHAR(255) REFERENCES metadata.data_sources(id),
    execution_time FLOAT,
    row_count INTEGER,
    status VARCHAR(50),
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response JSONB
);

-- Actual data tables
CREATE TABLE IF NOT EXISTS jobs_data.active_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    type VARCHAR(100),
    priority INTEGER,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS jobs_data.jobs_history (
    job_id VARCHAR(255) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration FLOAT,
    error TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS news_data.articles (
    article_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    published_at TIMESTAMP NOT NULL,
    source VARCHAR(255),
    category VARCHAR(100),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS news_data.analytics (
    article_id VARCHAR(255) REFERENCES news_data.articles(article_id),
    sentiment_score FLOAT,
    topic_tags TEXT[],
    read_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO jobs_data.active_jobs (job_id, status, created_at, type, priority, metadata) VALUES
('job_001', 'running', NOW(), 'data_processing', 1, '{"owner": "system", "cpu_usage": 45}'),
('job_002', 'queued', NOW(), 'ml_training', 2, '{"model": "bert", "batch_size": 32}'),
('job_003', 'running', NOW(), 'etl', 1, '{"source": "sales_db", "target": "dw"}');

INSERT INTO news_data.articles (article_id, title, content, published_at, source, category) VALUES
('art_001', 'AI Breakthrough', 'Scientists discover...', NOW(), 'TechNews', 'technology'),
('art_002', 'Market Update', 'Stock markets show...', NOW(), 'FinanceDaily', 'finance');

-- Create indexes
CREATE INDEX idx_jobs_status ON jobs_data.active_jobs(status);
CREATE INDEX idx_jobs_created ON jobs_data.active_jobs(created_at);
CREATE INDEX idx_news_category ON news_data.articles(category);
CREATE INDEX idx_news_published ON news_data.articles(published_at); 