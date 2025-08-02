# Professional Conversation Analysis Visualization using Streamlit and Plotly

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# ------------------------- Helper Functions -------------------------
TONE_COLORS = {
    "supportive": "#66BB6A",
    "anxious": "#EF5350",
    "assertive": "#42A5F5",
    "confident": "#AB47BC",
    "empathetic": "#26A69A",
    "frustrated": "#FF7043",
    "neutral": "#BDBDBD"
}

SPEAKER_COLORS = {
    "Manager": "#1f77b4",
    "Employee": "#ff7f0e",
    "Unknown": "#888"
}

# ------------------------ Visualization Functions ------------------------

def create_summary_cards(stats: Dict[str, Any]):
    cards = []
    cards.append(go.Indicator(
        mode="number",
        value=stats.get("total_utterances", 0),
        title={"text": "Total Utterances", "font": {"size": 16}},
        number={"font": {"size": 26}},
        domain={"row": 0, "column": 0}
    ))
    cards.append(go.Indicator(
        mode="number",
        value=stats.get("manager_utterances", 0),
        title={"text": "Manager Utterances", "font": {"size": 16}},
        number={"font": {"size": 26}},
        domain={"row": 0, "column": 1}
    ))
    cards.append(go.Indicator(
        mode="number",
        value=stats.get("employee_utterances", 0),
        title={"text": "Employee Utterances", "font": {"size": 16}},
        number={"font": {"size": 26}},
        domain={"row": 0, "column": 2}
    ))
    cards.append(go.Indicator(
        mode="gauge+number",
        value=stats.get("average_sentiment", 0),
        title={"text": "Average Sentiment (–1 to +1)", "font": {"size": 16}},
        number={"font": {"size": 24}},
        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#00BFC4"}},
        domain={"row": 0, "column": 3}
    ))

    fig = go.Figure(cards)
    fig.update_layout(
        grid={"rows": 1, "columns": 4, "pattern": "independent"},
        template="plotly_white",
        height=250,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor="#F0F2F6"
    )
    return fig

def create_sentiment_timeline(utterances: List[Dict[str, Any]]):
    x = list(range(1, len(utterances)+1))
    y = [u.get('sentiment_score', 0) for u in utterances]
    colors = ['green' if s > 0.2 else 'red' if s < -0.2 else 'gray' for s in y]
    speakers = [u.get('speaker', 'Unknown') for u in utterances]
    sentiments = [u.get('sentiment', 'neutral') for u in utterances]
    texts = [u.get('text', '')[:100] + '...' if len(u.get('text', '')) > 100 else u.get('text', '') for u in utterances]

    fig = go.Figure(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        marker=dict(size=8, color=colors, line=dict(color='white', width=1)),
        customdata=list(zip(speakers, sentiments, texts)),
        hovertemplate='<b>Utterance %{x}</b><br>Speaker: %{customdata[0]}<br>Sentiment: %{customdata[1]}<br>Score: %{y:.2f}<br>Text: %{customdata[2]}<extra></extra>'
    ))

    fig.update_layout(title='🧠 Sentiment Timeline', xaxis_title='Utterance #', yaxis_title='Score', yaxis=dict(range=[-1, 1]), height=350)
    return fig

def create_tone_distribution(utterances: List[Dict[str, Any]]):
    tones = [u.get('tone', 'neutral') for u in utterances]
    counts = Counter(tones)
    tones_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[t[1] for t in tones_sorted],
        y=[t[0].capitalize() for t in tones_sorted],
        orientation='h',
        marker=dict(color=[TONE_COLORS.get(t[0], '#ccc') for t in tones_sorted]),
        text=[t[1] for t in tones_sorted],
        textposition='auto'
    ))
    fig.update_layout(title='🎯 Tone Distribution', xaxis_title='Count', yaxis_title='Tone', height=350)
    return fig

def create_speaker_participation(utterances: List[Dict[str, Any]]):
    speakers = [u.get('speaker', 'Unknown') for u in utterances]
    counts = Counter(speakers)
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker_colors=[SPEAKER_COLORS.get(s, '#999') for s in counts.keys()],
        hole=0.4
    ))
    fig.update_layout(title='🗣️ Speaker Participation', height=350)
    return fig

def create_sentiment_by_speaker(utterances: List[Dict[str, Any]]):
    data = {}
    for u in utterances:
        speaker = u.get('speaker', 'Unknown')
        data.setdefault(speaker, []).append(u.get('sentiment_score', 0))

    fig = go.Figure()
    for i, (speaker, scores) in enumerate(data.items()):
        fig.add_trace(go.Box(y=scores, name=speaker, marker_color=SPEAKER_COLORS.get(speaker, '#999')))

    fig.update_layout(title='📈 Sentiment by Speaker', yaxis_title='Score', yaxis=dict(range=[-1, 1]), height=350)
    return fig

def create_wordcloud_image(utterances: List[Dict[str, Any]]):
    text = " ".join([u.get('text', '') for u in utterances])
    wc = WordCloud(width=700, height=350, background_color='white', colormap='viridis', max_words=100).generate(text)
    buf = io.BytesIO()
    plt.figure(figsize=(9, 4.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

# ------------------------ Streamlit Dashboard ------------------------

def display_dashboard(analysis_results: Dict[str, Any]):
    st.set_page_config(page_title="Conversation Dashboard", layout="wide")
    st.markdown("""
        <style>
            .main { background-color: #FAFAFA; }
            h1, h2, h3 { color: #00BFC4; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Growth Talk Analysis Dashboard")
    st.markdown("Comprehensive analysis of manager-employee conversation.")

    utterances = analysis_results.get('utterances', [])
    stats = analysis_results.get('mom_summary', {}).get('statistics', {})

    st.plotly_chart(create_summary_cards(stats), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_sentiment_timeline(utterances), use_container_width=True)
    with col2:
        st.plotly_chart(create_tone_distribution(utterances), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(create_speaker_participation(utterances), use_container_width=True)
    with col4:
        st.plotly_chart(create_sentiment_by_speaker(utterances), use_container_width=True)

    st.subheader("☁️ Word Cloud of Conversation")
    wc_buf = create_wordcloud_image(utterances)
    st.image(wc_buf, use_container_width=True)

# Example Usage
# import json
# with open('analysis_results.json') as f:
#     data = json.load(f)
# display_dashboard(data)
