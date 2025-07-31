import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

def create_sentiment_timeline(utterances: List[Dict[str, Any]]):
    """Create sentiment timeline visualization"""
    
    # Prepare data
    timestamps = []
    sentiment_scores = []
    speakers = []
    texts = []
    sentiments = []
    
    for i, utterance in enumerate(utterances):
        timestamps.append(i + 1)  # Use sequence number as x-axis
        sentiment_scores.append(utterance.get('sentiment_score', 0))
        speakers.append(utterance.get('speaker', 'Unknown'))
        texts.append(utterance.get('text', '')[:100] + '...' if len(utterance.get('text', '')) > 100 else utterance.get('text', ''))
        sentiments.append(utterance.get('sentiment', 'neutral'))
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=sentiment_scores,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Utterance %{x}</b><br>' +
                      'Speaker: %{customdata[0]}<br>' +
                      'Sentiment: %{customdata[1]}<br>' +
                      'Score: %{y:.2f}<br>' +
                      'Text: %{customdata[2]}<extra></extra>',
        customdata=list(zip(speakers, sentiments, texts))
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add positive/negative zones
    fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    # Update layout
    fig.update_layout(
        title='Sentiment Timeline Throughout Conversation',
        xaxis_title='Utterance Number',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_tone_distribution(utterances: List[Dict[str, Any]]):
    """Create tone distribution pie chart"""
    
    tones = [utterance.get('tone', 'neutral') for utterance in utterances]
    tone_counts = Counter(tones)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(tone_counts.keys()),
        values=list(tone_counts.values()),
        hole=0.3,
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Tone Distribution in Conversation',
        annotations=[dict(text='Tones', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=400
    )
    
    return fig

def create_emotional_flow(utterances: List[Dict[str, Any]]):
    """Create emotional flow heatmap"""
    
    # Prepare data for heatmap
    speakers = ['Manager', 'Employee']
    tones = ['empathetic', 'assertive', 'anxious', 'confident', 'frustrated', 'supportive']
    
    # Count tone-speaker combinations
    data_matrix = []
    for speaker in speakers:
        speaker_row = []
        speaker_utterances = [u for u in utterances if u.get('speaker') == speaker]
        speaker_tones = [u.get('tone', 'neutral') for u in speaker_utterances]
        tone_counts = Counter(speaker_tones)
        
        for tone in tones:
            count = tone_counts.get(tone, 0)
            speaker_row.append(count)
        data_matrix.append(speaker_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=tones,
        y=speakers,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='Speaker: %{y}<br>Tone: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Tone Distribution by Speaker',
        xaxis_title='Tone',
        yaxis_title='Speaker',
        height=300
    )
    
    return fig

def create_speaker_participation(utterances: List[Dict[str, Any]]):
    """Create speaker participation chart"""
    
    speakers = [utterance.get('speaker', 'Unknown') for utterance in utterances]
    speaker_counts = Counter(speakers)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(speaker_counts.keys()),
            y=list(speaker_counts.values()),
            marker_color=['#1f77b4', '#ff7f0e'],
            hovertemplate='<b>%{x}</b><br>Utterances: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Speaker Participation',
        xaxis_title='Speaker',
        yaxis_title='Number of Utterances',
        height=300
    )
    
    return fig

def create_sentiment_by_speaker(utterances: List[Dict[str, Any]]):
    """Create sentiment comparison by speaker"""
    
    # Group by speaker
    speaker_sentiments = {}
    for utterance in utterances:
        speaker = utterance.get('speaker', 'Unknown')
        sentiment_score = utterance.get('sentiment_score', 0)
        
        if speaker not in speaker_sentiments:
            speaker_sentiments[speaker] = []
        speaker_sentiments[speaker].append(sentiment_score)
    
    # Create box plot
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (speaker, scores) in enumerate(speaker_sentiments.items()):
        fig.add_trace(go.Box(
            y=scores,
            name=speaker,
            marker_color=colors[i % len(colors)],
            boxmean='sd'  # Show mean and standard deviation
        ))
    
    fig.update_layout(
        title='Sentiment Distribution by Speaker',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        height=400
    )
    
    return fig

def create_conversation_metrics_dashboard(analysis_results: Dict[str, Any]):
    """Create comprehensive metrics dashboard"""
    
    utterances = analysis_results.get('utterances', [])
    mom_stats = analysis_results.get('mom_summary', {}).get('statistics', {})
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Timeline', 'Tone Distribution', 
                       'Speaker Participation', 'Sentiment by Speaker'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # Sentiment timeline (top left)
    timestamps = list(range(1, len(utterances) + 1))
    sentiment_scores = [u.get('sentiment_score', 0) for u in utterances]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=sentiment_scores, mode='lines+markers', name='Sentiment'),
        row=1, col=1
    )
    
    # Tone distribution (top right)
    tones = [u.get('tone', 'neutral') for u in utterances]
    tone_counts = Counter(tones)
    
    fig.add_trace(
        go.Pie(labels=list(tone_counts.keys()), values=list(tone_counts.values()), name='Tones'),
        row=1, col=2
    )
    
    # Speaker participation (bottom left)
    speakers = [u.get('speaker', 'Unknown') for u in utterances]
    speaker_counts = Counter(speakers)
    
    fig.add_trace(
        go.Bar(x=list(speaker_counts.keys()), y=list(speaker_counts.values()), name='Participation'),
        row=2, col=1
    )
    
    # Sentiment by speaker (bottom right)
    for speaker in set(speakers):
        speaker_sentiments = [u.get('sentiment_score', 0) for u in utterances if u.get('speaker') == speaker]
        fig.add_trace(
            go.Box(y=speaker_sentiments, name=f'{speaker} Sentiment'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Conversation Analysis Dashboard")
    
    return fig
