import re
import os
from typing import Any, Dict, List
import streamlit as st

def validate_vtt_file(uploaded_file) -> bool:
    """
    Validate if uploaded file is a proper VTT file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid VTT file, False otherwise
    """
    try:
        # Check file extension
        if not uploaded_file.name.lower().endswith('.vtt'):
            return False
        
        # Read first few lines to check format
        content = uploaded_file.read().decode('utf-8')
        lines = content.split('\n')
        
        # Reset file pointer for later use
        uploaded_file.seek(0)
        
        # Check for VTT header
        if not lines[0].strip().startswith('WEBVTT'):
            return False
        
        # Check for at least one timestamp
        has_timestamp = any('-->' in line for line in lines)
        
        return has_timestamp
        
    except Exception as e:
        st.error(f"Error validating VTT file: {str(e)}")
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common VTT formatting tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove speaker indicators that might be embedded
    text = re.sub(r'^\w+:\s*', '', text)
    
    return text

def format_timestamp(seconds: float) -> str:
    """Format seconds to readable timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def calculate_speaking_time(utterances: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate total speaking time per speaker"""
    speaking_time = {}
    
    for utterance in utterances:
        speaker = utterance.get('speaker', 'Unknown')
        duration = utterance.get('duration', 0)
        
        if speaker not in speaking_time:
            speaking_time[speaker] = 0
        
        speaking_time[speaker] += duration
    
    return speaking_time

def get_conversation_stats(utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get comprehensive conversation statistics"""
    if not utterances:
        return {}
    
    total_utterances = len(utterances)
    total_duration = sum(u.get('duration', 0) for u in utterances)
    
    # Speaker stats
    speakers = [u.get('speaker', 'Unknown') for u in utterances]
    from collections import Counter
    speaker_counts = Counter(speakers)
    
    # Sentiment stats
    sentiments = [u.get('sentiment', 'neutral') for u in utterances if u.get('sentiment')]
    sentiment_counts = Counter(sentiments)
    
    # Average metrics
    sentiment_scores = [u.get('sentiment_score', 0) for u in utterances if u.get('sentiment_score') is not None]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    tone_confidences = [u.get('tone_confidence', 0) for u in utterances if u.get('tone_confidence') is not None]
    avg_tone_confidence = sum(tone_confidences) / len(tone_confidences) if tone_confidences else 0
    
    return {
        'total_utterances': total_utterances,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_timestamp(total_duration),
        'speaker_distribution': dict(speaker_counts),
        'sentiment_distribution': dict(sentiment_counts),
        'average_sentiment_score': round(avg_sentiment, 3),
        'average_tone_confidence': round(avg_tone_confidence, 3),
        'speaking_time_per_speaker': calculate_speaking_time(utterances)
    }

def export_data_to_csv(utterances: List[Dict[str, Any]]) -> str:
    """Export utterances to CSV format"""
    import csv
    import io
    
    output = io.StringIO()
    
    if not utterances:
        return ""
    
    # Get all possible keys
    all_keys = set()
    for utterance in utterances:
        all_keys.update(utterance.keys())
    
    fieldnames = sorted(list(all_keys))
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for utterance in utterances:
        # Convert timedelta objects to strings
        row = {}
        for key, value in utterance.items():
            if hasattr(value, 'total_seconds'):  # timedelta object
                row[key] = str(value)
            else:
                row[key] = value
        writer.writerow(row)
    
    return output.getvalue()

def create_download_filename(prefix: str = "gta_analysis", extension: str = "txt") -> str:
    """Create timestamped filename for downloads"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def log_analysis_metrics(analysis_results: Dict[str, Any]):
    """Log key metrics for monitoring (could be extended for production)"""
    try:
        utm_count = len(analysis_results.get('utterances', []))
        avg_sentiment = analysis_results.get('sentiment_stats', {}).get('average_sentiment', 0)
        dominant_tone = analysis_results.get('tone_stats', {}).get('dominant_tone', 'unknown')
        
        # For now, just use Streamlit's logging
        # In production, this could send to monitoring services
        st.write(f"Analysis completed: {utm_count} utterances, avg sentiment: {avg_sentiment:.2f}, dominant tone: {dominant_tone}")
        
    except Exception as e:
        st.warning(f"Could not log analysis metrics: {str(e)}")

def safe_get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get nested dictionary values using dot notation"""
    try:
        keys = path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def detect_language(text: str) -> str:
    """Simple language detection (can be enhanced with proper libraries)"""
    # This is a very basic implementation
    # In production, consider using langdetect or similar libraries
    
    english_indicators = ['the', 'and', 'is', 'to', 'in', 'you', 'that', 'it', 'with', 'for']
    
    text_lower = text.lower()
    english_score = sum(1 for word in english_indicators if word in text_lower)
    
    if english_score >= 2:
        return 'en'
    else:
        return 'unknown'

def validate_analysis_results(results: Dict[str, Any]) -> bool:
    """Validate analysis results structure"""
    required_keys = ['utterances', 'sentiment_stats', 'tone_stats']
    
    for key in required_keys:
        if key not in results:
            return False
    
    if not isinstance(results['utterances'], list):
        return False
    
    return True

def get_color_for_sentiment(sentiment: str) -> str:
    """Get color code for sentiment visualization"""
    color_map = {
        'positive': '#28a745',  # Green
        'negative': '#dc3545',  # Red
        'neutral': '#6c757d'    # Gray
    }
    return color_map.get(sentiment.lower(), '#6c757d')

def get_color_for_tone(tone: str) -> str:
    """Get color code for tone visualization"""
    color_map = {
        'empathetic': '#17a2b8',    # Cyan
        'assertive': '#ffc107',     # Yellow
        'anxious': '#fd7e14',       # Orange
        'confident': '#28a745',     # Green
        'frustrated': '#dc3545',    # Red
        'supportive': '#6f42c1',    # Purple
        'questioning': '#20c997',   # Teal
        'directive': '#e83e8c',     # Pink
        'encouraging': '#28a745',   # Green
        'concerned': '#fd7e14'      # Orange
    }
    return color_map.get(tone.lower(), '#6c757d')
