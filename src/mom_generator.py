try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available, using extractive summarization")

from typing import List, Dict, Any
import json
import re

class MoMGenerator:
    """Minutes of Meeting generator using transformer models"""
    
    def __init__(self, model_name: str = "T5-base"):
        self.model_name = model_name
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Load summarization model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, using extractive summarization.")
            self.summarizer = None
            return
            
        try:
            # Try to load T5 or BART model for summarization
            if "T5" in self.model_name:
                model_id = "t5-small"  # Use small model for faster inference
            elif "BART" in self.model_name:
                model_id = "facebook/bart-large-cnn"
            else:
                model_id = "t5-small"
            
            self.summarizer = pipeline(
                "summarization",
                model=model_id,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            
        except Exception as e:
            print(f"Warning: Could not load summarization model ({e}). Using extractive summarization.")
            self.summarizer = None
    
    def generate_summary(self, utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate structured Minutes of Meeting summary
        
        Args:
            utterances: List of analyzed utterances with sentiment and tone
            
        Returns:
            Structured MoM summary
        """
        try:
            # Prepare text for summarization
            full_transcript = self._prepare_transcript(utterances)
            
            # Extract key components
            key_points = self._extract_key_points(utterances)
            action_items = self._extract_action_items(utterances)
            emotional_summary = self._analyze_emotional_flow(utterances)
            tone_summary = self._analyze_tone_trends(utterances)
            
            # Generate overall summary
            if self.summarizer:
                summary_text = self._generate_ai_summary(full_transcript)
            else:
                summary_text = self._generate_extractive_summary(utterances)
            
            # Determine overall sentiment and outcome
            overall_sentiment = self._determine_overall_sentiment(utterances)
            meeting_outcome = self._determine_meeting_outcome(utterances, overall_sentiment)
            
            return {
                "key_points": key_points,
                "action_items": action_items,
                "emotional_summary": emotional_summary,
                "tone_summary": tone_summary,
                "overall_summary": summary_text,
                "overall_sentiment": overall_sentiment,
                "meeting_outcome": meeting_outcome,
                "statistics": self._compute_statistics(utterances)
            }
            
        except Exception as e:
            print(f"Error generating MoM: {e}")
            return self._generate_fallback_summary(utterances)
    
    def _prepare_transcript(self, utterances: List[Dict[str, Any]]) -> str:
        """Prepare clean transcript text for summarization"""
        transcript_parts = []
        for utterance in utterances:
            speaker = utterance['speaker']
            text = utterance['text']
            transcript_parts.append(f"{speaker}: {text}")
        
        return "\n".join(transcript_parts)
    
    def _extract_key_points(self, utterances: List[Dict[str, Any]]) -> List[str]:
        """Extract key discussion points from utterances"""
        key_points = []
        
        # Keywords that often indicate important points
        key_indicators = [
            'goal', 'objective', 'target', 'priority', 'important', 'focus',
            'improvement', 'development', 'feedback', 'performance', 'achievement',
            'challenge', 'opportunity', 'issue', 'concern', 'solution'
        ]
        
        for utterance in utterances:
            text_lower = utterance['text'].lower()
            
            # Check if utterance contains key indicators
            if any(indicator in text_lower for indicator in key_indicators):
                # Clean and add to key points
                if len(utterance['text']) > 20:  # Filter out very short utterances
                    key_points.append(utterance['text'])
            
            # Also include high-confidence assertive or directive statements
            if (utterance.get('tone') in ['assertive', 'directive'] and 
                utterance.get('tone_confidence', 0) > 0.7):
                if len(utterance['text']) > 20:
                    key_points.append(utterance['text'])
        
        # Limit to most relevant points
        return key_points[:5]
    
    def _extract_action_items(self, utterances: List[Dict[str, Any]]) -> List[str]:
        """Extract action items and next steps"""
        action_items = []
        
        action_patterns = [
            r'\bwill\s+\w+', r'\bgoing to\s+\w+', r'\bneed to\s+\w+',
            r'\bshould\s+\w+', r'\bmust\s+\w+', r'\bplan to\s+\w+',
            r'\bnext step', r'\baction item', r'\bfollow up',
            r'\bby\s+\w+day', r'\bdeadline', r'\bdue date'
        ]
        
        for utterance in utterances:
            text = utterance['text']
            
            # Check for action patterns
            for pattern in action_patterns:
                if re.search(pattern, text.lower()):
                    action_items.append(text)
                    break
        
        # Remove duplicates and limit
        unique_actions = list(set(action_items))
        return unique_actions[:3]
    
    def _analyze_emotional_flow(self, utterances: List[Dict[str, Any]]) -> str:
        """Analyze the emotional progression during the conversation"""
        sentiment_progression = [u.get('sentiment_score', 0) for u in utterances]
        
        if not sentiment_progression:
            return "No emotional data available."
        
        start_sentiment = sentiment_progression[0]
        end_sentiment = sentiment_progression[-1]
        avg_sentiment = sum(sentiment_progression) / len(sentiment_progression)
        
        # Determine trend
        if end_sentiment > start_sentiment + 0.2:
            trend = "improved significantly"
        elif end_sentiment > start_sentiment:
            trend = "improved slightly"
        elif end_sentiment < start_sentiment - 0.2:
            trend = "declined significantly"
        elif end_sentiment < start_sentiment:
            trend = "declined slightly"
        else:
            trend = "remained stable"
        
        # Describe overall emotional tone
        if avg_sentiment > 0.3:
            overall_tone = "positive"
        elif avg_sentiment < -0.3:
            overall_tone = "negative"
        else:
            overall_tone = "neutral"
        
        return f"The conversation maintained a {overall_tone} emotional tone and {trend} throughout the discussion."
    
    def _analyze_tone_trends(self, utterances: List[Dict[str, Any]]) -> str:
        """Analyze tone patterns and trends"""
        tones = [u.get('tone', 'neutral') for u in utterances]
        
        if not tones:
            return "No tone data available."
        
        from collections import Counter
        tone_counts = Counter(tones)
        dominant_tones = tone_counts.most_common(3)
        
        tone_description = f"The conversation was primarily {dominant_tones[0][0]}"
        if len(dominant_tones) > 1:
            tone_description += f", with elements of {dominant_tones[1][0]}"
        if len(dominant_tones) > 2:
            tone_description += f" and {dominant_tones[2][0]}"
        
        tone_description += " communication."
        
        return tone_description
    
    def _generate_ai_summary(self, transcript: str) -> str:
        """Generate AI-powered summary using transformer model"""
        try:
            # Truncate if too long
            max_input_length = 1000
            if len(transcript) > max_input_length:
                transcript = transcript[:max_input_length] + "..."
            
            summary = self.summarizer(transcript, max_length=150, min_length=50)
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"Error in AI summarization: {e}")
            return self._generate_extractive_summary([])
    
    def _generate_extractive_summary(self, utterances: List[Dict[str, Any]]) -> str:
        """Generate extractive summary as fallback"""
        if not utterances:
            return "Unable to generate summary from the conversation."
        
        # Find most important utterances based on length and tone
        important_utterances = []
        for utterance in utterances:
            text = utterance['text']
            tone_conf = utterance.get('tone_confidence', 0)
            
            # Score based on length and confidence
            score = len(text.split()) * 0.1 + tone_conf * 0.5
            
            if score > 0.3 and len(text) > 30:
                important_utterances.append((score, text))
        
        # Sort by score and take top utterances
        important_utterances.sort(reverse=True)
        top_utterances = [text for _, text in important_utterances[:3]]
        
        if top_utterances:
            return "Key discussion points: " + " | ".join(top_utterances)
        else:
            return "The conversation covered various topics related to growth and development."
    
    def _determine_overall_sentiment(self, utterances: List[Dict[str, Any]]) -> str:
        """Determine overall sentiment of the conversation"""
        sentiment_scores = [u.get('sentiment_score', 0) for u in utterances if u.get('sentiment_score') is not None]
        
        if not sentiment_scores:
            return "neutral"
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        if avg_sentiment > 0.2:
            return "positive"
        elif avg_sentiment < -0.2:
            return "negative"
        else:
            return "neutral"
    
    def _determine_meeting_outcome(self, utterances: List[Dict[str, Any]], overall_sentiment: str) -> str:
        """Determine the meeting outcome based on analysis"""
        action_count = len(self._extract_action_items(utterances))
        key_points_count = len(self._extract_key_points(utterances))
        
        if overall_sentiment == "positive" and action_count > 0:
            return "The conversation was productive with clear action items and positive engagement."
        elif overall_sentiment == "positive":
            return "The conversation was positive and constructive, building good rapport."
        elif overall_sentiment == "negative" and action_count > 0:
            return "Despite some challenges discussed, concrete steps were identified for improvement."
        elif overall_sentiment == "negative":
            return "The conversation addressed concerns that may require follow-up discussion."
        else:
            return "The conversation covered important topics with balanced discussion."
    
    def _compute_statistics(self, utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute conversation statistics"""
        total_utterances = len(utterances)
        manager_utterances = sum(1 for u in utterances if u.get('speaker') == 'Manager')
        employee_utterances = sum(1 for u in utterances if u.get('speaker') == 'Employee')
        
        avg_sentiment = 0
        if utterances:
            sentiment_scores = [u.get('sentiment_score', 0) for u in utterances]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        return {
            "total_utterances": total_utterances,
            "manager_utterances": manager_utterances,
            "employee_utterances": employee_utterances,
            "average_sentiment": round(avg_sentiment, 3),
            "participation_ratio": round(manager_utterances / max(1, employee_utterances), 2)
        }
    
    def _generate_fallback_summary(self, utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate basic summary when main process fails"""
        return {
            "key_points": ["Unable to extract key points"],
            "action_items": ["Please review transcript for action items"],
            "emotional_summary": "Emotional analysis unavailable",
            "tone_summary": "Tone analysis unavailable",
            "overall_summary": "Summary generation failed - please check the transcript manually",
            "overall_sentiment": "neutral",
            "meeting_outcome": "Unable to determine meeting outcome",
            "statistics": self._compute_statistics(utterances)
        }
