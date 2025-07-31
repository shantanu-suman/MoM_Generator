try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available, using rule-based tone analysis")

from typing import Dict, Any, List
import re

class ToneClassifier:
    """Tone classification for utterances"""
    
    def __init__(self, model_name: str = "DistilBERT-tone"):
        self.model_name = model_name
        self.tone_categories = [
            'empathetic', 'assertive', 'anxious', 'confident', 'frustrated',
            'supportive', 'questioning', 'directive', 'encouraging', 'concerned'
        ]
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize tone classification model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, using rule-based tone analysis.")
            self.classifier = None
            self.use_emotion_model = False
            return
            
        try:
            # Try to use emotion classification model as proxy for tone
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            self.use_emotion_model = True
        except Exception as e:
            print(f"Warning: Could not load emotion model ({e}). Using rule-based tone analysis.")
            self.classifier = None
            self.use_emotion_model = False
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify tone of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with tone label and confidence score
        """
        if self.classifier is not None and self.use_emotion_model:
            return self._emotion_to_tone_classification(text)
        else:
            return self._rule_based_tone_classification(text)
    
    def _emotion_to_tone_classification(self, text: str) -> Dict[str, Any]:
        """Use emotion model as proxy for tone classification"""
        try:
            # Get emotion predictions
            emotions = self.classifier(text)
            
            # Map emotions to tones
            emotion_to_tone_mapping = {
                'joy': 'encouraging',
                'sadness': 'concerned',
                'anger': 'frustrated',
                'fear': 'anxious',
                'surprise': 'questioning',
                'disgust': 'frustrated',
                'love': 'supportive'
            }
            
            # Find the highest scoring emotion
            top_emotion = max(emotions[0], key=lambda x: x['score'])
            emotion_label = top_emotion['label']
            confidence = top_emotion['score']
            
            # Map to tone
            tone_label = emotion_to_tone_mapping.get(emotion_label, 'assertive')
            
            return {
                "label": tone_label,
                "confidence": confidence,
                "emotion_source": emotion_label
            }
            
        except Exception as e:
            print(f"Error in emotion-based tone classification: {e}")
            return self._rule_based_tone_classification(text)
    
    def _rule_based_tone_classification(self, text: str) -> Dict[str, Any]:
        """Rule-based tone classification using linguistic patterns"""
        
        text_lower = text.lower()
        
        # Define tone patterns
        tone_patterns = {
            'empathetic': [
                r'\bi understand\b', r'\bi can see\b', r'\bthat must be\b',
                r'\bi hear you\b', r'\bi feel\b', r'\byour concerns?\b'
            ],
            'assertive': [
                r'\bwe need to\b', r'\byou should\b', r'\bit\'s important\b',
                r'\bmust\b', r'\brequire\b', r'\bexpect\b'
            ],
            'anxious': [
                r'\bi\'m worried\b', r'\bconcerned about\b', r'\bnot sure\b',
                r'\bwhat if\b', r'\bafraid\b', r'\buncomfortable\b'
            ],
            'confident': [
                r'\bi\'m confident\b', r'\bcertainly\b', r'\bdefinitely\b',
                r'\babsolutely\b', r'\bwithout a doubt\b', r'\bi know\b'
            ],
            'frustrated': [
                r'\bthis is ridiculous\b', r'\bi can\'t believe\b', r'\bso annoying\b',
                r'\bwhy is\b', r'\balways\b.*\bproblem\b', r'\bsick of\b'
            ],
            'supportive': [
                r'\bi\'ll help\b', r'\bwe can work\b', r'\blet me support\b',
                r'\bi\'m here\b', r'\btogether\b', r'\bteam\b'
            ],
            'questioning': [
                r'\bwhat do you think\b', r'\bhow do you feel\b', r'\bwould you\b',
                r'\bcan you explain\b', r'\bwhy\b', r'\bhow\b.*\?'
            ],
            'directive': [
                r'\bplease do\b', r'\bneed you to\b', r'\bgo ahead and\b',
                r'\bmake sure\b', r'\bensure that\b', r'\btake care of\b'
            ],
            'encouraging': [
                r'\bgreat job\b', r'\bwell done\b', r'\bkeep it up\b',
                r'\byou can do\b', r'\bexcellent\b', r'\bproud of\b'
            ],
            'concerned': [
                r'\bi\'m concerned\b', r'\bworried about\b', r'\btrouble with\b',
                r'\bissue with\b', r'\bproblem\b', r'\bdifficult\b'
            ]
        }
        
        # Score each tone
        tone_scores = {}
        for tone, patterns in tone_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            # Normalize by pattern count and text length
            normalized_score = score / (len(patterns) * max(1, len(text_lower.split()) / 10))
            tone_scores[tone] = normalized_score
        
        # Additional heuristics based on punctuation and structure
        if '?' in text:
            tone_scores['questioning'] += 0.3
        if '!' in text:
            tone_scores['assertive'] += 0.2
            tone_scores['frustrated'] += 0.1
        if text.isupper():
            tone_scores['assertive'] += 0.3
            tone_scores['frustrated'] += 0.2
        
        # Find dominant tone
        if max(tone_scores.values()) > 0:
            dominant_tone = max(tone_scores, key=tone_scores.get)
            confidence = min(0.9, tone_scores[dominant_tone] + 0.5)
        else:
            # Default to assertive if no patterns match
            dominant_tone = 'assertive'
            confidence = 0.5
        
        return {
            "label": dominant_tone,
            "confidence": confidence,
            "scores": tone_scores
        }
    
    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify tone for multiple texts"""
        return [self.classify(text) for text in texts]
