try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available, using fallback sentiment analysis")

import numpy as np
from typing import Dict, Any

class SentimentClassifier:
    """Sentiment classification using transformer models"""
    
    def __init__(self, model_name: str = "DistilBERT-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained sentiment analysis model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, using rule-based sentiment analysis.")
            self.model = None
            self.tokenizer = None
            return
            
        try:
            # Map friendly names to actual model names
            model_mapping = {
                "DistilBERT-base": "distilbert-base-uncased-finetuned-sst-2-english",
                "RoBERTa-base": "cardiffnlp/twitter-roberta-base-sentiment-latest"
            }
            
            actual_model_name = model_mapping.get(self.model_name, "distilbert-base-uncased-finetuned-sst-2-english")
            
            self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(actual_model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            # Fallback to a simple rule-based approach if model loading fails
            print(f"Warning: Could not load transformer model ({e}). Using fallback sentiment analysis.")
            self.model = None
            self.tokenizer = None
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify sentiment of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract results
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
            # Map to sentiment labels
            if "roberta" in self.model_name.lower():
                label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
            else:
                label_mapping = {0: "negative", 1: "positive"}
                # For binary models, add neutral zone
                if confidence < 0.7:
                    sentiment_label = "neutral"
                    sentiment_score = 0.5
                else:
                    sentiment_label = label_mapping[predicted_class]
                    # Convert to -1 to 1 scale
                    sentiment_score = confidence if predicted_class == 1 else -confidence
            
            if "roberta" in self.model_name.lower():
                sentiment_label = label_mapping[predicted_class]
                # Convert to -1 to 1 scale
                if sentiment_label == "positive":
                    sentiment_score = confidence
                elif sentiment_label == "negative":
                    sentiment_score = -confidence
                else:
                    sentiment_score = 0.0
            
            return {
                "label": sentiment_label,
                "score": sentiment_score,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error in sentiment classification: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis as fallback"""
        
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'positive', 'happy', 'pleased', 'satisfied', 'success', 'achievement',
            'progress', 'improvement', 'well', 'better', 'best', 'outstanding',
            'effective', 'efficient', 'helpful', 'supportive', 'encouraging'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated',
            'angry', 'upset', 'concerned', 'worried', 'problem', 'issue',
            'difficult', 'challenge', 'struggling', 'failed', 'worse', 'worst',
            'ineffective', 'inefficient', 'unhelpful', 'unsupportive', 'discouraging'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            score = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
            return {"label": "positive", "score": score, "confidence": score}
        elif negative_count > positive_count:
            score = max(-0.8, -0.5 - (negative_count - positive_count) * 0.1)
            return {"label": "negative", "score": score, "confidence": abs(score)}
        else:
            return {"label": "neutral", "score": 0.0, "confidence": 0.6}
    
    def batch_classify(self, texts: list) -> list:
        """Classify sentiment for multiple texts"""
        return [self.classify(text) for text in texts]
