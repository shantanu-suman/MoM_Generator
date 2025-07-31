import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelEvaluator:
    """Evaluate trained models for sentiment and tone classification"""
    
    def __init__(self):
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        self.tone_labels = ['Neutral', 'Encouraging', 'Concerned', 'Frustrated', 
                          'Anxious', 'Questioning', 'Frustrated_Alt', 'Directive']
    
    def evaluate_sentiment_model(self, model_path: str, test_texts: List[str], 
                                test_labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate sentiment classification model
        
        Args:
            model_path: Path to trained model
            test_texts: Test text samples
            test_labels: True labels for test samples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            # Make predictions
            predictions = []
            
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                    predictions.append(predicted_class)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                test_labels, predictions, average='weighted'
            )
            
            # Per-class metrics
            per_class_metrics = precision_recall_fscore_support(
                test_labels, predictions, average=None
            )
            
            # Confusion matrix
            cm = confusion_matrix(test_labels, predictions)
            
            # Classification report
            class_report = classification_report(
                test_labels, predictions, 
                target_names=self.sentiment_labels[:max(test_labels)+1],
                output_dict=True
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_precision': per_class_metrics[0].tolist(),
                'per_class_recall': per_class_metrics[1].tolist(),
                'per_class_f1': per_class_metrics[2].tolist(),
                'support': per_class_metrics[3].tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error evaluating sentiment model: {e}")
            return self._fallback_evaluation(test_labels, predictions if 'predictions' in locals() else [])
    
    def evaluate_tone_model(self, model_path: str, test_texts: List[str],
                           test_labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate tone classification model
        
        Args:
            model_path: Path to trained model
            test_texts: Test text samples
            test_labels: True labels for test samples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            # Make predictions
            predictions = []
            
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                 padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                    predictions.append(predicted_class)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                test_labels, predictions, average='weighted'
            )
            
            # Per-class metrics
            per_class_metrics = precision_recall_fscore_support(
                test_labels, predictions, average=None
            )
            
            # Confusion matrix
            cm = confusion_matrix(test_labels, predictions)
            
            # Classification report
            available_labels = self.tone_labels[:max(max(test_labels), max(predictions))+1]
            class_report = classification_report(
                test_labels, predictions,
                target_names=available_labels,
                output_dict=True
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_precision': per_class_metrics[0].tolist(),
                'per_class_recall': per_class_metrics[1].tolist(),
                'per_class_f1': per_class_metrics[2].tolist(),
                'support': per_class_metrics[3].tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error evaluating tone model: {e}")
            return self._fallback_evaluation(test_labels, predictions if 'predictions' in locals() else [])
    
    def evaluate_summarization_quality(self, generated_summaries: List[str],
                                     reference_summaries: List[str]) -> Dict[str, float]:
        """
        Evaluate summarization quality using ROUGE scores
        
        Args:
            generated_summaries: AI-generated summaries
            reference_summaries: Reference/gold standard summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            # Simple ROUGE-like implementation
            # In production, use proper ROUGE library
            rouge_scores = {
                'rouge_1_precision': 0.0,
                'rouge_1_recall': 0.0,
                'rouge_1_f1': 0.0,
                'rouge_2_precision': 0.0,
                'rouge_2_recall': 0.0,
                'rouge_2_f1': 0.0,
                'rouge_l_precision': 0.0,
                'rouge_l_recall': 0.0,
                'rouge_l_f1': 0.0
            }
            
            for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
                # Calculate word overlap (simplified ROUGE-1)
                gen_words = set(gen_summary.lower().split())
                ref_words = set(ref_summary.lower().split())
                
                if len(gen_words) > 0 and len(ref_words) > 0:
                    overlap = len(gen_words.intersection(ref_words))
                    
                    precision = overlap / len(gen_words)
                    recall = overlap / len(ref_words)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    rouge_scores['rouge_1_precision'] += precision
                    rouge_scores['rouge_1_recall'] += recall
                    rouge_scores['rouge_1_f1'] += f1
            
            # Average scores
            num_samples = len(generated_summaries)
            if num_samples > 0:
                for key in rouge_scores:
                    rouge_scores[key] /= num_samples
            
            return rouge_scores
            
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {key: 0.0 for key in rouge_scores.keys()}
    
    def create_evaluation_report(self, sentiment_metrics: Dict[str, Any],
                               tone_metrics: Dict[str, Any],
                               rouge_scores: Dict[str, float]) -> str:
        """
        Create comprehensive evaluation report
        
        Args:
            sentiment_metrics: Sentiment model evaluation results
            tone_metrics: Tone model evaluation results
            rouge_scores: Summarization ROUGE scores
            
        Returns:
            Formatted evaluation report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Sentiment Model Results
        report_lines.append("SENTIMENT CLASSIFICATION MODEL")
        report_lines.append("-" * 32)
        report_lines.append(f"Accuracy: {sentiment_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"Precision: {sentiment_metrics.get('precision', 0):.4f}")
        report_lines.append(f"Recall: {sentiment_metrics.get('recall', 0):.4f}")
        report_lines.append(f"F1-Score: {sentiment_metrics.get('f1_score', 0):.4f}")
        report_lines.append("")
        
        # Per-class sentiment metrics
        if 'per_class_f1' in sentiment_metrics:
            report_lines.append("Per-class F1 Scores:")
            for i, f1 in enumerate(sentiment_metrics['per_class_f1']):
                if i < len(self.sentiment_labels):
                    report_lines.append(f"  {self.sentiment_labels[i]}: {f1:.4f}")
        report_lines.append("")
        
        # Tone Model Results
        report_lines.append("TONE CLASSIFICATION MODEL")
        report_lines.append("-" * 26)
        report_lines.append(f"Accuracy: {tone_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"Precision: {tone_metrics.get('precision', 0):.4f}")
        report_lines.append(f"Recall: {tone_metrics.get('recall', 0):.4f}")
        report_lines.append(f"F1-Score: {tone_metrics.get('f1_score', 0):.4f}")
        report_lines.append("")
        
        # Per-class tone metrics
        if 'per_class_f1' in tone_metrics:
            report_lines.append("Per-class F1 Scores:")
            for i, f1 in enumerate(tone_metrics['per_class_f1']):
                if i < len(self.tone_labels):
                    report_lines.append(f"  {self.tone_labels[i]}: {f1:.4f}")
        report_lines.append("")
        
        # Summarization Results
        report_lines.append("SUMMARIZATION MODEL (ROUGE SCORES)")
        report_lines.append("-" * 34)
        report_lines.append(f"ROUGE-1 Precision: {rouge_scores.get('rouge_1_precision', 0):.4f}")
        report_lines.append(f"ROUGE-1 Recall: {rouge_scores.get('rouge_1_recall', 0):.4f}")
        report_lines.append(f"ROUGE-1 F1: {rouge_scores.get('rouge_1_f1', 0):.4f}")
        report_lines.append("")
        
        # Overall Assessment
        report_lines.append("OVERALL ASSESSMENT")
        report_lines.append("-" * 18)
        
        sentiment_quality = self._assess_model_quality(sentiment_metrics.get('f1_score', 0))
        tone_quality = self._assess_model_quality(tone_metrics.get('f1_score', 0))
        summarization_quality = self._assess_model_quality(rouge_scores.get('rouge_1_f1', 0))
        
        report_lines.append(f"Sentiment Model Quality: {sentiment_quality}")
        report_lines.append(f"Tone Model Quality: {tone_quality}")
        report_lines.append(f"Summarization Quality: {summarization_quality}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 15)
        recommendations = self._generate_recommendations(sentiment_metrics, tone_metrics, rouge_scores)
        for rec in recommendations:
            report_lines.append(f"• {rec}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def _assess_model_quality(self, f1_score: float) -> str:
        """Assess model quality based on F1 score"""
        if f1_score >= 0.9:
            return "Excellent"
        elif f1_score >= 0.8:
            return "Good"
        elif f1_score >= 0.7:
            return "Fair"
        elif f1_score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_recommendations(self, sentiment_metrics: Dict[str, Any],
                                tone_metrics: Dict[str, Any],
                                rouge_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        # Sentiment model recommendations
        if sentiment_metrics.get('f1_score', 0) < 0.8:
            recommendations.append("Consider fine-tuning sentiment model with more domain-specific data")
        
        if sentiment_metrics.get('accuracy', 0) < 0.75:
            recommendations.append("Increase training data size for sentiment classification")
        
        # Tone model recommendations
        if tone_metrics.get('f1_score', 0) < 0.7:
            recommendations.append("Tone classification needs improvement - consider data augmentation")
        
        # Check class imbalance
        if 'per_class_f1' in tone_metrics:
            min_f1 = min(tone_metrics['per_class_f1'])
            if min_f1 < 0.5:
                recommendations.append("Address class imbalance in tone training data")
        
        # Summarization recommendations
        if rouge_scores.get('rouge_1_f1', 0) < 0.3:
            recommendations.append("Summarization model requires significant improvement")
        
        if rouge_scores.get('rouge_1_f1', 0) < 0.5:
            recommendations.append("Consider using larger pre-trained models for summarization")
        
        if not recommendations:
            recommendations.append("Models are performing well - consider minor hyperparameter tuning")
        
        return recommendations
    
    def _fallback_evaluation(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        """Fallback evaluation when model loading fails"""
        if not predictions:
            predictions = [0] * len(true_labels)  # All predictions as class 0
        
        try:
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'error': 'Model evaluation failed - using fallback metrics'
            }
        except Exception:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': 'Complete evaluation failure'
            }
    
    def cross_validate_model(self, texts: List[str], labels: List[int], 
                           model_type: str = 'sentiment', cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on model
        
        Args:
            texts: Input texts
            labels: True labels
            model_type: Type of model ('sentiment' or 'tone')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        # This is a simplified implementation
        # In practice, you'd need to implement proper CV for transformer models
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            
            # Use simple baseline for CV evaluation
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            # Perform cross-validation
            cv_scores = cross_val_score(classifier, X, labels, cv=cv_folds, scoring='f1_weighted')
            
            return {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return {
                'mean_cv_score': 0.0,
                'std_cv_score': 0.0,
                'cv_scores': [0.0] * cv_folds,
                'error': str(e)
            }
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], 
                            labels: List[str], title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix as nested list
            labels: Class labels
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
