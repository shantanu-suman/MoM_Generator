import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoConfig
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import os

class GrowthTalkModelTrainer:
    """Train custom models for sentiment and tone classification"""
    
    def __init__(self, base_model: str = "distilbert-base-uncased"):
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_training_data(self, meld_data: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Prepare MELD dataset for training
        
        Args:
            meld_data: MELD dataset DataFrame
            
        Returns:
            Tuple of texts and labels
        """
        # Map MELD emotions to our tone categories
        emotion_to_tone_map = {
            'neutral': 0,
            'joy': 1,      # encouraging
            'sadness': 2,   # concerned
            'anger': 3,     # frustrated
            'fear': 4,      # anxious
            'surprise': 5,  # questioning
            'disgust': 6    # frustrated (alternative)
        }
        
        texts = meld_data['Utterance'].tolist()
        
        # Map emotions to tone labels
        labels = []
        for emotion in meld_data['Emotion']:
            labels.append(emotion_to_tone_map.get(emotion.lower(), 0))
        
        return texts, labels
    
    def create_sentiment_labels(self, texts: List[str]) -> List[int]:
        """
        Create sentiment labels using rule-based approach for augmentation
        
        Args:
            texts: List of text utterances
            
        Returns:
            List of sentiment labels (0: negative, 1: neutral, 2: positive)
        """
        positive_indicators = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'happy', 'pleased', 'satisfied', 'success', 'achievement', 'progress'
        ]
        
        negative_indicators = [
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated',
            'angry', 'upset', 'problem', 'issue', 'difficult', 'challenge'
        ]
        
        labels = []
        for text in texts:
            text_lower = text.lower()
            positive_score = sum(1 for word in positive_indicators if word in text_lower)
            negative_score = sum(1 for word in negative_indicators if word in text_lower)
            
            if positive_score > negative_score:
                labels.append(2)  # positive
            elif negative_score > positive_score:
                labels.append(0)  # negative
            else:
                labels.append(1)  # neutral
        
        return labels
    
    def train_sentiment_model(self, texts: List[str], labels: List[int], 
                            output_dir: str = "./models/sentiment"):
        """
        Train sentiment classification model
        
        Args:
            texts: Training texts
            labels: Sentiment labels
            output_dir: Directory to save trained model
        """
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        config = AutoConfig.from_pretrained(self.base_model)
        config.num_labels = 3  # negative, neutral, positive
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, config=config
        )
        
        # Prepare dataset
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Sentiment model saved to {output_dir}")
    
    def train_tone_model(self, texts: List[str], labels: List[int],
                        output_dir: str = "./models/tone"):
        """
        Train tone classification model
        
        Args:
            texts: Training texts
            labels: Tone labels
            output_dir: Directory to save trained model
        """
        # Initialize tokenizer and model for tone classification
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        config = AutoConfig.from_pretrained(self.base_model)
        config.num_labels = 7  # Number of tone categories
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, config=config
        )
        
        # Prepare dataset
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,  # More epochs for tone classification
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Tone model saved to {output_dir}")
    
    def _create_dataset(self, texts: List[str], labels: List[int]):
        """Create PyTorch dataset from texts and labels"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return CustomDataset(encodings, labels)
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def create_synthetic_growth_talk_data(self, num_samples: int = 1000) -> Tuple[List[str], List[int], List[int]]:
        """
        Create synthetic growth talk conversation data
        
        Args:
            num_samples: Number of synthetic samples to create
            
        Returns:
            Tuple of (texts, sentiment_labels, tone_labels)
        """
        import random
        
        # Manager statements templates
        manager_templates = [
            "I'd like to discuss your performance on the recent project",
            "Your work on {project} has been {performance_adj}",
            "We need to focus on improving your {skill} skills",
            "I've noticed some {observation} in your recent work",
            "Let's talk about your career development goals",
            "How do you feel about your progress this quarter?",
            "I want to provide some feedback on your {area} work",
            "Your {strength} skills are really {positive_adj}",
            "Going forward, I'd like you to focus on {improvement_area}",
            "What challenges are you facing in your current role?"
        ]
        
        # Employee responses templates
        employee_templates = [
            "I appreciate the feedback on my work",
            "I've been working hard to improve my {skill} abilities",
            "I'm feeling {emotion} about the recent changes",
            "Could you provide more specific guidance on {area}?",
            "I think I need additional support with {challenge}",
            "I'm confident that I can {action} going forward",
            "Thank you for recognizing my {achievement}",
            "I have some concerns about {concern_area}",
            "I would like to discuss opportunities for {growth_area}",
            "I understand your expectations for {expectation}"
        ]
        
        # Variables for templates
        projects = ["the client presentation", "the data analysis", "the team collaboration", "the quarterly report"]
        performance_adjs = ["excellent", "good", "satisfactory", "needs improvement"]
        skills = ["communication", "technical", "leadership", "analytical", "time management"]
        observations = ["improvements", "challenges", "inconsistencies", "progress"]
        positive_adjs = ["impressive", "strong", "excellent", "outstanding"]
        emotions = ["confident", "anxious", "motivated", "overwhelmed", "excited"]
        
        texts = []
        sentiment_labels = []
        tone_labels = []
        
        for _ in range(num_samples):
            # Randomly choose speaker and template
            if random.random() < 0.5:  # Manager
                template = random.choice(manager_templates)
                # Fill template variables
                text = template.format(
                    project=random.choice(projects),
                    performance_adj=random.choice(performance_adjs),
                    skill=random.choice(skills),
                    observation=random.choice(observations),
                    area=random.choice(skills),
                    positive_adj=random.choice(positive_adjs),
                    strength=random.choice(skills),
                    improvement_area=random.choice(skills)
                )
                
                # Assign likely tone for manager
                tone_label = random.choice([0, 1, 7])  # neutral, encouraging, directive
                
            else:  # Employee
                template = random.choice(employee_templates)
                text = template.format(
                    skill=random.choice(skills),
                    emotion=random.choice(emotions),
                    area=random.choice(skills),
                    challenge=random.choice(skills),
                    action="improve", # simplified
                    achievement=random.choice(skills),
                    concern_area=random.choice(skills),
                    growth_area=random.choice(skills),
                    expectation=random.choice(skills)
                )
                
                # Assign likely tone for employee
                tone_label = random.choice([0, 4, 5])  # neutral, anxious, questioning
            
            texts.append(text)
            
            # Assign sentiment based on keywords
            sentiment_label = self._assign_sentiment_to_synthetic(text)
            sentiment_labels.append(sentiment_label)
            tone_labels.append(tone_label)
        
        return texts, sentiment_labels, tone_labels
    
    def _assign_sentiment_to_synthetic(self, text: str) -> int:
        """Assign sentiment to synthetic text based on keywords"""
        positive_words = ["excellent", "good", "impressive", "strong", "outstanding", "appreciate", "confident", "excited"]
        negative_words = ["improvement", "challenges", "concerns", "anxious", "overwhelmed", "issues"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 2  # positive
        elif negative_count > positive_count:
            return 0  # negative
        else:
            return 1  # neutral
    
    def fine_tune_on_growth_talks(self, base_texts: List[str], base_labels: List[int]):
        """
        Fine-tune models specifically for growth talk conversations
        
        Args:
            base_texts: Base training texts (e.g., from MELD)
            base_labels: Base training labels
        """
        # Generate synthetic growth talk data
        synthetic_texts, synthetic_sentiment, synthetic_tone = self.create_synthetic_growth_talk_data(500)
        
        # Combine with base data
        combined_texts = base_texts + synthetic_texts
        combined_sentiment = base_labels + synthetic_sentiment
        combined_tone = base_labels + synthetic_tone
        
        # Train sentiment model
        print("Training sentiment model with growth talk data...")
        self.train_sentiment_model(combined_texts, combined_sentiment, "./models/growth_talk_sentiment")
        
        # Train tone model
        print("Training tone model with growth talk data...")
        self.train_tone_model(combined_texts, combined_tone, "./models/growth_talk_tone")
