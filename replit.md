# Growth Talk Assistant (GTA)

## Overview

The Growth Talk Assistant is a Streamlit-based web application designed to analyze manager-employee conversations from VTT transcript files. It provides AI-powered sentiment analysis, tone classification, automated Minutes of Meeting (MoM) generation, and comprehensive reporting capabilities. The system is built using a modular architecture with separate components for parsing, analysis, visualization, and reporting.

## User Preferences

Preferred communication style: Simple, everyday language.
Local PC setup: User wants complete instructions for running the system locally with full model training capabilities.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web application (`app.py`) providing an intuitive user interface
- **Core Processing**: Modular components in the `src/` directory handling different aspects of analysis
- **Model Training**: Separate training pipeline in the `models/` directory for custom model development
- **Data Processing**: MELD dataset integration for model training and evaluation

The architecture is designed to be extensible, allowing for easy integration of new models and analysis features.

## Key Components

### Transcript Processing
- **VTTParser** (`src/transcript_parser.py`): Parses WebVTT files and extracts structured utterances with timestamps and speaker identification
- **File Validation** (`src/utils.py`): Validates uploaded VTT files and provides text cleaning utilities

### AI Analysis Engine
- **SentimentClassifier** (`src/sentiment_classifier.py`): Uses transformer models (DistilBERT/RoBERTa) for sentiment analysis with fallback to rule-based classification
- **ToneClassifier** (`src/tone_classifier.py`): Classifies emotional tone using emotion-to-tone mapping with multiple tone categories (empathetic, assertive, anxious, etc.)

### Content Generation
- **MoMGenerator** (`src/mom_generator.py`): Generates structured Minutes of Meeting using T5/BART models for summarization
- **ReportGenerator** (`src/report_generator.py`): Creates comprehensive downloadable reports in text format

### Visualization
- **Visualizations** (`src/visualizations.py`): Creates interactive Plotly charts including sentiment timelines, tone distributions, and emotional flow analysis

### Model Training Pipeline
- **ModelTrainer** (`models/model_trainer.py`): Custom training pipeline for sentiment and tone classification models
- **ModelEvaluator** (`models/evaluation.py`): Comprehensive evaluation metrics and model performance analysis
- **MELD Dataset Integration** (`data/meld_loader.py`): Handles loading and processing of the MELD dataset for training

## Data Flow

1. **File Upload**: User uploads VTT transcript file through Streamlit interface
2. **Validation**: System validates file format and content structure
3. **Parsing**: VTT parser extracts utterances with timestamps and speaker identification
4. **Analysis**: Each utterance is processed through sentiment and tone classifiers
5. **Aggregation**: Results are combined and statistical summaries are generated
6. **Visualization**: Interactive charts are created showing sentiment trends and tone distributions
7. **Report Generation**: Structured MoM and comprehensive reports are generated
8. **Output**: Results are displayed in the web interface with download options

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **Transformers**: Hugging Face library for pre-trained models (DistilBERT, RoBERTa, T5, BART)
- **PyTorch**: Deep learning framework for model inference
- **Plotly**: Interactive visualization library for charts and graphs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities and evaluation metrics

### Pre-trained Models
- **Sentiment Analysis**: DistilBERT-SST2, CardiffNLP RoBERTa-sentiment
- **Emotion Classification**: j-hartmann/emotion-english-distilroberta-base
- **Summarization**: T5-small, BART-large-CNN

### Dataset Integration
- **MELD Dataset**: Multimodal EmotionLines Dataset for training custom models

## Deployment Strategy

The application is designed for Replit deployment with the following characteristics:

- **Single-file Entry Point**: `app.py` serves as the main application entry point
- **Modular Structure**: All components are organized in separate modules for maintainability
- **Resource Optimization**: Uses lightweight model variants (T5-small) for better performance in constrained environments
- **Graceful Degradation**: Fallback mechanisms ensure functionality even if some models fail to load
- **Session State Management**: Streamlit session state maintains analysis results across interactions

The system includes comprehensive error handling and fallback mechanisms to ensure reliability in various deployment environments. Model loading is optimized for memory constraints, and the interface provides clear feedback to users throughout the analysis process.