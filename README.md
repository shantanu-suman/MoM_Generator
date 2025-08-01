# Growth Talk Assistant - Local Setup Guide

This guide will help you set up and run the Talk Assistant on your local PC with full AI model training capabilities.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: At least 8GB (16GB recommended for model training)
- **Storage**: 5GB free space for models and datasets
- **OS**: Windows, macOS, or Linux

### Check Python Version
```bash
python --version
# or
python3 --version
```

## Step-by-Step Setup

### 1. Download the Project
Download all project files to your local machine. The project structure should look like:
```
growth-talk-assistant/
├── app.py
├── train_models.py
├── requirements.txt
├── src/
├── models/
├── data/
└── ...
```

### 2. Create Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd growth-talk-assistant

# Create virtual environment
python -m venv gta_env

# Activate virtual environment
# On Windows:
gta_env\Scripts\activate
# On macOS/Linux:
source gta_env/bin/activate
```

### 3. Install Dependencies
Create a `requirements.txt` file with all necessary packages:

```bash
pip install streamlit
pip install transformers
pip install torch
pip install plotly
pip install pandas
pip install numpy
pip install scikit-learn
pip install requests
pip install matplotlib
pip install seaborn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### 4. Create Streamlit Configuration
Create the `.streamlit` directory and config file:

```bash
# Create .streamlit directory
mkdir .streamlit

# Create config.toml file
echo '[server]
headless = true
address = "0.0.0.0"
port = 8501' > .streamlit/config.toml
```

## Model Training Process

### Step 1: Train the Models
Run the complete training pipeline:
```bash
python train_models.py
```

This will:
- Download the MELD dataset (13,708 conversation samples)
- Process and prepare training data
- Train sentiment classification model using DistilBERT
- Train tone classification model using DistilBERT
- Create synthetic growth talk conversation data
- Evaluate model performance
- Save trained models to `./models/` directory

**Training Time**: 30-60 minutes depending on your hardware

### Step 2: Verify Training Results
After training completes, you should see:
```
./models/
├── trained_sentiment/          # Sentiment classification model
├── trained_tone/              # Tone classification model
├── evaluation_report.txt      # Model performance metrics
└── ...

./data/
├── meld/                      # Original MELD dataset
├── processed/                 # Processed training data
└── synthetic/                 # Generated growth talk data
```

## Running the Application

### Start the Streamlit App
```bash
streamlit run app.py
```

The application will start and display:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Access the Web Interface
1. Open your web browser
2. Go to `http://localhost:8501`
3. You'll see the Growth Talk Assistant interface

## Testing the Application

### Sample VTT File for Testing
Create a test file `sample_meeting.vtt`:
```
WEBVTT

00:00:01.000 --> 00:00:05.000
Manager: Good morning! How are you feeling about your current projects?

00:00:06.000 --> 00:00:12.000
Employee: I'm doing well, though I've been struggling with the deadline on the analytics project.

00:00:13.000 --> 00:00:18.000
Manager: I understand. What specific challenges are you facing?

00:00:19.000 --> 00:00:25.000
Employee: The data integration is more complex than expected. I might need an extra week.

00:00:26.000 --> 00:00:32.000
Manager: That's perfectly reasonable. Let's discuss how we can support you better.
```

### Upload and Analyze
1. Upload the VTT file through the web interface
2. View the analysis results including:
   - Sentiment timeline
   - Tone classification
   - Speaker analysis
   - Generated Minutes of Meeting
   - Downloadable reports

## Troubleshooting

### Common Issues and Solutions

#### "No module named 'transformers'"
```bash
pip install transformers torch
```

#### "CUDA out of memory" (during training)
Add this to your training script:
```python
# Use CPU for training if GPU memory is limited
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

#### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

#### Model training too slow
- Use smaller batch sizes
- Train on subset of data
- Use CPU instead of GPU for smaller datasets

### Performance Optimization

#### For Faster Training
```python
# Edit train_models.py and reduce dataset size
# Line ~45, change:
texts, emotion_labels, sentiment_labels = meld_loader.prepare_training_data('train')[:1000]  # Use only 1000 samples
```

#### For Lower Memory Usage
```python
# Use smaller model variants
base_model="distilbert-base-uncased"  # Instead of bert-base-uncased
```

## Understanding the Results

### Model Performance Metrics
After training, check `./models/evaluation_report.txt` for:
- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Balanced precision and recall
- **Confusion Matrix**: Detailed error analysis

### Typical Performance Expectations
- **Sentiment Analysis**: 75-85% accuracy
- **Tone Classification**: 60-75% accuracy
- **Training Time**: 30-60 minutes on modern CPU

## Retraining Models

To retrain with new data:
```bash
# Remove existing models
rm -rf ./models/trained_*

# Run training again
python train_models.py
```

## Project Structure After Setup
```
growth-talk-assistant/
├── .streamlit/config.toml     # Streamlit configuration
├── app.py                     # Main web application
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── gta_env/                  # Virtual environment
├── src/                      # Source code modules
├── models/                   # Trained models
│   ├── trained_sentiment/
│   ├── trained_tone/
│   └── evaluation_report.txt
├── data/                     # Datasets
│   ├── meld/                # MELD dataset
│   ├── processed/           # Processed data
│   └── synthetic/           # Generated data
└── sample_meeting.vtt       # Test file
```

## Next Steps

1. **Complete Setup**: Follow all installation steps
2. **Train Models**: Run the training pipeline
3. **Test Application**: Upload sample VTT files
4. **Customize**: Modify models for your specific use case
5. **Deploy**: Consider cloud deployment for team access

## Tips for Success

- **First Run**: Start with the sample VTT file to verify everything works
- **GPU Training**: If you have an NVIDIA GPU, install `torch` with CUDA support for faster training
- **Custom Data**: Add your own conversation data to improve model performance
- **Regular Updates**: Retrain models periodically with new conversation data

Your Growth Talk Assistant will be fully functional with AI-powered analysis once you complete these steps!
