import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import zipfile

from src.transcript_parser import VTTParser
from src.sentiment_classifier import SentimentClassifier
from src.tone_classifier import ToneClassifier
from src.mom_generator import MoMGenerator
from src.report_generator import ReportGenerator
from src.visualizations import create_sentiment_timeline, create_tone_distribution, create_emotional_flow
from src.utils import validate_vtt_file

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None

def main():
    st.title("🎯 Growth Talk Assistant (GTA)")
    st.markdown("Analyze manager-employee conversations with AI-powered sentiment and tone analysis")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.info("Upload a .vtt transcript file to begin analysis")
        
        # Model selection (for future use)
        sentiment_model = st.selectbox(
            "Sentiment Model",
            ["DistilBERT-base", "RoBERTa-base"],
            help="Select the transformer model for sentiment analysis"
        )
        
        tone_model = st.selectbox(
            "Tone Model", 
            ["DistilBERT-tone", "RoBERTa-tone"],
            help="Select the model for tone classification"
        )
        
        summarization_model = st.selectbox(
            "Summarization Model",
            ["T5-base", "BART-base"],
            help="Select the model for MoM generation"
        )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Analyze", "📊 Analysis Results", "📝 Minutes of Meeting", "📈 Visualizations"])
    
    with tab1:
        st.header("Upload Transcript File")
        
        uploaded_file = st.file_uploader(
            "Choose a .vtt transcript file",
            type=['vtt'],
            help="Upload a WebVTT transcript file from manager-employee conversation"
        )
        
        if uploaded_file is not None:
            # Validate file
            if validate_vtt_file(uploaded_file):
                st.success("✅ Valid VTT file uploaded")
                
                # Parse transcript
                with st.spinner("Parsing transcript..."):
                    parser = VTTParser()
                    transcript_data = parser.parse_vtt(uploaded_file)
                    st.session_state.transcript_data = transcript_data
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Utterances", len(transcript_data))
                with col2:
                    managers = sum(1 for item in transcript_data if item['speaker'] == 'Manager')
                    st.metric("Manager Utterances", managers)
                with col3:
                    employees = sum(1 for item in transcript_data if item['speaker'] == 'Employee')
                    st.metric("Employee Utterances", employees)
                
                # Analyze button
                if st.button("🔍 Start Analysis", type="primary"):
                    analyze_transcript(transcript_data, sentiment_model, tone_model, summarization_model)
            else:
                st.error("❌ Invalid VTT file format. Please upload a valid WebVTT file.")
    
    with tab2:
        st.header("Analysis Results")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Overview metrics
            st.subheader("📈 Analysis Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_sentiment = results['sentiment_stats']['average_sentiment']
                st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")
            
            with col2:
                positive_ratio = results['sentiment_stats']['positive_ratio']
                st.metric("Positive Sentiment %", f"{positive_ratio:.1f}%")
            
            with col3:
                dominant_tone = results['tone_stats']['dominant_tone']
                st.metric("Dominant Tone", dominant_tone)
            
            with col4:
                total_duration = results.get('total_duration', 0)
                st.metric("Duration (min)", f"{total_duration:.1f}")
            
            # Detailed results table
            st.subheader("📋 Utterance-level Analysis")
            df = pd.DataFrame(results['utterances'])
            st.dataframe(
                df[['timestamp', 'speaker', 'text', 'sentiment', 'sentiment_score', 'tone', 'tone_confidence']],
                use_container_width=True
            )
            
            # Download options
            st.subheader("📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generate Text Report"):
                    report_generator = ReportGenerator()
                    report_text = report_generator.generate_text_report(results)
                    st.download_button(
                        label="Download Text Report",
                        data=report_text,
                        file_name=f"gta_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📊 Generate CSV Export"):
                    csv_data = pd.DataFrame(results['utterances']).to_csv(index=False)
                    st.download_button(
                        label="Download CSV Data",
                        data=csv_data,
                        file_name=f"gta_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("👆 Please upload and analyze a transcript file first")
    
    with tab3:
        st.header("Minutes of Meeting (MoM)")
        
        if st.session_state.analysis_results is not None:
            mom_data = st.session_state.analysis_results.get('mom_summary')
            
            if mom_data:
                # Key Discussion Points
                st.subheader("🎯 Key Discussion Points")
                for i, point in enumerate(mom_data['key_points'], 1):
                    st.write(f"{i}. {point}")
                
                # Action Items
                st.subheader("✅ Action Items")
                if mom_data['action_items']:
                    for i, action in enumerate(mom_data['action_items'], 1):
                        st.write(f"{i}. {action}")
                else:
                    st.info("No specific action items identified")
                
                # Emotional Flow Summary
                st.subheader("💭 Emotional Flow")
                st.write(mom_data['emotional_summary'])
                
                # Tone Trends
                st.subheader("🎭 Tone Analysis")
                st.write(mom_data['tone_summary'])
                
                # Meeting Outcome
                st.subheader("🎯 Meeting Outcome")
                outcome_color = "green" if mom_data['overall_sentiment'] == "positive" else "orange" if mom_data['overall_sentiment'] == "neutral" else "red"
                st.markdown(f"**Overall Sentiment:** :{outcome_color}[{mom_data['overall_sentiment'].title()}]")
                st.write(mom_data['meeting_outcome'])
            else:
                st.info("MoM summary will appear here after analysis")
        else:
            st.info("👆 Please upload and analyze a transcript file first")
    
    with tab4:
        st.header("Visualizations")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Sentiment Timeline
            st.subheader("📈 Sentiment Timeline")
            sentiment_fig = create_sentiment_timeline(results['utterances'])
            st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Tone Distribution
            st.subheader("🎭 Tone Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                tone_fig = create_tone_distribution(results['utterances'])
                st.plotly_chart(tone_fig, use_container_width=True)
            
            with col2:
                # Speaker comparison
                speaker_sentiment = {}
                for utterance in results['utterances']:
                    speaker = utterance['speaker']
                    if speaker not in speaker_sentiment:
                        speaker_sentiment[speaker] = []
                    speaker_sentiment[speaker].append(utterance['sentiment_score'])
                
                speaker_avg = {speaker: sum(scores)/len(scores) for speaker, scores in speaker_sentiment.items()}
                
                fig_speaker = go.Figure(data=[
                    go.Bar(x=list(speaker_avg.keys()), y=list(speaker_avg.values()))
                ])
                fig_speaker.update_layout(title="Average Sentiment by Speaker")
                st.plotly_chart(fig_speaker, use_container_width=True)
            
            # Emotional Flow
            st.subheader("🌊 Emotional Flow")
            flow_fig = create_emotional_flow(results['utterances'])
            st.plotly_chart(flow_fig, use_container_width=True)
        else:
            st.info("👆 Please upload and analyze a transcript file first")

def analyze_transcript(transcript_data, sentiment_model, tone_model, summarization_model):
    """Perform complete analysis of the transcript"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize models
        status_text.text("Initializing models...")
        progress_bar.progress(10)
        
        sentiment_classifier = SentimentClassifier(model_name=sentiment_model)
        tone_classifier = ToneClassifier(model_name=tone_model)
        mom_generator = MoMGenerator(model_name=summarization_model)
        
        # Perform sentiment analysis
        status_text.text("Analyzing sentiment...")
        progress_bar.progress(30)
        
        analyzed_utterances = []
        for utterance in transcript_data:
            sentiment_result = sentiment_classifier.classify(utterance['text'])
            tone_result = tone_classifier.classify(utterance['text'])
            
            analyzed_utterance = {
                **utterance,
                'sentiment': sentiment_result['label'],
                'sentiment_score': sentiment_result['score'],
                'tone': tone_result['label'],
                'tone_confidence': tone_result['confidence']
            }
            analyzed_utterances.append(analyzed_utterance)
        
        progress_bar.progress(60)
        status_text.text("Generating Minutes of Meeting...")
        
        # Generate MoM
        mom_summary = mom_generator.generate_summary(analyzed_utterances)
        
        progress_bar.progress(80)
        status_text.text("Computing statistics...")
        
        # Compute statistics
        sentiment_scores = [u['sentiment_score'] for u in analyzed_utterances]
        sentiment_labels = [u['sentiment'] for u in analyzed_utterances]
        tone_labels = [u['tone'] for u in analyzed_utterances]
        
        sentiment_stats = {
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores),
            'positive_ratio': (sentiment_labels.count('positive') / len(sentiment_labels)) * 100,
            'neutral_ratio': (sentiment_labels.count('neutral') / len(sentiment_labels)) * 100,
            'negative_ratio': (sentiment_labels.count('negative') / len(sentiment_labels)) * 100
        }
        
        from collections import Counter
        tone_counts = Counter(tone_labels)
        tone_stats = {
            'dominant_tone': tone_counts.most_common(1)[0][0],
            'tone_distribution': dict(tone_counts)
        }
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results
        st.session_state.analysis_results = {
            'utterances': analyzed_utterances,
            'mom_summary': mom_summary,
            'sentiment_stats': sentiment_stats,
            'tone_stats': tone_stats,
            'total_duration': sum(float(u.get('duration', 0)) for u in analyzed_utterances) / 60.0
        }
        
        st.success("✅ Analysis completed successfully!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()
