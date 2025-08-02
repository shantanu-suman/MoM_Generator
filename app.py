# Updated app.py for Growth Talk Assistant (GTA) using refactored visualizations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter

from src.transcript_parser import VTTParser
from src.sentiment_classifier import SentimentClassifier
from src.tone_classifier import ToneClassifier
from src.mom_generator import MoMGenerator
from src.report_generator import ReportGenerator
from src.visualizations import create_wordcloud_image, display_dashboard
from src.utils import validate_vtt_file

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None

def main():
    st.set_page_config(page_title="Growth Talk Assistant", layout="wide")
    st.title("🎯 Growth Talk Assistant (GTA)")
    st.markdown("Analyze manager-employee conversations with AI-powered sentiment and tone analysis")

    with st.sidebar:
        st.header("Configuration")
        st.info("Upload a .vtt transcript file to begin analysis")

        sentiment_model = st.selectbox("Sentiment Model", ["DistilBERT-base", "RoBERTa-base"])
        tone_model = st.selectbox("Tone Model", ["DistilBERT-tone", "RoBERTa-tone"])
        summarization_model = st.selectbox("Summarization Model", ["T5-base", "BART-base"])

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Analyze", "📊 Analysis Results", "📝 Minutes of Meeting", "📈 Visualizations"])

    with tab1:
        st.header("Upload Transcript File")
        uploaded_file = st.file_uploader("Choose a .vtt transcript file", type=['vtt'])

        if uploaded_file and validate_vtt_file(uploaded_file):
            st.success("✅ Valid VTT file uploaded")

            with st.spinner("Parsing transcript..."):
                parser = VTTParser()
                transcript_data = parser.parse_vtt(uploaded_file)
                st.session_state.transcript_data = transcript_data

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Utterances", len(transcript_data))
            col2.metric("Manager Utterances", sum(1 for u in transcript_data if u['speaker'] == 'Manager'))
            col3.metric("Employee Utterances", sum(1 for u in transcript_data if u['speaker'] == 'Employee'))

            if st.button("🔍 Start Analysis", type="primary"):
                analyze_transcript(transcript_data, sentiment_model, tone_model, summarization_model)

        elif uploaded_file:
            st.error("❌ Invalid VTT file format. Please upload a valid WebVTT file.")

    with tab2:
        st.header("Analysis Results")
        results = st.session_state.analysis_results

        if results:
            st.subheader("📈 Analysis Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Sentiment (–1 to +1)", f"{results['sentiment_stats']['average_sentiment']:.2f}")
            col2.metric("Positive Sentiment %", f"{results['sentiment_stats']['positive_ratio']:.1f}%")
            col3.metric("Dominant Tone", results['tone_stats']['dominant_tone'])
            col4.metric("Duration (min)", f"{results.get('total_duration', 0):.1f}")

            st.subheader("📋 Utterance-level Analysis")
            df = pd.DataFrame(results['utterances'])
            st.dataframe(df[['timestamp', 'speaker', 'text', 'sentiment', 'sentiment_score', 'tone', 'tone_confidence']], use_container_width=True)

            st.subheader("📥 Download Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Generate Text Report"):
                    report = ReportGenerator().generate_text_report(results)
                    st.download_button("Download Text Report", data=report, file_name="gta_report.txt", mime="text/plain")

            with col2:
                if st.button("📊 Generate CSV Export"):
                    csv = pd.DataFrame(results['utterances']).to_csv(index=False)
                    st.download_button("Download CSV Data", data=csv, file_name="gta_data.csv", mime="text/csv")
        else:
            st.info("👆 Please upload and analyze a transcript file first")

    with tab3:
        st.header("Minutes of Meeting (MoM)")
        results = st.session_state.analysis_results

        if results:
            mom = results.get('mom_summary')
            if mom:
                st.subheader("🎯 Key Discussion Points")
                for i, point in enumerate(mom['key_points'], 1):
                    st.write(f"{i}. {point}")

                st.subheader("✅ Action Items")
                for i, action in enumerate(mom['action_items'], 1):
                    st.write(f"{i}. {action}")

                st.subheader("💭 Emotional Flow")
                st.write(mom['emotional_summary'])

                st.subheader("🎭 Tone Analysis")
                st.write(mom['tone_summary'])

                st.subheader("🎯 Meeting Outcome")
                sentiment_color = "green" if mom['overall_sentiment'] == "positive" else "orange" if mom['overall_sentiment'] == "neutral" else "red"
                st.markdown(f"**Overall Sentiment:** :{sentiment_color}[{mom['overall_sentiment'].title()}]")
                st.write(mom['meeting_outcome'])
            else:
                st.info("MoM summary will appear here after analysis")
        else:
            st.info("👆 Please upload and analyze a transcript file first")

    with tab4:
        st.header("📈 Visualizations")
        results = st.session_state.analysis_results

        if results:
            display_dashboard(results)
        else:
            st.info("👆 Please upload and analyze a transcript file first")

def analyze_transcript(transcript_data, sentiment_model, tone_model, summarization_model):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Initializing models...")
        progress_bar.progress(10)

        sentiment_classifier = SentimentClassifier(model_name=sentiment_model)
        tone_classifier = ToneClassifier(model_name=tone_model)
        mom_generator = MoMGenerator(model_name=summarization_model)

        analyzed_utterances = []
        for utterance in transcript_data:
            sentiment = sentiment_classifier.classify(utterance['text'])
            tone = tone_classifier.classify(utterance['text'])

            analyzed_utterances.append({
                **utterance,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'tone': tone['label'],
                'tone_confidence': tone['confidence']
            })

        progress_bar.progress(60)
        status_text.text("Generating Minutes of Meeting...")
        mom_summary = mom_generator.generate_summary(analyzed_utterances)

        sentiment_scores = [u['sentiment_score'] for u in analyzed_utterances]
        sentiment_labels = [u['sentiment'] for u in analyzed_utterances]
        tone_labels = [u['tone'] for u in analyzed_utterances]

        sentiment_stats = {
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores),
            'positive_ratio': (sentiment_labels.count('positive') / len(sentiment_labels)) * 100,
            'neutral_ratio': (sentiment_labels.count('neutral') / len(sentiment_labels)) * 100,
            'negative_ratio': (sentiment_labels.count('negative') / len(sentiment_labels)) * 100,
        }

        tone_counts = Counter(tone_labels)
        tone_stats = {
            'dominant_tone': tone_counts.most_common(1)[0][0],
            'tone_distribution': dict(tone_counts)
        }

        progress_bar.progress(100)
        status_text.text("Analysis complete!")

        st.session_state.analysis_results = {
            'utterances': analyzed_utterances,
            'mom_summary': mom_summary,
            'sentiment_stats': sentiment_stats,
            'tone_stats': tone_stats,
            'total_duration': sum(float(u.get('duration', 0)) for u in analyzed_utterances) / 60.0
        }

        st.success("✅ Analysis completed successfully!")
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()