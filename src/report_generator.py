from datetime import datetime
from typing import Dict, Any, List
import json

class ReportGenerator:
    """Generate downloadable reports from analysis results"""
    
    def __init__(self):
        pass
    
    def generate_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive text report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("GROWTH TALK ASSISTANT - ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        
        stats = analysis_results.get('sentiment_stats', {})
        tone_stats = analysis_results.get('tone_stats', {})
        mom = analysis_results.get('mom_summary', {})
        
        report_lines.append(f"Overall Sentiment: {mom.get('overall_sentiment', 'N/A').title()}")
        report_lines.append(f"Average Sentiment Score: {stats.get('average_sentiment', 0):.2f}")
        report_lines.append(f"Dominant Tone: {tone_stats.get('dominant_tone', 'N/A').title()}")
        report_lines.append(f"Total Utterances: {len(analysis_results.get('utterances', []))}")
        report_lines.append("")
        
        # Meeting Outcome
        if mom.get('meeting_outcome'):
            report_lines.append("MEETING OUTCOME")
            report_lines.append("-" * 15)
            report_lines.append(mom['meeting_outcome'])
            report_lines.append("")
        
        # Key Discussion Points
        if mom.get('key_points'):
            report_lines.append("KEY DISCUSSION POINTS")
            report_lines.append("-" * 22)
            for i, point in enumerate(mom['key_points'], 1):
                report_lines.append(f"{i}. {point}")
            report_lines.append("")
        
        # Action Items
        if mom.get('action_items'):
            report_lines.append("ACTION ITEMS")
            report_lines.append("-" * 12)
            for i, action in enumerate(mom['action_items'], 1):
                report_lines.append(f"{i}. {action}")
            report_lines.append("")
        
        # Emotional Analysis
        report_lines.append("EMOTIONAL ANALYSIS")
        report_lines.append("-" * 18)
        report_lines.append(f"Emotional Flow: {mom.get('emotional_summary', 'N/A')}")
        report_lines.append(f"Tone Trends: {mom.get('tone_summary', 'N/A')}")
        report_lines.append("")
        
        # Sentiment Breakdown
        report_lines.append("SENTIMENT BREAKDOWN")
        report_lines.append("-" * 19)
        report_lines.append(f"Positive: {stats.get('positive_ratio', 0):.1f}%")
        report_lines.append(f"Neutral: {stats.get('neutral_ratio', 0):.1f}%")
        report_lines.append(f"Negative: {stats.get('negative_ratio', 0):.1f}%")
        report_lines.append("")
        
        # Tone Distribution
        if tone_stats.get('tone_distribution'):
            report_lines.append("TONE DISTRIBUTION")
            report_lines.append("-" * 17)
            for tone, count in tone_stats['tone_distribution'].items():
                percentage = (count / len(analysis_results.get('utterances', []))) * 100
                report_lines.append(f"{tone.title()}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Detailed Utterance Analysis
        report_lines.append("DETAILED UTTERANCE ANALYSIS")
        report_lines.append("-" * 28)
        report_lines.append("")
        
        utterances = analysis_results.get('utterances', [])
        for i, utterance in enumerate(utterances, 1):
            timestamp = utterance.get('timestamp', 'N/A')
            speaker = utterance.get('speaker', 'Unknown')
            text = utterance.get('text', '')
            sentiment = utterance.get('sentiment', 'neutral')
            sentiment_score = utterance.get('sentiment_score', 0)
            tone = utterance.get('tone', 'neutral')
            tone_confidence = utterance.get('tone_confidence', 0)
            
            report_lines.append(f"[{i}] {timestamp} - {speaker}")
            report_lines.append(f"    Text: {text}")
            report_lines.append(f"    Sentiment: {sentiment.title()} (Score: {sentiment_score:.2f})")
            report_lines.append(f"    Tone: {tone.title()} (Confidence: {tone_confidence:.2f})")
            report_lines.append("")
        
        # Statistics Summary
        if mom.get('statistics'):
            stats_data = mom['statistics']
            report_lines.append("CONVERSATION STATISTICS")
            report_lines.append("-" * 23)
            report_lines.append(f"Total Utterances: {stats_data.get('total_utterances', 0)}")
            report_lines.append(f"Manager Utterances: {stats_data.get('manager_utterances', 0)}")
            report_lines.append(f"Employee Utterances: {stats_data.get('employee_utterances', 0)}")
            report_lines.append(f"Participation Ratio (Mgr:Emp): {stats_data.get('participation_ratio', 0):.2f}")
            report_lines.append(f"Average Sentiment: {stats_data.get('average_sentiment', 0):.3f}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 60)
        report_lines.append("End of Report")
        report_lines.append("Generated by Growth Talk Assistant (GTA)")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def generate_json_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate JSON format report"""
        json_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "Growth Talk Assistant (GTA)",
                "version": "1.0"
            },
            "analysis_results": analysis_results
        }
        
        return json.dumps(json_report, indent=2, ensure_ascii=False)
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate concise summary report"""
        mom = analysis_results.get('mom_summary', {})
        stats = analysis_results.get('sentiment_stats', {})
        
        summary_lines = []
        summary_lines.append("GROWTH TALK SUMMARY")
        summary_lines.append("=" * 20)
        summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        summary_lines.append("")
        
        summary_lines.append(f"Overall Sentiment: {mom.get('overall_sentiment', 'N/A').title()}")
        summary_lines.append(f"Meeting Outcome: {mom.get('meeting_outcome', 'N/A')}")
        summary_lines.append("")
        
        if mom.get('key_points'):
            summary_lines.append("Key Points:")
            for point in mom['key_points'][:3]:  # Top 3 only
                summary_lines.append(f"• {point}")
            summary_lines.append("")
        
        if mom.get('action_items'):
            summary_lines.append("Action Items:")
            for action in mom['action_items']:
                summary_lines.append(f"• {action}")
            summary_lines.append("")
        
        summary_lines.append(f"Sentiment Distribution: {stats.get('positive_ratio', 0):.0f}% Positive, "
                           f"{stats.get('neutral_ratio', 0):.0f}% Neutral, "
                           f"{stats.get('negative_ratio', 0):.0f}% Negative")
        
        return "\n".join(summary_lines)
