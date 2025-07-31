import re
import io
from typing import List, Dict, Any
from datetime import timedelta

class VTTParser:
    """Parser for WebVTT transcript files"""
    
    def __init__(self):
        self.speaker_patterns = {
            'manager': ['manager', 'supervisor', 'lead', 'boss', 'director'],
            'employee': ['employee', 'team member', 'staff', 'worker', 'associate']
        }
    
    def parse_vtt(self, file_content) -> List[Dict[str, Any]]:
        """
        Parse VTT file and extract structured utterances
        
        Args:
            file_content: Uploaded file content
            
        Returns:
            List of dictionaries containing parsed utterances
        """
        try:
            # Read file content
            if hasattr(file_content, 'read'):
                content = file_content.read().decode('utf-8')
            else:
                content = str(file_content)
            
            utterances = []
            lines = content.strip().split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and headers
                if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                    i += 1
                    continue
                
                # Check if line contains timestamp
                if '-->' in line:
                    timestamp_line = line
                    i += 1
                    
                    # Collect text lines until next timestamp or end
                    text_lines = []
                    while i < len(lines) and '-->' not in lines[i] and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if text_lines:
                        # Parse timestamp
                        start_time, end_time = self._parse_timestamp(timestamp_line)
                        
                        # Combine text lines
                        full_text = ' '.join(text_lines)
                        
                        # Extract speaker and clean text
                        speaker, clean_text = self._extract_speaker(full_text)
                        
                        if clean_text:  # Only add if there's actual text content
                            utterance = {
                                'timestamp': start_time,
                                'end_time': end_time,
                                'duration': (end_time - start_time).total_seconds(),
                                'speaker': speaker,
                                'text': clean_text,
                                'raw_text': full_text
                            }
                            utterances.append(utterance)
                else:
                    i += 1
            
            return utterances
            
        except Exception as e:
            raise Exception(f"Error parsing VTT file: {str(e)}")
    
    def _parse_timestamp(self, timestamp_line: str) -> tuple:
        """Parse timestamp line to extract start and end times"""
        try:
            # Extract timestamps (format: HH:MM:SS.mmm --> HH:MM:SS.mmm)
            timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', timestamp_line)
            
            if timestamp_match:
                start_str, end_str = timestamp_match.groups()
                start_time = self._time_to_timedelta(start_str)
                end_time = self._time_to_timedelta(end_str)
                return start_time, end_time
            else:
                # Fallback for simpler timestamp formats
                parts = timestamp_line.split('-->')
                if len(parts) == 2:
                    start_time = self._time_to_timedelta(parts[0].strip())
                    end_time = self._time_to_timedelta(parts[1].strip())
                    return start_time, end_time
        except Exception:
            pass
        
        # Default fallback
        return timedelta(0), timedelta(0)
    
    def _time_to_timedelta(self, time_str: str) -> timedelta:
        """Convert time string to timedelta object"""
        try:
            # Handle format HH:MM:SS.mmm
            if '.' in time_str:
                time_part, ms_part = time_str.split('.')
                ms = int(ms_part)
            else:
                time_part = time_str
                ms = 0
            
            time_components = time_part.split(':')
            if len(time_components) == 3:
                hours, minutes, seconds = map(int, time_components)
                return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=ms)
            elif len(time_components) == 2:
                minutes, seconds = map(int, time_components)
                return timedelta(minutes=minutes, seconds=seconds, milliseconds=ms)
        except Exception:
            pass
        
        return timedelta(0)
    
    def _extract_speaker(self, text: str) -> tuple:
        """Extract speaker information and clean text"""
        # Common speaker patterns in transcripts
        speaker_patterns = [
            r'^([A-Za-z\s]+):\s*(.+)$',  # "Speaker Name: text"
            r'^<v\s+([^>]+)>(.+)$',      # "<v Speaker Name>text"
            r'^\[([^\]]+)\]\s*(.+)$',    # "[Speaker Name] text"
            r'^([A-Z][A-Za-z\s]+)\s*-\s*(.+)$'  # "Speaker Name - text"
        ]
        
        for pattern in speaker_patterns:
            match = re.match(pattern, text.strip())
            if match:
                speaker_name = match.group(1).strip()
                clean_text = match.group(2).strip()
                
                # Classify speaker as Manager or Employee
                speaker_type = self._classify_speaker(speaker_name)
                return speaker_type, clean_text
        
        # No speaker pattern found, try to infer from content or use default
        speaker_type = self._infer_speaker_from_content(text)
        return speaker_type, text.strip()
    
    def _classify_speaker(self, speaker_name: str) -> str:
        """Classify speaker as Manager or Employee based on name/title"""
        speaker_lower = speaker_name.lower()
        
        for manager_keyword in self.speaker_patterns['manager']:
            if manager_keyword in speaker_lower:
                return 'Manager'
        
        for employee_keyword in self.speaker_patterns['employee']:
            if employee_keyword in speaker_lower:
                return 'Employee'
        
        # If name contains typical manager indicators
        if any(word in speaker_lower for word in ['mgr', 'mgmt', 'head', 'chief', 'senior']):
            return 'Manager'
        
        # Default to Employee if uncertain
        return 'Employee'
    
    def _infer_speaker_from_content(self, text: str) -> str:
        """Infer speaker type from content when no explicit speaker is mentioned"""
        text_lower = text.lower()
        
        # Manager-like phrases
        manager_indicators = [
            'let me provide feedback', 'your performance', 'going forward',
            'expectations', 'goals for you', 'development plan',
            'team objectives', 'your role', 'areas for improvement'
        ]
        
        # Employee-like phrases  
        employee_indicators = [
            'i feel that', 'my concern is', 'i would like to',
            'can i get', 'i need support', 'my challenge',
            'i think', 'from my perspective'
        ]
        
        manager_score = sum(1 for indicator in manager_indicators if indicator in text_lower)
        employee_score = sum(1 for indicator in employee_indicators if indicator in text_lower)
        
        if manager_score > employee_score:
            return 'Manager'
        else:
            return 'Employee'
