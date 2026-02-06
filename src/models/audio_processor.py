"""
Audio Processing Module using OpenAI Whisper
Handles speech-to-text transcription for medical audio data
"""

import whisper
import numpy as np
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing module using OpenAI Whisper for speech transcription.
    Optimized for medical audio streams and conversations.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize the Audio Processor with Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run the model on (cpu, cuda)
        """
        self.model_size = model_size
        self.device = device
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")
    
    def transcribe_audio(
        self, 
        audio_path: str, 
        language: Optional[str] = "en",
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: English)
            task: Task type - 'transcribe' or 'translate'
        
        Returns:
            Dictionary containing transcription results with text and metadata
        """
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=False
            )
            
            logger.info("Audio transcription completed")
            return {
                "text": result["text"],
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def process_audio_stream(
        self, 
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Process audio stream from numpy array.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Whisper expects 16kHz audio
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate is {sample_rate}, Whisper expects 16000")
            
            logger.info("Processing audio stream")
            result = self.model.transcribe(audio_data, verbose=False)
            
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error processing audio stream: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def extract_medical_keywords(self, transcription: str) -> list:
        """
        Extract potential medical keywords from transcription.
        
        Args:
            transcription: Transcribed text
        
        Returns:
            List of medical-related keywords
        """
        # Simple keyword extraction - can be enhanced with NER
        medical_terms = [
            "symptom", "diagnosis", "treatment", "medication", "patient",
            "pain", "fever", "cough", "allergy", "prescription", "doctor",
            "hospital", "surgery", "condition", "disease", "infection"
        ]
        
        found_keywords = []
        transcription_lower = transcription.lower()
        
        for term in medical_terms:
            if term in transcription_lower:
                found_keywords.append(term)
        
        return found_keywords
