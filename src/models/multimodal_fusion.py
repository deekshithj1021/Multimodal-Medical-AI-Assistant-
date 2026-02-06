"""
Multimodal Data Fusion Module
Synthesizes heterogeneous visual and audio streams for automated medical reasoning
"""

from typing import Dict, Any, Optional, List
import logging
from .audio_processor import AudioProcessor
from .visual_processor import VisualProcessor
from .qwen_llm import QwenLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    Core data fusion module that synthesizes visual and audio streams
    for comprehensive medical analysis and reasoning.
    
    This module integrates:
    - Audio processing (OpenAI Whisper)
    - Visual processing (BLIP)
    - Language reasoning (Qwen LLM)
    """
    
    def __init__(
        self,
        audio_model_size: str = "base",
        visual_model: str = "Salesforce/blip-image-captioning-base",
        llm_model: str = "Qwen/Qwen-1_8B-Chat",
        device: str = "cpu"
    ):
        """
        Initialize the Multimodal Fusion system.
        
        Args:
            audio_model_size: Whisper model size
            visual_model: BLIP model identifier
            llm_model: Qwen LLM model identifier
            device: Device to run models on (cpu, cuda)
        """
        self.device = device
        logger.info("Initializing Multimodal Fusion System")
        
        # Initialize individual processors
        self.audio_processor = AudioProcessor(
            model_size=audio_model_size,
            device=device
        )
        self.visual_processor = VisualProcessor(
            model_name=visual_model,
            device=device
        )
        self.llm = QwenLLM(
            model_name=llm_model,
            device=device
        )
        
        logger.info("Multimodal Fusion System initialized successfully")
    
    def fuse_audio_visual(
        self,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_text: Optional[str] = None,
        visual_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fuse audio and visual data for medical reasoning.
        
        Args:
            audio_path: Path to audio file (if not pre-processed)
            image_path: Path to image file (if not pre-processed)
            audio_text: Pre-processed audio transcription
            visual_description: Pre-processed visual description
        
        Returns:
            Dictionary containing fused analysis
        """
        try:
            logger.info("Starting multimodal data fusion")
            
            # Process audio if path provided
            if audio_path and not audio_text:
                logger.info("Processing audio stream")
                audio_result = self.audio_processor.transcribe_audio(audio_path)
                audio_text = audio_result.get("text", "")
                audio_keywords = self.audio_processor.extract_medical_keywords(audio_text)
            else:
                audio_keywords = []
            
            # Process visual if path provided
            if image_path and not visual_description:
                logger.info("Processing visual stream")
                visual_result = self.visual_processor.extract_visual_features(image_path)
                visual_description = f"Image Caption: {visual_result.get('caption', '')}"
                
                # Add VQA results
                for vqa in visual_result.get("vqa_results", []):
                    visual_description += f"\n{vqa['question']}: {vqa['answer']}"
            
            # Ensure we have data to process
            if not audio_text and not visual_description:
                return {
                    "success": False,
                    "error": "No audio or visual data provided"
                }
            
            # Prepare fusion data
            fusion_data = {
                "audio_text": audio_text or "No audio data",
                "visual_description": visual_description or "No visual data",
                "audio_keywords": audio_keywords
            }
            
            # Synthesize using LLM
            logger.info("Synthesizing multimodal information with LLM")
            synthesis_result = self.llm.synthesize_multimodal_information(
                audio_text=fusion_data["audio_text"],
                visual_description=fusion_data["visual_description"]
            )
            
            return {
                "audio_input": fusion_data["audio_text"],
                "visual_input": fusion_data["visual_description"],
                "medical_keywords": audio_keywords,
                "synthesized_analysis": synthesis_result.get("synthesis", ""),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error in multimodal fusion: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_medical_case(
        self,
        audio_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        patient_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a complete medical case with multiple modalities.
        
        Args:
            audio_path: Path to audio consultation/notes
            image_paths: List of paths to medical images
            patient_context: Additional patient context information
        
        Returns:
            Comprehensive medical case analysis
        """
        try:
            logger.info("Processing complete medical case")
            
            # Process audio
            audio_text = ""
            if audio_path:
                audio_result = self.audio_processor.transcribe_audio(audio_path)
                audio_text = audio_result.get("text", "")
            
            # Process all images
            visual_descriptions = []
            if image_paths:
                for img_path in image_paths:
                    visual_result = self.visual_processor.extract_visual_features(img_path)
                    if visual_result["success"]:
                        visual_descriptions.append({
                            "path": img_path,
                            "caption": visual_result.get("caption", ""),
                            "analysis": visual_result.get("vqa_results", [])
                        })
            
            # Combine visual descriptions
            combined_visual = ""
            for i, desc in enumerate(visual_descriptions, 1):
                combined_visual += f"\n\nImage {i} ({desc['path']}):\n"
                combined_visual += f"Description: {desc['caption']}\n"
                for vqa in desc.get("analysis", []):
                    combined_visual += f"- {vqa['question']}: {vqa['answer']}\n"
            
            # Synthesize with LLM
            synthesis_result = self.llm.synthesize_multimodal_information(
                audio_text=audio_text,
                visual_description=combined_visual,
                additional_context=patient_context
            )
            
            return {
                "audio_transcription": audio_text,
                "visual_analyses": visual_descriptions,
                "patient_context": patient_context,
                "comprehensive_analysis": synthesis_result.get("synthesis", ""),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error processing medical case: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def answer_medical_query(
        self,
        query: str,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a specific medical query using multimodal data.
        
        Args:
            query: Medical question to answer
            audio_path: Optional audio context
            image_path: Optional visual context
        
        Returns:
            Answer based on multimodal analysis
        """
        try:
            logger.info(f"Answering medical query: {query}")
            
            # Gather multimodal context
            context = []
            
            if audio_path:
                audio_result = self.audio_processor.transcribe_audio(audio_path)
                context.append(f"Audio Context: {audio_result.get('text', '')}")
            
            if image_path:
                visual_result = self.visual_processor.caption_image(image_path)
                context.append(f"Visual Context: {visual_result.get('caption', '')}")
            
            # Combine context
            combined_context = "\n".join(context) if context else "No additional context"
            
            # Use LLM for reasoning
            reasoning_result = self.llm.medical_reasoning(
                context=combined_context,
                question=query
            )
            
            return {
                "query": query,
                "context": combined_context,
                "answer": reasoning_result.get("reasoning", ""),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error answering medical query: {str(e)}")
            return {
                "query": query,
                "answer": "",
                "error": str(e),
                "success": False
            }
