"""
Visual Processing Module using BLIP (Bootstrapping Language-Image Pre-training)
Handles image understanding and captioning for medical images
"""

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional, Dict, Any, List
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualProcessor:
    """
    Visual processing module using BLIP for image understanding and captioning.
    Optimized for medical images and visual data analysis.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        """
        Initialize the Visual Processor with BLIP model.
        
        Args:
            model_name: BLIP model identifier from HuggingFace
            device: Device to run the model on (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        logger.info("BLIP model loaded successfully")
    
    def caption_image(
        self, 
        image_path: str, 
        max_length: int = 50,
        num_beams: int = 4
    ) -> Dict[str, Any]:
        """
        Generate caption for an image using BLIP.
        
        Args:
            image_path: Path to image file
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
        
        Returns:
            Dictionary containing caption and metadata
        """
        try:
            logger.info(f"Generating caption for image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Caption generated: {caption}")
            return {
                "caption": caption,
                "image_path": image_path,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {
                "caption": "",
                "error": str(e),
                "success": False
            }
    
    def answer_visual_question(
        self,
        image_path: str,
        question: str,
        max_length: int = 50
    ) -> Dict[str, Any]:
        """
        Answer questions about an image using BLIP's VQA capabilities.
        
        Args:
            image_path: Path to image file
            question: Question about the image
            max_length: Maximum length of answer
        
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            logger.info(f"Answering visual question for image: {image_path}")
            logger.info(f"Question: {question}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length
                )
            
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Answer: {answer}")
            return {
                "question": question,
                "answer": answer,
                "image_path": image_path,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error answering visual question: {str(e)}")
            return {
                "question": question,
                "answer": "",
                "error": str(e),
                "success": False
            }
    
    def process_image_from_pil(
        self,
        image: Image.Image,
        task: str = "caption",
        question: Optional[str] = None,
        max_length: int = 50
    ) -> Dict[str, Any]:
        """
        Process PIL Image directly without file path.
        
        Args:
            image: PIL Image object
            task: Task type - 'caption' or 'vqa' (visual question answering)
            question: Question for VQA task
            max_length: Maximum length of generated text
        
        Returns:
            Dictionary containing results
        """
        try:
            if task == "caption":
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            elif task == "vqa" and question:
                inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            else:
                raise ValueError("Invalid task or missing question for VQA")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "text": text,
                "task": task,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def extract_visual_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract visual features from image for fusion.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary containing visual features
        """
        try:
            logger.info(f"Extracting visual features from: {image_path}")
            
            # Generate caption as visual feature
            caption_result = self.caption_image(image_path)
            
            # Additional medical-specific visual questions
            medical_questions = [
                "What type of medical image is this?",
                "Are there any abnormalities visible?"
            ]
            
            vqa_results = []
            for question in medical_questions:
                answer_result = self.answer_visual_question(image_path, question)
                if answer_result["success"]:
                    vqa_results.append({
                        "question": question,
                        "answer": answer_result["answer"]
                    })
            
            return {
                "caption": caption_result.get("caption", ""),
                "vqa_results": vqa_results,
                "image_path": image_path,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error extracting visual features: {str(e)}")
            return {
                "caption": "",
                "vqa_results": [],
                "error": str(e),
                "success": False
            }
