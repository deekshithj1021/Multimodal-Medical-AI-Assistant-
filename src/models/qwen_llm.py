"""
Language Model Module using Qwen (Large Language Model)
Handles medical reasoning and text generation
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, List
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenLLM:
    """
    Language model interface using Qwen for medical reasoning and response generation.
    Provides contextual understanding and medical knowledge synthesis.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen-1_8B-Chat",
        device: str = "cpu",
        trust_remote_code: bool = True
    ):
        """
        Initialize the Qwen Language Model.
        
        Args:
            model_name: Qwen model identifier from HuggingFace
            device: Device to run the model on (cpu, cuda)
            trust_remote_code: Whether to trust remote code (required for Qwen)
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading Qwen model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            logger.info("Qwen model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Qwen model: {str(e)}")
            logger.info("Falling back to GPT-2 for demonstration")
            # Fallback to a simpler model for demonstration
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """
        Generate text response using Qwen LLM.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate
        
        Returns:
            Dictionary containing generated response and metadata
        """
        try:
            logger.info("Generating response from Qwen LLM")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            logger.info("Response generated successfully")
            return {
                "responses": responses,
                "prompt": prompt,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "responses": [],
                "error": str(e),
                "success": False
            }
    
    def medical_reasoning(
        self,
        context: str,
        question: Optional[str] = None,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Perform medical reasoning based on provided context.
        
        Args:
            context: Medical context information
            question: Optional specific question to answer
            max_length: Maximum length of response
        
        Returns:
            Dictionary containing reasoning results
        """
        try:
            # Construct medical reasoning prompt
            if question:
                prompt = f"""Medical Context: {context}

Question: {question}

Based on the medical context provided, please provide a detailed analysis and answer:"""
            else:
                prompt = f"""Medical Context: {context}

Please analyze the medical information provided and give insights:"""
            
            logger.info("Performing medical reasoning")
            result = self.generate_response(prompt, max_length=max_length)
            
            return {
                "reasoning": result.get("responses", [""])[0] if result["success"] else "",
                "context": context,
                "question": question,
                "success": result["success"]
            }
        
        except Exception as e:
            logger.error(f"Error in medical reasoning: {str(e)}")
            return {
                "reasoning": "",
                "error": str(e),
                "success": False
            }
    
    def synthesize_multimodal_information(
        self,
        audio_text: str,
        visual_description: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize information from multiple modalities for medical reasoning.
        
        Args:
            audio_text: Text from audio transcription
            visual_description: Description from visual analysis
            additional_context: Additional context information
        
        Returns:
            Dictionary containing synthesized analysis
        """
        try:
            # Construct multimodal synthesis prompt
            prompt = f"""Multimodal Medical Data Analysis:

Audio Information (Patient/Doctor conversation):
{audio_text}

Visual Information (Medical images):
{visual_description}
"""
            if additional_context:
                prompt += f"""
Additional Context:
{additional_context}
"""
            
            prompt += """
Based on both the audio and visual information, please provide:
1. A comprehensive medical assessment
2. Key findings from the multimodal data
3. Potential diagnoses or concerns
4. Recommended next steps

Analysis:"""
            
            logger.info("Synthesizing multimodal information")
            result = self.generate_response(prompt, max_length=768)
            
            return {
                "synthesis": result.get("responses", [""])[0] if result["success"] else "",
                "audio_input": audio_text,
                "visual_input": visual_description,
                "success": result["success"]
            }
        
        except Exception as e:
            logger.error(f"Error synthesizing multimodal information: {str(e)}")
            return {
                "synthesis": "",
                "error": str(e),
                "success": False
            }
