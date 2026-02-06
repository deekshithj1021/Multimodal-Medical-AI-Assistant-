"""
Example Usage of the Multimodal Medical AI Assistant

This script demonstrates various use cases of the system.
"""

from src.models import MultimodalFusion, AudioProcessor, VisualProcessor, QwenLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_audio_processing():
    """Example: Process audio data with Whisper."""
    logger.info("\n=== Example 1: Audio Processing ===")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(model_size="base", device="cpu")
    
    # Example transcription (would need actual audio file)
    # result = audio_processor.transcribe_audio("patient_conversation.wav")
    # print(f"Transcription: {result['text']}")
    
    # Example with text
    sample_text = "Patient complains of chest pain and shortness of breath for the past 3 days"
    keywords = audio_processor.extract_medical_keywords(sample_text)
    logger.info(f"Medical keywords: {keywords}")


def example_visual_processing():
    """Example: Process medical images with BLIP."""
    logger.info("\n=== Example 2: Visual Processing ===")
    
    # Initialize visual processor
    visual_processor = VisualProcessor(device="cpu")
    
    # Example image captioning (would need actual image)
    # result = visual_processor.caption_image("xray_chest.jpg")
    # print(f"Image caption: {result['caption']}")
    
    logger.info("Visual processor initialized for medical image analysis")


def example_llm_reasoning():
    """Example: Medical reasoning with Qwen LLM."""
    logger.info("\n=== Example 3: LLM Medical Reasoning ===")
    
    # Initialize LLM
    llm = QwenLLM(device="cpu")
    
    # Example medical reasoning
    context = """
    Patient: 45-year-old male
    Symptoms: Chest pain, shortness of breath
    Duration: 3 days
    Medical history: Hypertension
    """
    
    result = llm.medical_reasoning(
        context=context,
        question="What could be the potential diagnosis?"
    )
    
    if result['success']:
        logger.info(f"Medical reasoning: {result['reasoning'][:200]}...")


def example_multimodal_fusion():
    """Example: Complete multimodal fusion."""
    logger.info("\n=== Example 4: Multimodal Data Fusion ===")
    
    # Initialize fusion system
    fusion = MultimodalFusion(
        audio_model_size="base",
        device="cpu"
    )
    
    # Example with text inputs (simulating processed data)
    audio_text = "Patient reports severe chest pain radiating to left arm, started 3 hours ago"
    visual_desc = "Chest X-ray shows enlarged cardiac silhouette with possible fluid accumulation"
    
    result = fusion.fuse_audio_visual(
        audio_text=audio_text,
        visual_description=visual_desc
    )
    
    if result['success']:
        logger.info("\nAudio Input: " + result['audio_input'])
        logger.info("\nVisual Input: " + result['visual_input'])
        logger.info("\nSynthesized Analysis: " + result['synthesized_analysis'][:300] + "...")


def example_medical_case():
    """Example: Process complete medical case."""
    logger.info("\n=== Example 5: Complete Medical Case Processing ===")
    
    # Initialize system
    fusion = MultimodalFusion(device="cpu")
    
    # Simulate a medical case (would use actual files)
    logger.info("Processing medical case with multimodal data...")
    logger.info("- Audio: Patient consultation recording")
    logger.info("- Images: X-ray, CT scan, lab results")
    logger.info("- Context: Patient history and vitals")
    
    # This would process actual files:
    # result = fusion.process_medical_case(
    #     audio_path="consultation.wav",
    #     image_paths=["xray.jpg", "ct_scan.jpg"],
    #     patient_context="Patient history: Diabetes, Hypertension"
    # )


def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("Multimodal Medical AI Assistant - Usage Examples")
    logger.info("=" * 80)
    
    try:
        example_audio_processing()
        example_visual_processing()
        example_llm_reasoning()
        example_multimodal_fusion()
        example_medical_case()
        
        logger.info("\n" + "=" * 80)
        logger.info("Examples completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
