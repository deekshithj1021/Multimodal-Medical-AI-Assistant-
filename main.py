#!/usr/bin/env python3
"""
Multimodal Medical AI Assistant - Main Application
Synthesizes heterogeneous visual and audio streams for automated medical reasoning

This application integrates:
- Qwen (LLM) for medical reasoning
- OpenAI Whisper for audio transcription
- BLIP for visual understanding
- Multimodal Data Fusion for comprehensive analysis
"""

import argparse
import sys
from pathlib import Path
from src.models import MultimodalFusion
from src.utils import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Multimodal Medical AI Assistant - Synthesize visual and audio streams for medical reasoning'
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file (patient conversation, notes, etc.)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to medical image file'
    )
    
    parser.add_argument(
        '--images',
        nargs='+',
        help='Paths to multiple medical image files'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Medical query to answer'
    )
    
    parser.add_argument(
        '--context',
        type=str,
        help='Additional patient context information'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fusion', 'case', 'query'],
        default='fusion',
        help='Operation mode: fusion (basic), case (comprehensive), or query (Q&A)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run models on'
    )
    
    parser.add_argument(
        '--audio-model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.audio and not args.image and not args.images:
        logger.error("At least one of --audio, --image, or --images must be provided")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize the multimodal fusion system
        logger.info("=" * 80)
        logger.info("Multimodal Medical AI Assistant")
        logger.info("=" * 80)
        
        logger.info(f"Initializing system on device: {args.device}")
        fusion_system = MultimodalFusion(
            audio_model_size=args.audio_model,
            device=args.device
        )
        
        # Execute based on mode
        if args.mode == 'fusion':
            logger.info("\nMode: Basic Multimodal Fusion")
            result = fusion_system.fuse_audio_visual(
                audio_path=args.audio,
                image_path=args.image
            )
            
            if result['success']:
                logger.info("\n" + "=" * 80)
                logger.info("FUSION RESULTS")
                logger.info("=" * 80)
                
                if result.get('audio_input'):
                    logger.info("\n[AUDIO INPUT]")
                    logger.info(result['audio_input'])
                
                if result.get('visual_input'):
                    logger.info("\n[VISUAL INPUT]")
                    logger.info(result['visual_input'])
                
                if result.get('medical_keywords'):
                    logger.info("\n[MEDICAL KEYWORDS]")
                    logger.info(", ".join(result['medical_keywords']))
                
                logger.info("\n[SYNTHESIZED ANALYSIS]")
                logger.info(result['synthesized_analysis'])
            else:
                logger.error(f"Fusion failed: {result.get('error', 'Unknown error')}")
        
        elif args.mode == 'case':
            logger.info("\nMode: Comprehensive Medical Case Analysis")
            image_paths = args.images if args.images else ([args.image] if args.image else None)
            
            result = fusion_system.process_medical_case(
                audio_path=args.audio,
                image_paths=image_paths,
                patient_context=args.context
            )
            
            if result['success']:
                logger.info("\n" + "=" * 80)
                logger.info("MEDICAL CASE ANALYSIS")
                logger.info("=" * 80)
                
                if result.get('audio_transcription'):
                    logger.info("\n[AUDIO TRANSCRIPTION]")
                    logger.info(result['audio_transcription'])
                
                if result.get('visual_analyses'):
                    logger.info("\n[VISUAL ANALYSES]")
                    for i, analysis in enumerate(result['visual_analyses'], 1):
                        logger.info(f"\nImage {i}: {analysis['path']}")
                        logger.info(f"Caption: {analysis['caption']}")
                
                if result.get('patient_context'):
                    logger.info("\n[PATIENT CONTEXT]")
                    logger.info(result['patient_context'])
                
                logger.info("\n[COMPREHENSIVE ANALYSIS]")
                logger.info(result['comprehensive_analysis'])
            else:
                logger.error(f"Case analysis failed: {result.get('error', 'Unknown error')}")
        
        elif args.mode == 'query':
            if not args.query:
                logger.error("--query must be provided in query mode")
                sys.exit(1)
            
            logger.info(f"\nMode: Medical Query - {args.query}")
            result = fusion_system.answer_medical_query(
                query=args.query,
                audio_path=args.audio,
                image_path=args.image
            )
            
            if result['success']:
                logger.info("\n" + "=" * 80)
                logger.info("QUERY RESPONSE")
                logger.info("=" * 80)
                logger.info(f"\nQuery: {result['query']}")
                logger.info(f"\nContext:\n{result['context']}")
                logger.info(f"\nAnswer:\n{result['answer']}")
            else:
                logger.error(f"Query failed: {result.get('error', 'Unknown error')}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis Complete")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
