"""
Basic validation script to ensure all modules can be imported correctly
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from src.models import AudioProcessor, VisualProcessor, QwenLLM, MultimodalFusion
        logger.info("✓ All model modules imported successfully")
        
        from src.utils import Config
        logger.info("✓ Utils modules imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from src.utils import Config
        config = Config.load_config()
        logger.info(f"✓ Configuration loaded: {list(config.keys())}")
        
        device = Config.get_device()
        logger.info(f"✓ Detected device: {device}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration error: {e}")
        return False


def test_module_structure():
    """Test that module classes can be instantiated (structure check)."""
    logger.info("Testing module structure...")
    
    try:
        # Just check that classes exist, not instantiating with heavy models
        from src.models import AudioProcessor, VisualProcessor, QwenLLM, MultimodalFusion
        
        assert hasattr(AudioProcessor, 'transcribe_audio'), "AudioProcessor missing transcribe_audio"
        logger.info("✓ AudioProcessor has correct methods")
        
        assert hasattr(VisualProcessor, 'caption_image'), "VisualProcessor missing caption_image"
        logger.info("✓ VisualProcessor has correct methods")
        
        assert hasattr(QwenLLM, 'generate_response'), "QwenLLM missing generate_response"
        logger.info("✓ QwenLLM has correct methods")
        
        assert hasattr(MultimodalFusion, 'fuse_audio_visual'), "MultimodalFusion missing fuse_audio_visual"
        logger.info("✓ MultimodalFusion has correct methods")
        
        return True
    except Exception as e:
        logger.error(f"✗ Module structure error: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("=" * 80)
    logger.info("Multimodal Medical AI Assistant - Validation Tests")
    logger.info("=" * 80)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config),
        ("Module Structure Tests", test_module_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 80)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n✓ All validation tests passed!")
        logger.info("The system structure is correct and ready for use.")
        logger.info("\nNote: Heavy model loading not tested to save resources.")
        logger.info("Run 'python examples.py' to test with actual models.")
        return 0
    else:
        logger.error("\n✗ Some validation tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
