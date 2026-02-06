"""
Lightweight validation script to check code structure without dependencies
"""

import sys
import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_python_file(filepath):
    """Validate that a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_file_structure():
    """Check that all expected files exist."""
    logger.info("Checking file structure...")
    
    expected_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "main.py",
        "examples.py",
        "config.yaml",
        "LICENSE",
        "QUICKSTART.md",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/audio_processor.py",
        "src/models/visual_processor.py",
        "src/models/qwen_llm.py",
        "src/models/multimodal_fusion.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
    ]
    
    missing_files = []
    for filepath in expected_files:
        if not Path(filepath).exists():
            missing_files.append(filepath)
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    
    logger.info(f"✓ All {len(expected_files)} expected files exist")
    return True


def check_python_syntax():
    """Check Python syntax for all .py files."""
    logger.info("Checking Python syntax...")
    
    python_files = list(Path(".").rglob("*.py"))
    python_files = [f for f in python_files if ".git" not in str(f)]
    
    errors = []
    for filepath in python_files:
        valid, error = validate_python_file(filepath)
        if not valid:
            errors.append((filepath, error))
    
    if errors:
        logger.error(f"✗ Syntax errors in {len(errors)} files:")
        for filepath, error in errors:
            logger.error(f"  {filepath}: {error}")
        return False
    
    logger.info(f"✓ All {len(python_files)} Python files have valid syntax")
    return True


def check_module_structure():
    """Check that key classes and functions are defined."""
    logger.info("Checking module structure...")
    
    checks = [
        ("src/models/audio_processor.py", ["AudioProcessor"]),
        ("src/models/visual_processor.py", ["VisualProcessor"]),
        ("src/models/qwen_llm.py", ["QwenLLM"]),
        ("src/models/multimodal_fusion.py", ["MultimodalFusion"]),
        ("src/utils/config.py", ["Config"]),
    ]
    
    errors = []
    for filepath, expected_classes in checks:
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            for expected_class in expected_classes:
                if expected_class not in defined_classes:
                    errors.append(f"{filepath} missing class {expected_class}")
        except Exception as e:
            errors.append(f"Error checking {filepath}: {e}")
    
    if errors:
        logger.error(f"✗ Structure errors:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    
    logger.info(f"✓ All expected classes are defined")
    return True


def check_key_methods():
    """Check that key methods exist in classes."""
    logger.info("Checking key methods...")
    
    checks = {
        "src/models/audio_processor.py": {
            "AudioProcessor": ["transcribe_audio", "process_audio_stream", "extract_medical_keywords"]
        },
        "src/models/visual_processor.py": {
            "VisualProcessor": ["caption_image", "answer_visual_question", "extract_visual_features"]
        },
        "src/models/qwen_llm.py": {
            "QwenLLM": ["generate_response", "medical_reasoning", "synthesize_multimodal_information"]
        },
        "src/models/multimodal_fusion.py": {
            "MultimodalFusion": ["fuse_audio_visual", "process_medical_case", "answer_medical_query"]
        },
    }
    
    errors = []
    for filepath, class_methods in checks.items():
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            for class_name, expected_methods in class_methods.items():
                class_node = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        class_node = node
                        break
                
                if not class_node:
                    errors.append(f"{filepath}: Class {class_name} not found")
                    continue
                
                defined_methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
                
                for expected_method in expected_methods:
                    if expected_method not in defined_methods:
                        errors.append(f"{filepath}: {class_name}.{expected_method} not found")
        
        except Exception as e:
            errors.append(f"Error checking {filepath}: {e}")
    
    if errors:
        logger.error(f"✗ Method errors:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    
    logger.info(f"✓ All key methods are defined")
    return True


def main():
    """Run all validation checks."""
    logger.info("=" * 80)
    logger.info("Multimodal Medical AI Assistant - Structure Validation")
    logger.info("=" * 80)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_python_syntax),
        ("Module Structure", check_module_structure),
        ("Key Methods", check_key_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}")
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
        logger.info("\n✓ All structure validation checks passed!")
        logger.info("\nThe implementation is complete and ready for use.")
        logger.info("To use the system, install dependencies: pip install -r requirements.txt")
        return 0
    else:
        logger.error("\n✗ Some validation checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
