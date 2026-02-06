# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Examples

```bash
python examples.py
```

This will demonstrate the capabilities without requiring actual audio/image files.

### Step 3: Process Your Own Data

#### Example 1: Transcribe Medical Audio

```bash
python main.py --audio your_audio.wav --mode fusion
```

#### Example 2: Analyze Medical Image

```bash
python main.py --image xray.jpg --mode fusion
```

#### Example 3: Combined Audio + Visual Analysis

```bash
python main.py --audio consultation.wav --image scan.jpg --mode fusion
```

## 🎯 Common Use Cases

### Use Case 1: Patient Consultation Analysis

```python
from src.models import MultimodalFusion

fusion = MultimodalFusion()
result = fusion.process_medical_case(
    audio_path="patient_consultation.wav",
    image_paths=["xray.jpg", "blood_test.jpg"],
    patient_context="45-year-old male with chest pain"
)

print(result['comprehensive_analysis'])
```

### Use Case 2: Quick Medical Query

```python
from src.models import MultimodalFusion

fusion = MultimodalFusion()
result = fusion.answer_medical_query(
    query="What does this X-ray show?",
    image_path="chest_xray.jpg"
)

print(result['answer'])
```

### Use Case 3: Audio-Only Transcription

```python
from src.models import AudioProcessor

audio = AudioProcessor(model_size="base")
result = audio.transcribe_audio("doctor_notes.wav")

print(result['text'])
keywords = audio.extract_medical_keywords(result['text'])
print(f"Medical keywords: {keywords}")
```

## 📊 Model Selection

### Audio Models (Whisper)
- `tiny`: Fastest, least accurate (~1GB)
- `base`: Good balance (default, ~1GB)  ✅
- `small`: Better accuracy (~2GB)
- `medium`: High accuracy (~5GB)
- `large`: Best accuracy (~10GB)

### Visual Model (BLIP)
- Default: `Salesforce/blip-image-captioning-base` ✅
- Alternative: `Salesforce/blip-image-captioning-large`

### LLM (Qwen)
- Default: `Qwen/Qwen-1_8B-Chat` ✅
- Alternative: `Qwen/Qwen-7B-Chat` (requires more memory)

## 🔧 Configuration

Edit `config.yaml` to change model settings:

```yaml
audio:
  model_size: base
  device: cpu  # Change to 'cuda' for GPU

visual:
  device: cpu  # Change to 'cuda' for GPU

llm:
  device: cpu  # Change to 'cuda' for GPU
  temperature: 0.7  # Lower = more focused, Higher = more creative
```

## 💡 Tips

1. **GPU Acceleration**: Set `device: cuda` in config for faster processing
2. **Memory Management**: Use smaller models (`tiny`, `base`) on limited hardware
3. **Batch Processing**: Process multiple cases programmatically
4. **Custom Prompts**: Modify LLM prompts in `qwen_llm.py` for specific use cases

## ⚠️ Troubleshooting

### Out of Memory
- Use smaller models: `--audio-model tiny`
- Process on CPU: `--device cpu`

### Model Download Issues
- Models are downloaded automatically on first use
- Requires internet connection
- Large models may take time to download

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: Python 3.8+ required

## 📚 Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore [examples.py](examples.py) for more code samples
3. Customize the system for your specific needs
4. Integrate with your existing medical workflow

## 🆘 Support

For issues or questions:
1. Check the [README.md](README.md)
2. Open an issue on GitHub
3. Review the code documentation

---

**Remember**: This is a research tool. Always verify results with medical professionals for clinical use.
