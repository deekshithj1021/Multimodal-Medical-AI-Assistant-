# Multimodal Medical AI Assistant 🏥🤖

**Data Fusion: Synthesized heterogeneous visual and audio streams for automated medical reasoning**

A comprehensive AI-powered medical assistant that integrates multiple modalities (audio, visual, and text) to provide intelligent medical analysis and reasoning.

## 🌟 Features

This system integrates cutting-edge AI models for multimodal medical analysis:

- **🎙️ Audio Processing (OpenAI Whisper)**: Transcribe patient conversations, doctor's notes, and medical consultations
- **👁️ Visual Analysis (BLIP)**: Analyze medical images, X-rays, CT scans, and generate descriptive captions
- **🧠 Medical Reasoning (Qwen LLM)**: Advanced language model for medical knowledge synthesis and reasoning
- **🔄 Multimodal Data Fusion**: Intelligently combine audio and visual data streams for comprehensive medical analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Multimodal Medical AI Assistant            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Audio Stream  │  │ Visual Stream│  │  Text Context   │ │
│  │   (Whisper)   │  │    (BLIP)    │  │                 │ │
│  └───────┬───────┘  └──────┬───────┘  └────────┬────────┘ │
│          │                  │                    │          │
│          └──────────────────┼────────────────────┘          │
│                             ▼                               │
│                  ┌──────────────────────┐                   │
│                  │  Data Fusion Module  │                   │
│                  └──────────┬───────────┘                   │
│                             ▼                               │
│                  ┌──────────────────────┐                   │
│                  │   Qwen LLM Reasoning │                   │
│                  └──────────┬───────────┘                   │
│                             ▼                               │
│                  ┌──────────────────────┐                   │
│                  │  Medical Analysis    │                   │
│                  └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/deekshithj1021/Multimodal-Medical-AI-Assistant-.git
cd Multimodal-Medical-AI-Assistant-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Multimodal Fusion

Combine audio and visual data for medical analysis:

```bash
python main.py --audio patient_conversation.wav --image xray_chest.jpg --mode fusion
```

### Comprehensive Medical Case Analysis

Process multiple images with audio and context:

```bash
python main.py \
  --audio consultation.wav \
  --images xray.jpg ct_scan.jpg lab_results.jpg \
  --context "Patient history: Diabetes, Hypertension" \
  --mode case
```

### Medical Query Answering

Answer specific medical questions with multimodal context:

```bash
python main.py \
  --audio symptoms.wav \
  --image medical_scan.jpg \
  --query "What could be the potential diagnosis?" \
  --mode query
```

### Python API Usage

```python
from src.models import MultimodalFusion

# Initialize the system
fusion = MultimodalFusion(
    audio_model_size="base",
    device="cpu"  # or "cuda" for GPU
)

# Process multimodal data
result = fusion.fuse_audio_visual(
    audio_path="patient_audio.wav",
    image_path="medical_image.jpg"
)

if result['success']:
    print("Synthesized Analysis:", result['synthesized_analysis'])
```

## 🔧 Components

### 1. Audio Processor (Whisper)
- Transcribes medical audio recordings
- Extracts medical keywords
- Supports multiple languages
- Models: tiny, base, small, medium, large

### 2. Visual Processor (BLIP)
- Generates captions for medical images
- Visual question answering
- Extracts visual features for fusion
- Optimized for medical imaging

### 3. Qwen LLM
- Medical reasoning and analysis
- Multimodal information synthesis
- Context-aware responses
- Medical knowledge integration

### 4. Multimodal Fusion
- Combines audio and visual streams
- Synthesizes heterogeneous data
- Comprehensive medical case analysis
- Query-based reasoning

## 📚 Examples

Run the examples script to see demonstrations:

```bash
python examples.py
```

This includes:
- Audio transcription with medical keyword extraction
- Medical image captioning and analysis
- LLM-based medical reasoning
- Complete multimodal data fusion
- Medical case processing

## 🎯 Use Cases

- **Patient Consultation Analysis**: Transcribe and analyze doctor-patient conversations with relevant medical images
- **Diagnostic Support**: Combine verbal symptoms with visual medical scans for comprehensive diagnosis
- **Medical Documentation**: Automatically generate medical reports from audio notes and images
- **Research**: Analyze multimodal medical datasets for research purposes
- **Telemedicine**: Remote patient assessment using audio and visual data

## ⚙️ Configuration

The system can be configured through command-line arguments or programmatically:

```python
from src.utils import Config

# Load custom configuration
config = Config.load_config("config.yaml")

# Initialize with custom config
fusion = MultimodalFusion(
    audio_model_size=config['audio']['model_size'],
    visual_model=config['visual']['model_name'],
    llm_model=config['llm']['model_name'],
    device=config['fusion']['device']
)
```

## 🔒 Privacy & Security

- All processing can be done locally (no data sent to external servers)
- Patient data privacy compliant
- Secure handling of medical information
- HIPAA considerations supported

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Audio transcription
- **Salesforce BLIP**: Visual understanding
- **Qwen**: Language model reasoning
- **HuggingFace**: Model hosting and transformers library

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This system is designed for research and educational purposes. For clinical use, please ensure compliance with relevant medical regulations and standards.
