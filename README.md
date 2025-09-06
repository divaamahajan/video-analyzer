# 🎬 Video Analysis System

A comprehensive AI-powered video analysis system that evaluates communication effectiveness through visual, audio, and content analysis.

## 🌟 Features

### 👁️ Visual Language Analysis (Mute & Watch)
- **Body Posture & Positioning**: Posture scores, slouching detection, consistency metrics
- **Facial Expressions & Eye Contact**: Smile intensity, eye contact tracking, engagement levels
- **Gestures & Hand Movements**: Gesture frequency, consistency, and patterns
- **Overall Engagement**: Energy scores, expressiveness levels, high engagement frame detection

### 🧍 Body Language & Pose Analysis
- **Basic Posture**: Fidgeting, slouching, leaning, head tilt, forward head posture
- **Head Movement**: Nods, shakes, forward/back movements
- **Arm & Hand Positions**: Crossed arms, hands on hips, face touching, gesturing
- **Energy & Movement**: Low/medium/high energy classification, movement magnitude

### 🎭 Facial Expression & Emotion Analysis
- **Emotion Distribution**: Happy, neutral, sad, angry, surprise, fear detection
- **Emotion Mismatch Analysis**: Compares detected vs intended emotions
- **Analysis Quality**: Confidence scores, valid detection rates

### 🎵 Audio Voice, Tone & Clarity Analysis (Listen)
- **Transcription**: Automatic speech-to-text conversion
- **Pace & Rhythm**: Words per minute, pause analysis, filler word detection
- **Pitch & Tone**: Pitch range, variance, intonation patterns, emotional indicators
- **Volume & Clarity**: Volume levels, clipping detection, dynamic range, clarity scores
- **Pronunciation & Articulation**: Articulation scores, mispronunciation detection
- **Emotion & Sentiment**: Audio emotional intensity, text sentiment analysis

### 📝 Content & Linguistic Analysis
- **Structure & Organization**: Sentence length, complexity, idea flow, transitions
- **Clarity & Conciseness**: Redundancy detection, wordiness, ambiguity analysis
- **Fillers & Disfluencies**: Filler word counting, hesitation phrase detection
- **Vocabulary & Word Choice**: Lexical diversity, technical vs simple words, power words
- **Readability & Engagement**: Flesch-Kincaid scores, rhetorical devices, emphasis
- **Topic & Semantic Flow**: Main ideas, supporting evidence, coherence analysis
- **Sentiment & Confidence**: Confidence markers, tentative language detection

### 🤖 AI-Powered Analysis & Recommendations
- **Comprehensive Diagnosis**: Identifies key issues across all analysis areas
- **Priority Focus Areas**: Ranks improvements by impact and urgency
- **Actionable Prescriptions**: Converts metrics into specific practice drills
- **Final Insights Report**: Generates `Final_insights_and_recommendations_YYYYMMDD_HHMMSS.md`

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Video-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

5. **Run the analysis**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
Video-analysis/
├── main.py                          # Main entry point
├── requirements.txt                  # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── input_video.mp4                  # Input video file
├── input_audio.wav                  # Extracted audio file
├── emotion_analysis.json            # Emotion analysis results
├── reports/                         # Generated analysis reports
│   ├── comprehensive_analysis_report_*.md
│   ├── Final_insights_and_recommendations_*.md
│   ├── pose_analysis_report.md
│   └── emotion_analysis_report.md
├── src/                            # Source code
│   ├── analysis/                   # Analysis modules
│   │   ├── visual/                # Visual analysis
│   │   │   ├── simple_visual_analyzer.py
│   │   │   ├── pose_analyzer.py
│   │   │   └── video_processor.py
│   │   ├── audio/                 # Audio analysis
│   │   │   ├── audio_processor.py
│   │   │   ├── pace_analyzer.py
│   │   │   ├── pronunciation_analyzer.py
│   │   │   └── sentiment_analyzer.py
│   │   ├── text/                  # Text analysis
│   │   │   └── content_analyzer.py
│   │   └── report_analyzer.py      # AI report analysis
│   ├── config/                    # Configuration
│   │   └── settings.py
│   └── utils/                     # Utilities
│       ├── ai_agent.py
│       └── report_generator.py
├── legacy/                        # Legacy analysis modules
│   ├── emotion_detection.py
│   ├── pose_detection.py
│   └── report_generator.py
└── tests/                        # Test files
    └── test_*.py
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Settings
Modify `src/config/settings.py` to adjust:
- Model URLs and parameters
- Analysis thresholds
- File paths and intervals

## 📊 Output Files

### Comprehensive Analysis Report
- **File**: `reports/comprehensive_analysis_report_YYYYMMDD_HHMMSS.md`
- **Content**: Complete analysis results from all modules
- **Sections**: Visual, pose, emotion, audio, and content analysis

### Final Insights Report
- **File**: `reports/Final_insights_and_recommendations_YYYYMMDD_HHMMSS.md`
- **Content**: AI-powered diagnosis, prioritization, and actionable recommendations
- **Format**: Structured insights with specific practice drills

### Individual Analysis Reports
- **Pose Analysis**: `reports/pose_analysis_report.md`
- **Emotion Analysis**: `reports/emotion_analysis_report.md`
- **Emotion Data**: `emotion_analysis.json`

## 🔧 Usage Examples

### Basic Analysis
```python
from src.analysis.visual import SimpleVisualAnalyzer
from src.analysis.audio import AudioAnalyzer
from src.analysis.text import ContentAnalyzer

# Visual analysis
visual_analyzer = SimpleVisualAnalyzer()
visual_results = visual_analyzer.analyze_video("input_video.mp4")

# Audio analysis
audio_analyzer = AudioAnalyzer()
audio_results = audio_analyzer.analyze_audio("input_video.mp4")

# Content analysis
content_analyzer = ContentAnalyzer("your_openai_api_key")
content_results = content_analyzer.analyze_content(audio_results["transcription"])
```

### Custom Analysis
```python
from src.utils.report_generator import ComprehensiveReportGenerator

# Generate comprehensive report
report_generator = ComprehensiveReportGenerator()
report_path = report_generator.generate_comprehensive_report(
    visual_results=visual_results,
    pose_results=pose_results,
    emotion_results=emotion_results,
    audio_results=audio_results,
    content_results=content_results
)
```

## 🧪 Testing

Run individual tests:
```bash
python tests/test_content_analyzer_standalone.py
```

## 📈 Analysis Metrics

### Visual Metrics
- **Engagement Score**: 0.0 - 1.0 (higher = more engaging)
- **Posture Score**: 0.0 - 1.0 (higher = better posture)
- **Eye Contact**: 0.0 - 1.0 (higher = more eye contact)
- **Gesture Frequency**: Gestures per second

### Audio Metrics
- **Words Per Minute**: Speech pace
- **Pitch Variance**: Vocal variety
- **Clarity Score**: 0.0 - 1.0 (higher = clearer)
- **Articulation Score**: 0.0 - 1.0 (higher = better articulation)

### Content Metrics
- **Lexical Diversity**: 0.0 - 1.0 (higher = more diverse vocabulary)
- **Readability Score**: Flesch-Kincaid grade level
- **Clarity Score**: 0.0 - 1.0 (higher = clearer content)
- **Engagement Score**: 0.0 - 1.0 (higher = more engaging)

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated
2. **OpenAI API Error**: Check API key in `.env` file
3. **CUDA/GPU Issues**: Install CPU-only TensorFlow if needed
4. **Memory Issues**: Reduce frame sampling interval in settings

### Dependencies
- **TensorFlow**: For pose detection
- **OpenCV**: For video processing
- **Librosa**: For audio analysis
- **DeepFace**: For emotion detection
- **OpenAI**: For AI-powered analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow Hub**: For MoveNet pose detection model
- **DeepFace**: For facial expression analysis
- **OpenAI**: For AI-powered content analysis
- **Librosa**: For audio processing and analysis

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed description

---

**Happy Analyzing! 🎬✨**