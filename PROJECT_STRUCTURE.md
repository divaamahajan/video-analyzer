# Video Analysis System - Project Structure

## 📁 Directory Organization

```
Video-analysis/
├── main.py                          # Main entry point
├── requirements.txt                 # Project dependencies
├── README.md                       # Project overview
├── input_video.mp4                 # Input video file
├── input_audio.wav                 # Generated audio file
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── analysis/                   # Analysis modules (organized by type)
│   │   ├── __init__.py
│   │   ├── visual/                 # Visual analysis
│   │   │   ├── __init__.py
│   │   │   ├── video_processor.py      # Video frame extraction
│   │   │   ├── simple_visual_analyzer.py # Visual analysis
│   │   │   └── pose_analyzer.py        # Pose detection
│   │   ├── audio/                  # Audio analysis
│   │   │   ├── __init__.py
│   │   │   ├── audio_processor.py      # Direct audio analysis
│   │   │   └── audio_analyzer.py       # Audio analysis coordinator
│   │   └── text/                   # Text-based analysis
│   │       ├── __init__.py
│   │       ├── transcription_service.py # Basic transcription
│   │       ├── transcription_analyzer.py # Transcription coordinator
│   │       ├── pace_analyzer.py        # Pace & rhythm analysis
│   │       ├── pronunciation_analyzer.py # Pronunciation analysis
│   │       └── sentiment_analyzer.py   # Sentiment analysis
│   ├── config/                     # Configuration
│   │   ├── __init__.py
│   │   └── settings.py             # Project settings
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── summary_generator.py    # Summary generation
│       └── ai_agent.py             # AI agent utilities
│
├── legacy/                         # Legacy code (not used in main.py)
│   ├── emotion_detection.py
│   ├── emotion_report_generator.py
│   ├── pose_detection.py
│   ├── pose_report_generator.py
│   └── report_generator.py
│
├── reports/                        # Generated reports
│   ├── analysis_report.md
│   ├── emotion_analysis_report.md
│   ├── emotion_analysis.json
│   └── pose_analysis_report.md
│
├── tests/                          # Test files
│   ├── test_audio_separation.py
│   ├── test_separate_reports.py
│   ├── test_transcription_separation.py
│   └── example_agent_usage.py
│
└── docs/                           # Documentation
    ├── AI_AGENT_README.md
    ├── AUDIO_SEPARATION_README.md
    └── TRANSCRIPTION_SEPARATION_README.md
```

## 🎯 Core Components (Used in main.py)

### Analysis Pipeline
1. **Video Processing**: Extract frames from video
2. **Visual Analysis**: Analyze visual content using SimpleVisualAnalyzer
3. **Pose Analysis**: Detect poses using PoseAnalyzer
4. **Emotion Analysis**: Detect emotions using legacy emotion_detection
5. **Audio Analysis**: Complete audio analysis using AudioAnalyzer

### Key Files
- `main.py` - Main entry point
- `src/analysis/` - Core analysis modules
- `src/config/settings.py` - Configuration
- `src/utils/` - Utility functions

## 🗑️ Cleaned Up Files

### Moved to `legacy/`
- Old emotion detection and report generation
- Old pose detection and report generation
- Old report generation utilities

### Moved to `tests/`
- All test files
- Example usage files

### Moved to `docs/`
- All documentation files
- README files for specific components

### Moved to `reports/`
- Generated analysis reports
- JSON data files

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py
```

## 📚 Additional Documentation

- [Analysis Structure](ANALYSIS_STRUCTURE.md) - Detailed analysis module organization
- [Audio Analysis Separation](docs/AUDIO_SEPARATION_README.md)
- [Transcription Analysis Separation](docs/TRANSCRIPTION_SEPARATION_README.md)
- [AI Agent Documentation](docs/AI_AGENT_README.md)

## 📝 Notes

- The `myenv/` directory contains a virtual environment and can be removed
- All core functionality is in the `src/` directory
- Analysis modules are organized by type (visual, audio, text)
- Legacy code is preserved but not used in the main pipeline
- Test files are organized in the `tests/` directory
- Documentation is centralized in the `docs/` directory
