# LongAudioBench

**LongAudioBench: A Comprehensive Benchmark for Long-Context Audio Understanding**

A benchmark suite designed to expose fundamental architectural limitations of Audio-Large Language Models (LALMs) on long-form audio (15-60+ minutes). Targets ICLR/NeurIPS/ICML submission.

## 🎯 Core Tasks (4 Failure Modes)

| Task | Failure Mode | Duration | Key Challenge |
|------|--------------|----------|---------------|
| **Acoustic Needle-in-Haystack (ANiH)** | Long-range acoustic retrieval / KV-cache eviction | 60 min | Find 0.5s artifact buried in 1hr audio |
| **Speaker Diarization Drift** | Speaker identity tracking / Recency bias | 60 min | Identify speaker who returns after 40min silence |
| **Environmental Soundscape Timeline** | Speech overfitting / Non-speech scene analysis | 30 min | Pure environmental audio, no speech |
| **Acoustic Narrative Coherence** | Multi-hop cross-modal reasoning | 40 min | Link acoustic clue (5min) → semantic ref (20min) → alibi (35min) |

## 📁 Repository Structure

```
longaudiobench/
├── longaudiobench/
│   ├── tasks/              # Task implementations
│   │   ├── base.py         # Abstract base classes
│   │   ├── anih/           # Acoustic Needle-in-Haystack
│   │   ├── speaker_drift/  # Speaker Diarization Drift
│   │   ├── soundscape/     # Environmental Soundscape Timeline
│   │   └── narrative_coherence/  # Acoustic Narrative Coherence
│   ├── baselines/          # Model baselines
│   │   ├── cascaded        # Whisper + LLM
│   │   └── native          # SALMONN, Qwen-Audio, LTU
│   ├── metrics/            # Evaluation metrics
│   └── configs/            # YAML configurations
├── scripts/
│   ├── generate_data.py    # Generate task instances from public datasets
│   ├── run_evaluation.py   # Run baselines on tasks
│   └── analyze_results.py  # Paper-ready tables & figures
├── data/                   # Generated instances (gitignored)
└── results/                # Evaluation outputs (gitignored)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Prepare Public Datasets
Download and organize these public datasets (paths configured in `longaudiobench/configs/default.yaml`):

| Dataset | Use For | Download |
|---------|---------|----------|
| **AudioSet** | Backgrounds, events | [Link](https://research.google.com/audioset/) |
| **FSD50k** | Backgrounds, events | [Link](https://zenodo.org/record/4060432) |
| **VoxCeleb2** | Speaker drift | [Link](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) |
| **AMI** | Speaker drift | [Link](http://groups.inf.ed.ac.uk/ami/corpus/) |
| **LibriSpeech** | Speaker drift, stories | [Link](https://www.openslr.org/12/) |
| **LibriVox** | Narrative coherence | [Link](https://librivox.org/) |
| **DCASE 2024 Task 1** | Soundscapes | [Link](https://dcase.community/challenge2024/) |

### 3. Generate Task Instances
```bash
# Generate all tasks
python scripts/generate_data.py --tasks all --output-dir data/generated

# Or specific tasks
python scripts/generate_data.py --tasks anih speaker_drift --output-dir data/generated
```

### 4. Run Baselines
```bash
# Evaluate all models on all tasks
python scripts/run_evaluation.py --tasks all --models whisper_llama3 salmonn qwen_audio ltu

# Single task/model
python scripts/run_evaluation.py --tasks anih --models whisper_llama3
```

### 5. Analyze Results
```bash
python scripts/analyze_results.py --results-dir results --output-dir analysis
```

Outputs:
- `analysis/main_results_table.csv/.tex` — Main paper table
- `analysis/*_detailed_metrics.csv` — Per-task detailed metrics
- `analysis/paper_summary.md` — Paper-ready summary

## 📊 Evaluation Metrics

| Task | Primary Metric | Secondary Metrics |
|------|----------------|-------------------|
| ANiH | Composite (timestamp@5s + preceding sound IoU) | Timestamp error (s), Hit@1s, Hit@5s |
| Speaker Drift | Joint Accuracy (speaker + first appearance) | Speaker Acc, Appearance Hit@5s |
| Soundscape | Composite (Event F1 + Timeline IoU + Ordering + Overlap) | Event F1, Timeline IoU, Ordering Acc |
| Narrative | Composite (Verdict + Reasoning + Timestamp F1) | Verdict Acc, Reasoning Completeness, Citation F1 |

## 🔬 Baselines Supported

| Model | Type | Status |
|-------|------|--------|
| Whisper Large-v3 + Llama-3-8B | Cascaded | ✅ Implemented |
| SALMONN-7B | Native | 🚧 Template |
| Qwen-Audio-Chat | Native | 🚧 Template |
| LTU-7B | Native | 🚧 Template |

## 📝 Citation

```bibtex
@inproceedings{longaudiobench2025,
  title={LongAudioBench: Exposing Fundamental Limitations of Audio-LLMs on Long-Form Audio},
  author={Anonymous},
  booktitle={ICLR},
  year={2025}
}
```

## 🤝 Contributing

1. Add new tasks in `longaudiobench/tasks/`
2. Add new baselines in `longaudiobench/baselines/`
3. Extend metrics in `longaudiobench/metrics/`
4. Submit PR with tests

## 📄 License

MIT License - see LICENSE file.