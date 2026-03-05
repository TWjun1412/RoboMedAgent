# RoboMedAgent

Analysis Code for: **[RoboMedAgent: Harmonizing Active Listening, Consulting, and Caring to Deliver Accurate and Empathetic Healthcare Guidance]**

# 🌟 Overview

This repository contains the official implementation of **RoboMedAgent**, a collaborative multi-agent framework designed to rigorously denoise raw medical dialogues. Unlike generic LLMs, RoboMedAgent employs a **Parallel Feedback Strategy** to rectify semantic noise and produces a **Traceable EditList** to ensure clinical transparency and verifiability.

The framework utilizes a Parallel Feedback Strategy to rectify five categories of semantic noise and generates a Traceable EditList to ensure clinical transparency. Even under 100% noise perturbation, RoboMedAgent achieves near-perfect restoration performance, outperforming generic LLMs like GPT-4o.

---

# 📂 Repository Structure

The scripts are organized into four core functional modules mirroring the workflow described in the manuscript:

### 1. Error Detector Module (Parallel Feedback Strategy)

This module simultaneously identifies five categories of semantic noise to ensure no clinical information is misinterpreted.

| Class Name | Noise Category | Description | Key Methods/Packages |
| --- | --- | --- | --- |
| `SpellChecker` | **Spelling (SPL)** | Detects ASR-induced typos while protecting professional medical terms. | `symspellpy`, Medical Dictionary |
| `RepetitionDetector` | **Repetition (RPT)** | Identifies redundant oral fragments and n-gram stutters. | `collections.Counter`, `re` |
| `GECTagger` | **Grammar (GRM)** | Corrects syntactic fragmentation using seq2seq models. | `transformers` (GEC Model) |
| `CombinedMedicalDetector` | **Ambiguity (AMB) & Interference (NOS)** | LLM-powered detection of ambiguous clinical terms and non-medical noise. | GPT-4o-mini, Medical KG |

### 2. Semantic Editing Module (Traceable EditList)

Translates detection signals into precise modifications while maintaining a verifiable audit trail.

| Class Name | Description | Key Features |
| --- | --- | --- |
| `EditManager` | Organizes edits into the **Traceable EditList**, calculating confidence and cost for each change. | Edit Tracking, Cost Scoring |
| `EditorPipeline` | Executes targeted semantic restoration, focusing on clinical Word Sense Disambiguation (WSD). | Contextual Interpretation |

### 3. Output Control Module (Iterative Quality Loop)

A multi-agent arbitration system that resolves edit conflicts and ensures output standards.

| Class Name | Description | Key Methods/Packages |
| --- | --- | --- |
| `ArbiterPipeline` | Resolves conflicts between overlapping edit candidates using a priority-based selection strategy. | Conflict Resolution Logic |
| `DenoisingQualityGEval` | Performs multi-dimensional quality assessment (Accuracy, Integrity, Smoothness) to trigger iterative refinement. | **G-Eval Framework** |

### 4. Interactive & Evaluation Module

The final stage for clinical recommendation and framework performance analysis.

| Class Name | Description | Key Metrics |
| --- | --- | --- |
| `Recommendation_Gen` | Generates empathetic suggestions based on restored text and reconstructed patient intent. | Safety, Empathy, Completeness |
| `EvaluationMetrics` | Comprehensive calculation of term retention rates and semantic similarity. | Term Retention, Cohen's Kappa |

---

# 🛠️ Prerequisites & Installation

To run RoboMedAgent, ensure you have **Python >= 3.8** and the following dependencies:

```bash
# NLP & Transformer dependencies
pip install transformers symspellpy nltk sentence-transformers

# LLM API support
pip install openai

# Medical dictionary & NLTK data
python -c "import nltk; nltk.download('punkt')"

```

**Note:** You will need an OpenAI API key to power the *Parallel Feedback Strategy* and the *Output Control Module*.

---

# 🚀 Quick Start

```python
from robomedagent import RoboMedAgent

# Initialize the framework with your clinical dictionary and API key
agent = RoboMedAgent(
    medical_dict_path="path/to/dict.json",
    api_key="your-openai-api-key"
)

# Process a noisy clinical dialogue
raw_dialogue = "Pt: I... I feel a bit, uh, cough, cough... is it COVID? (background noise: barking)"

# The system executes: Detection -> Editing -> Output Control
result = agent.run_restoration(raw_dialogue)

print(f"Restored Text: {result.final_text}")
print(f"Traceable EditList: {result.edit_list}")

```

---

# 📊 Performance Metrics

As validated in the study across 220,000+ dialogues:

* **Restoration Accuracy**: 4.995/5.000 (at 100% noise)
* **Medical Term Retention**: Significantly higher than standard GPT-4 models.
* **Clinical Safety**: Zero-hallucination restoration via the Traceable EditList.

---

# 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{robomedagent2024,
  title={RoboMedAgent: Harmonizing Active Listening, Consulting, and Caring to Deliver Accurate and Empathetic Healthcare Guidance},
  author={...},
  year={2024}
}

```

---

*This repository is for research purposes. Ensure compliance with local regulations regarding medical data (e.g., HIPAA) when using LLMs.*
