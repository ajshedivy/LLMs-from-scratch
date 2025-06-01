# Build an LLM from Scratch - Demo Materials 🚀

**Author:** [Adam Shedivy](https://github.com/ajshedivy)

**Based on:** [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by [Sebastian Raschka](https://sebastianraschka.com)

This directory contains comprehensive educational materials that guide you through building a Large Language Model from the ground up, combining theory with hands-on implementation for a complete learning experience.

---

## 🎯 What You'll Find Here

### 📚 **Complete Tutorial Notebook**
- **`build-llm-from-scratch.ipynb`**: A comprehensive, presentation-ready notebook covering the entire LLM development pipeline from basic tokenization to production-ready text generation.

### 🔬 **Interactive Demo Scripts**
- **`gpt_visual.py`**: GPT model architecture visualization and exploration
- **`gpt_visual_train.py`**: Training demonstration with visual progress monitoring  
- **`gpt_visual_attention.py`**: Interactive attention mechanism visualization

### 📊 **Supporting Materials**
- **Training plots and visualizations** showing learning progress
- **Example outputs** demonstrating model capabilities at different stages

---

## 📖 Learning Journey Overview

### **1. Text Data Processing & Tokenization**
- Convert raw text into model-processable tokens
- Implement Byte-Pair Encoding (BPE) for robust tokenization
- Create efficient data loading pipelines with sliding windows
- Transform discrete tokens into continuous vector embeddings

### **2. Attention Mechanisms Deep Dive**
- Understand how models focus on relevant contextual information
- Implement Query-Key-Value attention systems from scratch
- Explore multi-head attention for parallel relationship processing
- Visualize attention patterns with interactive tools (BertViz)

### **3. Complete GPT Architecture Implementation**
- Build layer normalization for training stability
- Implement GELU activation and feed-forward networks
- Add residual connections for deep network gradient flow
- Assemble complete Transformer blocks

### **4. End-to-End Model Training**
- Implement cross-entropy loss functions and perplexity metrics
- Build comprehensive training loops with progress monitoring
- Detect overfitting through validation loss tracking
- Analyze training dynamics and performance patterns

### **5. Advanced Text Generation Strategies**
- Control output randomness with temperature scaling
- Implement top-k sampling for quality-diversity balance
- Create production-ready decoding strategies
- Build interactive generation systems with customizable parameters

### **6. Production Integration & Pretrained Models**
- Load state-of-the-art OpenAI GPT-2 weights (124M-1558M parameters)
- Compare small-scale training vs. massive pretraining results
- Integrate custom implementations with industry-standard models
- Deploy practical text generation applications

---

## 🛠️ Setup & Installation

Before running these materials, ensure you have the proper Python environment. For comprehensive setup instructions, see the [main setup guide](../setup/README.md).

### **Quick Start (Recommended)**
If you already have Python installed:
```bash
# Navigate to repository root
cd /path/to/LLMs-from-scratch

# Install dependencies directly
pip install -r requirements.txt
```

### **Alternative: UV Package Manager**
For advanced dependency management:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with UV
uv pip install -r requirements.txt
```

### **Cloud Options**
- **Google Colab**: Run directly in browser with GPU support
- **Lightning AI Studio**: Persistent cloud environment with VSCode/Jupyter
- **Local with GPU**: Automatic NVIDIA GPU acceleration if available

See [setup/README.md](../setup/README.md) for detailed instructions on:
- Python environment setup preferences
- Docker DevContainer alternatives  
- Cloud platform configuration
- VSCode extensions and recommendations

### **Verify Installation**
```bash
# Test basic imports
python -c "import torch, tiktoken, matplotlib; print('✅ All dependencies ready!')"
```

---

## 🚀 Getting Started

### **Option 1: Complete Tutorial (Recommended)**
Launch the comprehensive notebook for a guided learning experience:
```bash
jupyter lab build-llm-from-scratch.ipynb
```

### **Option 2: Interactive Demos**
Explore specific components with visualization scripts:

```bash
# Visualize GPT architecture
uv run python gpt_visual.py

# Watch model training in real-time
uv run python gpt_visual_train.py

# Explore attention mechanisms
uv run python gpt_visual_attention.py
```

---

## 🎓 Skills You'll Master

### **Technical Expertise**
✅ **Deep Learning Fundamentals:** Attention, training pipelines, optimization  
✅ **NLP Specialization:** Tokenization, language modeling, text generation  
✅ **PyTorch Mastery:** Model implementation, training, deployment  
✅ **Production Skills:** Pretrained models, generation control, real applications

### **Practical Applications**
✅ **Custom LLM Development:** Build models for specific domains or tasks  
✅ **Text Generation Systems:** Create chatbots, content generators, writing assistants  
✅ **Research Foundation:** Understand cutting-edge architectures and techniques  
✅ **Industry Integration:** Work with production LLM systems and APIs

---

## 📖 Original Source & Attribution

This educational content adapts and extends concepts from Sebastian Raschka's excellent book:

**Citation:**
```bibtex
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```

**Original Repository:** [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

---

## 💡 Educational Philosophy

This tutorial emphasizes:
- **Deep Understanding:** Learn the "why" behind each component, not just the "how"
- **Hands-On Implementation:** Build everything from scratch for complete comprehension
- **Progressive Complexity:** Each section builds naturally on previous knowledge
- **Production Readiness:** Connect educational concepts to real-world applications
- **Interactive Learning:** Visualizations and experiments reinforce theoretical concepts

---

## 🌟 Ready to Build the Future?

Whether you're a student, researcher, or practitioner, these materials provide the foundation you need to understand, implement, and innovate with Large Language Models.

**Start your LLM journey today!** 🚀
