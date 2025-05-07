<a id="readme-top"></a>

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Jax922/VLM-EQ">
    <img src="images/logo.png" alt="Logo" width="800" height="auto">
  </a>

  <!-- <h3 align="center">VLM-EQ</h3> -->

  <!-- <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p> -->
</div>


# AICA-VLM Benchmark

This project provides a benchmark framework for evaluating Vision-Language Models (VLMs) on **emotion understanding**,  **emotion reasoning** and   **emotion-guided content generation** tasks.
It is designed for standardized evaluation across multiple datasets and task formulations.

---

## 🛠 Installation

### 📦 For Users

Install the minimal runtime environment:

```bash
# Install in editable mode (recommended for CLI use)
pip install -e .

# Or traditional method
pip install -r requirements.txt
```

### 🧑‍💻 For Develope
To contribute or extend this project, follow the development setup below:
```bash
# 1. Create and activate a virtual environment (recommended)
conda create -n aica-vlm
conda activate aica-vlm

# 2. Install core and dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Set up pre-commit hooks
pre-commit install
```

Run pre-commit on all files:
```bash
pre-commit run --all-files
```

## 📚 Usage
Once installed, use the CLI tool aica-vlm to run dataset construction and instruction generation.
### Build Dataset
```bash
aica-vlm build-dataset run benchmark_datasets/example.yaml --mode random
```

* mode: random(default), balanced

### Build Instruction
```bash
# For Base instruction generation
aica-vlm build-instruction run benchmark_datasets/example.yaml

# For Chain of Thought (CoT) generation
aica-vlm build-instruction run-cot benchmark_datasets/example_CoT.yaml
```

### Run Evaluation or Benchmark
```bash
aica-vlm benchmark benchmark_datasets/example.yaml
```
