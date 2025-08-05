# Evaluation Benchmark Data Generation System

This repository contains a production-grade Python library for automatically curating large volumes of high-quality evaluation questions for language-model benchmarking. The system is designed to be modular, configurable, and scalable, allowing for flexible generation strategies and robust quality control.

## Table of Contents

1.  [Project Vision](https://www.google.com/search?q=%23project-vision)
2.  [System Architecture](https://www.google.com/search?q=%23system-architecture)
3.  [Getting Started: Installation & Setup](https://www.google.com/search?q=%23getting-started-installation--setup)
      * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      * [Installation](https://www.google.com/search?q=%23installation)
      * [Environment Variables](https://www.google.com/search?q=%23environment-variables)
4.  [Core Workflow & Key Components](https://www.google.com/search?q=%23core-workflow--key-components)
      * [Phase 1: Taxonomy-Driven Context Curation](https://www.google.com/search?q=%23phase-1-taxonomy-driven-context-curation)
      * [Phase 2: Grounded Question Generation](https://www.google.com/search?q=%23phase-2-grounded-question-generation)
      * [Phase 3: Automated Validation & Difficulty Scaling](https://www.google.com/search?q=%23phase-3-automated-validation--difficulty-scaling)
      * [Phase 4: Expert Human Review](https://www.google.com/search?q=%23phase-4-expert-human-review)
5.  [Command-Line Interface (CLI) & Usage](https://www.google.com/search?q=%23command-line-interface-cli--usage)
      * [Setup & Seeding](https://www.google.com/search?q=%23setup--seeding)
      * [Running the Generation Pipeline](https://www.google.com/search?q=%23running-the-generation-pipeline)
      * [Running Evaluations](https://www.google.com/search?q=%23running-evaluations)
      * [Utility Commands](https://www.google.com/search?q=%23utility-commands)
6.  [Directory Structure](https://www.google.com/search?q=%23directory-structure)

-----

## Project Vision

Our goal is to build a reliable and scalable "factory" for producing high-fidelity benchmark questions to test the legal reasoning of Large Language Models (LLMs). The system is built on a philosophy of **Grounded Generation**, where every question is derived from and verifiable against a specific set of source texts. This ensures accuracy and prevents the model from "hallucinating" facts. The entire pipeline is driven by configuration, making it highly flexible and extensible.

## System Architecture

The library is divided into several logical layers to ensure clear separation of concerns:

  * **Generation Layer (`rag_generator.py`):** Implements the core logic for generating questions. The `RAGGenerator` class takes a context bundle and a prompt to produce questions grounded in the provided text.
  * **Model-Provider Layer (`gemini_provider.py`, `perplexity_sonar_provider.py`):** Contains wrappers for various third-party LLM APIs. Each provider handles its own authentication, API request format, and error handling, making it easy to swap or mix models.
  * **Data-Access Layer (`taxonomy_manager.py`, `prompt_manager.py`):** Abstract access to our data stores (e.g., MongoDB). This layer is responsible for fetching taxonomies and prompt templates, decoupling them from the core application logic.
  * **Knowledge & Retrieval Layer (`faiss_retriever.py`, `index_builder.py`):** Manages the creation and querying of our document index (using FAISS). This layer is responsible for finding relevant passages for a given topic.
  * **Orchestration Layer (`main.py`, `bundle_builder.py`):** Wires all the components together. It reads configurations, manages the flow of data from one stage to the next (from taxonomy to bundle to question), and executes the pipeline.
  * **Evaluation Layer (`test_mcq.py`):** A separate module for evaluating the generated questions. It uses a panel of "judge" LLMs to automatically assess question difficulty and correctness.

-----

## Getting Started: Installation & Setup

Follow these steps to set up the project on your local machine.

### Prerequisites

  * Python 3.9+
  * MongoDB instance (local or remote)
  * For macOS users wanting to use FAISS:
      * Homebrew for installing dependencies.
      * `brew install swig cmake`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies in editable mode:**
    This command installs the project and its requirements. The `-e` flag allows you to make changes to the source code without needing to reinstall.

    ```bash
    pip install -e .
    ```

4.  **Special Installation for FAISS on macOS:**
    If the standard installation fails for `faiss-cpu`, run the following command to compile it from source:

    ```bash
    pip install --no-binary :all: faiss-cpu
    ```

### Environment Variables

The system relies on environment variables for configuration, especially for API keys.

1.  Create a file named `.env` in the root of the project.

2.  Add the following keys to the file, filling in your own credentials:

    ```env
    # API Keys for Model Providers
    GOOGLE_API_KEY="your_google_api_key"
    PERPLEXITY_API_KEY="your_perplexity_api_key"
    GROQ_API_KEY="your_groq_api_key"
    TOGETHERAI_API_KEY="your_together_api_key"
    HF_API_KEY="your_huggingface_api_key"
    # ... add any other keys you use

    # MongoDB Connection
    MONGO_URI="mongodb://localhost:27017/"
    MONGO_DB_NAME="your_db_name"

    # Directory Paths (defaults are usually fine)
    QUESTIONS_DIR="tmp/questions"
    EVAL_OUT="tmp/eval_results_models.json"
    EVAL_QUESTIONS_OUT="tmp/eval_per_question.json"
    EVAL_DEBUG_OUT="tmp/eval_debug_per_question.json"
    ```

-----

## Core Workflow & Key Components

The system operates in a four-phase pipeline:

#### Phase 1: Taxonomy-Driven Context Curation

The process starts with a curated **Legal Taxonomy**. The `TaxonomyManager` selects a topic, which is used as a query by the `FaissRetriever` to find the `top-k` relevant passages from our document index. The `BundleBuilder` then aggregates these passages into a **Context Bundle**.

#### Phase 2: Grounded Question Generation

The `RAGGenerator` takes the Context Bundle and, using a prompt from the `PromptManager`, instructs a "Generator" LLM (via a provider like `GeminiProvider`) to create questions that are factually grounded in the bundle's text.

#### Phase 3: Automated Validation & Difficulty Scaling

The `test_mcq.py` script takes the generated questions and presents them to a panel of diverse "Judge" LLMs. By measuring the consensus on the correct answer, we can automatically assign a difficulty score (Easy, Medium, Hard) to each question.

#### Phase 4: Expert Human Review

This is a manual, offline process where legal experts review the automatically-generated and scored questions to ensure the highest level of quality, accuracy, and fairness.

-----

## Command-Line Interface (CLI) & Usage

Here are the most common commands for interacting with the system.

### Setup & Seeding

First, you need to populate your database with the necessary data.

```bash
# To build the FAISS index from your source documents
# This needs to be done once, or whenever your source documents change.
python -m eval_data_gen.core.knowledge.index_builder

# To load your taxonomies (from YAML files) into MongoDB
python -m eval_data_gen.cli.main load-taxonomies-to-db

# To load your prompt templates into MongoDB
python -m eval_data_gen.cli.main load-prompts-to-db
```

### Running the Generation Pipeline

These commands orchestrate the full pipeline from retrieval to generation.

```bash
# Build retrieval bundles for each leaf in a taxonomy file
# A "bundle" is a collection of relevant passages for a topic.
eval-data-gen build-bundles \
    --taxonomy-path sample_data/taxonomy_law.yaml \
    --out-dir tmp/bundles_en \
    --k 20

# Generate questions using the created bundles
# This will use the models specified in the code (e.g., Gemini).
eval-data-gen generate-questions \
    --bundle-dir tmp/bundles_en \
    --n 5

# --- Alternative way to run the entire pipeline ---
# This command runs the pipeline defined in main.py
python main.py pipeline-run --taxonomy-dir sample_data --bundle-dir tmp/bundles --n 3 --k 4

# This command runs the pipeline via the CLI entry point
python -m eval_data_gen.cli.main pipeline-run --n 5 --k 10 --prompt-id mcq_conceptual --options 8
```

### Running Evaluations

Once questions are generated, use this command to evaluate them.

```bash
# Run the evaluation script with full debugging output
# This will test all generated questions against all judge models.
DEBUG_DUMP=1 PYTHONPATH=src python -m eval_data_gen.core.evaluation.test_mcq
```

### Utility Commands

```bash
# To manually test the retrieval system
eval-data-gen retrieve-faiss
eval-data-gen retrieve-big --k 4 --window 2

# To generate a randomized CSV of questions for creating evaluation forms
python -m eval_data_gen.core.utilities.generate_ques_csv
```

-----

## Directory Structure

A brief overview of the key files based on what you've provided:

```
.
├── src/
│   └── eval_data_gen/
│       ├── core/
│       │   ├── knowledge/
│       │   │   ├── index_builder.py   # Builds the FAISS index
│       │   │   └── faiss_retriever.py # Retrieves documents
│       │   ├── providers/
│       │   │   ├── gemini_provider.py         # Wrapper for Gemini API
│       │   │   └── perplexity_sonar_provider.py # Wrapper for Perplexity API
│       │   ├── evaluation/
│       │   │   ├── test_mcq.py          # Main evaluation script
│       │   │   └── test_rag_generatedQ.py # (Likely an alternative test script)
│       │   ├── data_access/
│       │   │   ├── taxonomy_manager.py # Manages taxonomies
│       │   │   └── prompt_manager.py   # Manages prompts
│       │   └── utilities/
│       │       └── generate_ques_csv.py # Utility for CSV generation
│       ├── cli/
│       │   └── main.py          # Entry point for command-line interface
│       ├── bundle_builder.py    # Creates context bundles
│       └── rag_generator.py     # Generates questions from bundles
├── main.py                    # Main script for running the pipeline
└── ...
```