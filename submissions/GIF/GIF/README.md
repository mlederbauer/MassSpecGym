# General Intelligence-based Fragmentation (GIF): A framework for peak-labeled spectra simulation


Generative Iterative Fragmentation (**GIF**) is a framework for spectra simulation using large language models (LLMs) such as GPT.


This repository contains code to implement the General Intelligence-based Fragmentation (**GIF**) framework. GIF performs spectra simulation for mass spectra annotation and uses a "fragment" then "score" approach.
This repository specifically contains the full GIF pipeline, including data preprocessing, fragment generation, iterative model querying, evaluation scripts, and an example downstream application.


---

## Overview

The GIF approach decomposes spectra simulation into smaller, interpretable steps:

1. **Fragment generation**  
   Molecules are broken down into valid fragments, represented as SELFIES.

2. **Intensity prediction**  
   The intensity value of each generated fragment is predicted. 

Running the pipeline also generates the MassSpecGym QA-sim dataset, a standardized QA-style benchmark for evaluating molecular reasoning performance from mass spectrometry data.

---
## Repository Structure

```text
GIF/
├── preprocess_data.py         # Prepare spectra and fragment data
├── single_query.py            # Send a single LLM query
├── query.py                   # Batch and parallel query manager
├── run_GIF_GPT.py             # Run the full GIF pipeline with GPT models
├── run_GIF_LLM.py             # Alternate entry point for non-GPT or local models
├── metrics.py                 # Evaluate generated fragments and reconstructions
├── iterate_fragmentation.py   # Iterative refinement on fragmentation 
├── iterate_intensity.py       # Iterative refinement on intensity prediction
├── finetuning.py              # Fine-tune GPT models
├── prompts.py                 # Templates to create prompts
└── application_task.py        # Example downstream application of GIF
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HassounLab/GIF.git
cd GIF
```

### 2. Create and activate a Python environment

It’s recommended to use **conda** to keep dependencies isolated:

```bash
conda create -n gif python=3.10
conda activate gif
```




### 3. Install dependencies

Using yaml file:

```bash
conda env update --file environment.yml
conda activate gif
```



### 4. Set your OpenAI API key

The code uses the OpenAI API for LLM-based fragment reasoning.

You can set it as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or pass it directly as a command-line argument:

```bash
python GIF/run_GIF_GPT.py --api_key your_api_key_here
```



---

## Input Data

If you provide your own molecular data, inputs are typically JSON files containing fields such as identifiers, SELFIES of the query molecule, and experiment settings.

Make sure your data format matches what the preprocessing scripts expect — see `preprocess_data.py` for schema details.

## Running the Full GIF Pipeline

Run the full GIF workflow using GPT-based reasoning:

```bash
python GIF/run_GIF_GPT.py \
    --api_key YOUR_API_KEY \
    --data_path ./data/ \
    --save_path ./output/
```


---




## Example Application

The file [`GIF/application_task.py`](GIF/application_task.py) demonstrates how to use GIF outputs for downstream applications. For example, we compare two molecules to a query spectrum using GIF.

Run the example application with:

```bash
python GIF/application_task.py --api_key YOUR_API_KEY
```

---

## Generating the MassSpecGym QA-sim Dataset

GIF automatically produces the **MassSpecGym QA-sim dataset** during the end-to-end pipeline run. 
It is derived from the MassSpecGym benchmark dataset[[1]](#1) and allows comparison to other baseline methods evaluated using MassSpecGym.
This dataset provides standardized question–answer pairs for evaluating LLMs on molecular reasoning from mass spectrometry data.

To generate it:

```bash
python GIF/run_GIF_GPT.py --api_key YOUR_API_KEY 
```

## References

<a id="1">[1]</a> 
Bushuiev, Roman, et al. "MassSpecGym: A benchmark for the discovery and identification of molecules." Advances in Neural Information Processing Systems 37 (2024): 110010-110027.