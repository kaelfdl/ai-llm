# AI Large Language Model
## Overview
This repository contains programs designed to use large language models (i.e text generation, sematic analysis, etc.)

## Repository Structure
Each specific use case is under its own folder and are named accordingly.

## Environment Setup
First, clone the repository.
```
git clone https://github.com/kaelfdl/ai-llm
cd ai-llm
```

Next, create a python virtual environment and install the dependencies.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Example
For instance, we'll run the masked text generator program using the BERT large language model.
```
python bert/mask.py
Text: What is dog [MASK] doing on top of the table?
What is the dog doing on top of the table?
What is the dog sitting on top of the table?
What is the dog hiding on top of the table?
```
