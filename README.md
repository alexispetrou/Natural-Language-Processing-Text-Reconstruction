# Natural Language Processing Project: Text Reconstruction

## Project Description
This project explores various Natural Language Processing (NLP) techniques for semantic reconstruction of text, focusing on enhancing intelligibility and communicative effectiveness while preserving original meaning. The implementation compares manual automata approaches with pre-trained pipeline solutions for grammar correction and text simplification.

## Team Members
- Alexios Petrou (P22142)
- Lampros Alexandris (P22007)
- Sotirios Mpratos (P22113)

## Table of Contents
1. [Introduction](#introduction)
2. [Methodologies](#methodologies)
   - [Manual Automata](#manual-automata)
   - [Pipeline Reconstruction](#pipeline-reconstruction)
3. [Experiments and Results](#experiments-and-results)
4. [Bibliography](#bibliography)

## Introduction
Semantic reconstruction plays a crucial role in enhancing text intelligibility. Unlike syntactic correction, semantic reconstruction interprets intended meaning and rephrases text while maintaining or enhancing that meaning. This project implements NLP techniques including:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Cosine similarity for semantic comparison
- Automated grammar correction
- Text simplification

## Methodologies

### Manual Automata
Two distinct approaches for sentence reconstruction:

#### First Approach
- Uses ChatGPT-generated sentences as targets
- Implements regular expressions for tokenization
- Creates context-free grammars for original and target sentences
- Compares POS tags to reconstruct sentences

#### Second Approach
- Employs TreeBankTokenizer for tokenization
- Implements context-aware grammar rules:
  - Subject-verb agreement correction
  - Article insertion
  - Pronoun number agreement
  - Negative verb form correction

### Pipeline Reconstruction
Three Hugging Face transformer pipelines were compared:

1. **vennify/t5-base-grammar-correction**
   - T5 architecture fine-tuned for grammar correction
   - Text-to-text generation framework
   - Handles nuanced, context-dependent errors

2. **prithivida/grammar_error_correcter_v1**
   - Distilled T5 backbone optimized for speed
   - Focuses on surface-level grammatical errors
   - Minimalist output strategy

3. **sshleifer/distilbart-cnn-12-6**
   - Text simplification pipeline
   - Rewrites complex sentences into simpler forms
   - Improves readability while preserving meaning

## Experiments and Results

### Manual Automata Development
- Tested on sample sentences with grammatical errors
- Compared results against ChatGPT reconstructions
- Achieved 0.82 cosine similarity with target sentences

### Pipeline Development
Processed two problematic texts through each pipeline:

**Text 1 Example:**
Original: "Hope you too, to enjoy it as my deepest wishes."
- vennify: "Hope you too, enjoy it as my deepest wishes."
- prithivida: "Hope you too, to enjoy it as my deepest wishes."
- distilbart: Extended explanation with additional context

**Text 2 Example:**
Original: "the updates was confusing as it not included the full feedback"
- All pipelines corrected to: "the updates were confusing as they did not include the full feedback"

### Pipeline Comparison
| Metric               | vennify | prithivida | distilbart |
|----------------------|---------|------------|------------|
| Grammar Correction   | ✔       | ✔          | ✘          |
| Semantic Preservation| High    | Medium     | High       |
| Readability          | Medium  | Medium     | High       |
| Speed                | Medium  | Fast       | Slow       |

## Bibliography
1. Natural Language Processing with Python
2. Natural Language Processing Recipes: Unlocking Text Data with Machine Learning and Deep Learning using Python
3. Overview of Natural Language Processing (Harvard)
4. Natural Language Processing (NLP) in Healthcare
