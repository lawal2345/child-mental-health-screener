# Child Mental Health Screening Tool

## Overview

This is a research prototype demonstrating responsible AI for early identification in child mental health pathways. The tool uses Microsoft's Phi-3-mini-4k-instruct model to analyze text and identify potential indicators of anxiety and depression in children, aligned with the THRIVE framework for mental health support.

## Use the Tool on HuggingFace
Access the tool on Huggingface via this link: https://huggingface.co/spaces/lawalll/child-mental-health-screener

## Features

- **Structured Assessment**: Provides risk level categorization (Low/Moderate/High)
- **Clinical Indicators**: Identifies specific anxiety and depression indicators
- **Explainable AI**: Includes clinical reasoning for all assessments
- **Confidence Scoring**: Model expresses uncertainty levels
- **Age-Appropriate Analysis**: Tailored for different age groups and contexts

## Technical Approach

- **Model**: Microsoft Phi-3-mini-4k-instruct
- **Method**: Structured prompt engineering with clinical reasoning
- **Framework**: Gradio for user interface
- **Principles**: Transparency, interpretability, and human oversight

## Responsible AI Principles

1. **Transparency**: All reasoning processes are explained
2. **Interpretability**: Clear identification of indicators
3. **Confidence Scoring**: Explicit uncertainty quantification
4. **Human Oversight**: Designed to support, not replace, clinical judgment
5. **Safety**: Multiple disclaimers and limitations clearly stated

## Usage

1. Enter text describing a child's feelings, behaviors, or concerns
2. Select the appropriate age group or context
3. Click "Analyze" to receive a structured assessment
4. Review the results including risk level, indicators, and recommendations

## Important Disclaimers

**RESEARCH PROTOTYPE ONLY**
- This is NOT a diagnostic instrument
- Should NOT be used for clinical decision-making without professional oversight
- Developed for research and demonstration purposes only
- Requires human clinical supervision for any practical application

## Validation

This prototype demonstrates the technical approach. The full research project will involve:
- Validation using CADRE (Child and Adolescent Data Resource)
- Federated analytics across multiple healthcare trusts
- External validation with longitudinal datasets
- Systems engineering approaches for practical implementation

## Citation

If you use this tool or approach in your research, please reference:

> Lawal, J. (2025). Child Mental Health Screening Tool: Responsible AI for Early Identification.
