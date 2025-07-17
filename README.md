# AI-Driven Analysis of Respiratory and Vocal Biomarkers for Health Diagnostics

This project explores how audio features extracted from short voice recordings can be used as potential health biomarkers. The provided script records a five second voice sample, extracts acoustic features using [OpenSMILE](https://audeering.github.io/opensmile/), and stores those values for further machine learning experiments.

## Directory Structure

- `Data/`
  - `samples/` – contains example audio data. Running the script saves new recordings here.
  - `features/` – CSV feature exports produced by the script.
- `Notebooks/`
  - `feature_extraction.ipynb` – placeholder notebook for experiments and analysis.
- `Scripts/`
  - `voice_feature_extractor.py` – records audio, computes MFCCs and OpenSMILE features, and saves them to `Data/features`.

## Installation

It is recommended to work inside a virtual environment. Install the required Python packages using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install sounddevice scipy librosa matplotlib numpy pandas opensmile
```

## Running the Voice Feature Extractor

From the repository root run:

```bash
python Scripts/voice_feature_extractor.py
```

The script records from your microphone, displays MFCC visualisations, and stores extracted features in `Data/features/recorded_features.csv`.

## Project Goals

The goal is to build a foundation for analysing respiratory and vocal signals to assist in health diagnostics. The expected outcome is a dataset of voice-derived features that can be used to train machine learning models for detecting potential health conditions.
