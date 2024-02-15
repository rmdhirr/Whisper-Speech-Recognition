# Whisper Speech Recognition

This repository contains the Python scripts and notebooks for a speech recognition project using the Whisper model, implemented in a Google Colab environment. The project focuses on leveraging advanced speech recognition technology to accurately transcribe spoken language, with a special emphasis on the MINDS-14 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Room for Improvement](#room-for-improvement)
- [Attachment](#attachment)

# Introduction

This project is centered on harnessing the Whisper Medium model to advance speech recognition capabilities, particularly in the realm of processing and understanding a wide spectrum of accents and dialects present in spoken language. The initiative is crafted to demonstrate the effectiveness of advanced speech recognition technologies in managing and interpreting complex audio data. By focusing on the challenges posed by the variability in spoken language, the project aims to explore and highlight the potential of the Whisper model in providing reliable and accurate transcriptions across a diverse range of linguistic characteristics.

---

# Dataset

The project utilizes the MINDS-14 dataset, a comprehensive collection of spoken language samples. The dataset features a variety of languages, accents, and dialects, making it an ideal choice for testing the robustness and versatility of the Whisper model in speech recognition tasks.

MINDS-14 serves as both a training and evaluation resource specifically designed for the intent detection task using spoken data. It covers 14 distinct intents that have been extracted from a commercial system in the e-banking domain. Each intent is associated with spoken examples presented in 14 diverse language varieties, encompassing a wide range of linguistic and phonetic characteristics.

### Data Fields
The data fields remain consistent across all splits of the dataset, providing a uniform structure for ease of use:

- `path` (str): The filesystem path to the audio file.
- `audio` (dict): An audio object that includes the loaded audio array, its sampling rate, and the path to the audio file.
- `transcription` (str): The verbatim transcription of the spoken content in the audio file.
- `english_transcription` (str): The English translation of the transcription, facilitating non-native speakers' understanding and use of the dataset.
- `intent_class` (int): A numerical identifier representing the classified intent of the spoken content.
- `lang_id` (int): A numerical identifier for the language variety of the audio sample.

### Data Splits
To accommodate various training and testing scenarios, the dataset is structured into splits. However, each configuration within the dataset currently includes only a "train" split. This split consists of approximately 600 examples, providing a substantial volume of data for developing and refining speech recognition models in the context of intent detection within the e-banking domain.

---

# Methodology
## Exploratory Data Analysis



---

## Installation

To run the scripts, the following dependencies are required:
- datasets
- matplotlib
- librosa
- seaborn
- evaluate
- jiwer
- accelerate
- transformers

Install all dependencies using the following command: pip install datasets matplotlib librosa seaborn evaluate jiwer accelerate transformers

## Usage

You can try out the model here: [Google Colab](https://colab.research.google.com/drive/1GPmQMYNKdDdoFOsoSsziuE-MTGOYqrzJ#scrollTo=Lxu2nMfW2guX)



