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

### 1. Dataset Overview
Begin by printing a sample from the dataset to get a general idea of the data structure and content.

```
{'path': ['/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~ATM_LIMIT/602b982a05f96973d6794398.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~APP_ERROR/602ba2dabb1e6d0fbce92004.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~JOINT_ACCOUNT/602b9a59bb1e6d0fbce91f51.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~ATM_LIMIT/602ba562bb1e6d0fbce92066.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~CARD_ISSUES/602bae22bb1e6d0fbce9223a.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~FREEZE/602baece05f96973d6794506.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~APP_ERROR/602ba220bb1e6d0fbce91ff5.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~PAY_BILL/602baeb9bb1e6d0fbce92280.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~CARD_ISSUES/602bae44bb1e6d0fbce9224b.wav', '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~FREEZE/602b9c29bb1e6d0fbce91f79.wav'], 'audio': [{'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~ATM_LIMIT/602b982a05f96973d6794398.wav', 'array': array([ 6.91978494e-06,  1.71445790e-05, -7.28946179e-06, ...,
       -2.45892326e-04, -4.56941518e-04, -3.47009045e-04]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~APP_ERROR/602ba2dabb1e6d0fbce92004.wav', 'array': array([ 2.27957484e-04,  1.34665694e-04,  1.63455261e-05, ...,
       -3.22428328e-04, -2.37068045e-04, -9.29032976e-05]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~JOINT_ACCOUNT/602b9a59bb1e6d0fbce91f51.wav', 'array': array([ 1.75034511e-05,  2.17475026e-05, -1.82090153e-05, ...,
       -3.46608285e-05, -6.60970458e-04, -6.23460044e-04]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~ATM_LIMIT/602ba562bb1e6d0fbce92066.wav', 'array': array([ 1.54685840e-05,  1.51604472e-04,  2.27479512e-04, ...,
        2.24093892e-04,  4.77186331e-05, -9.20673556e-05]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~CARD_ISSUES/602bae22bb1e6d0fbce9223a.wav', 'array': array([-8.36749678e-07, -1.44678561e-05,  7.57565431e-07, ...,
       -7.45588768e-05, -2.48033630e-05,  4.71918793e-05]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~FREEZE/602baece05f96973d6794506.wav', 'array': array([-2.09534428e-05, -9.11740790e-06,  2.23169773e-05, ...,
        6.25934172e-06,  1.36021627e-05, -3.34269043e-06]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~APP_ERROR/602ba220bb1e6d0fbce91ff5.wav', 'array': array([ 2.35169195e-04,  3.09759416e-05, -2.35350482e-04, ...,
       -3.13565310e-04, -2.53080740e-04, -1.05573003e-04]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~PAY_BILL/602baeb9bb1e6d0fbce92280.wav', 'array': array([-1.52364373e-05,  7.63164135e-05,  2.60528876e-04, ...,
       -6.94348244e-04, -4.91044018e-04, -2.04122218e-04]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~CARD_ISSUES/602bae44bb1e6d0fbce9224b.wav', 'array': array([ 2.25627082e-05,  5.26002914e-05, -2.26977863e-05, ...,
        9.64273204e-06,  2.08271013e-05, -4.72028478e-06]), 'sampling_rate': 16000}, {'path': '/root/.cache/huggingface/datasets/downloads/extracted/28aa727f91fee90575c34956bab09d1716cfaf460c6afcba86a10f04a7d58b83/en-US~FREEZE/602b9c29bb1e6d0fbce91f79.wav', 'array': array([ 9.59142290e-06, -3.77536162e-06, -1.12360121e-05, ...,
        2.66744319e-05,  2.68959324e-04,  1.86235789e-04]), 'sampling_rate': 16000}], 'transcription': ["hello currently in my face cam and I wanted to know what's the max amount of money I can withdraw", "my transactions aren't loading", 'hello I was just calling to see if I can make it a joint account with my wife thank you', "I need to go to the ATM and are usually withdrawal about $200 I wanted can I up that and make it $300 when I'm at the ATM and make my maximum amount I can take out 300 just let me know thank you", "I'm trying to find out why my car keeps being declined", 'I need a freeze my card', 'if my app is not loading and I cannot access my account', 'hi my credit card payment is due I need to pay that bill thank you', "is my car because it doesn't work", 'honey I like to freeze my card'], 'english_transcription': ["hello currently in my face cam and I wanted to know what's the max amount of money I can withdraw", "my transactions aren't loading", 'hello I was just calling to see if I can make it a joint account with my wife thank you', "I need to go to the ATM and are usually withdrawal about $200 I wanted can I up that and make it $300 when I'm at the ATM and make my maximum amount I can take out 300 just let me know thank you", "I'm trying to find out why my car keeps being declined", 'I need a freeze my card', 'if my app is not loading and I cannot access my account', 'hi my credit card payment is due I need to pay that bill thank you', "is my car because it doesn't work", 'honey I like to freeze my card'], 'intent_class': [3, 2, 11, 3, 6, 9, 2, 13, 6, 9], 'lang_id': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}
```

### 2. Class Distribution
Understanding the distribution of classes is crucial for identifying any class imbalances that could affect model training and evaluation.
```
['abroad', 'address', 'app_error', 'atm_limit', 'balance', 'business_loan', 'card_issues', 'cash_deposit', 'direct_debit', 'freeze', 'high_value_payment', 'joint_account', 'latest_transactions', 'pay_bill']
```

### 3. Waveform Analysis
Visualizing the waveform of audio samples helps in understanding the amplitude variations over time, providing insights into the audio signal's characteristics.

![image](https://github.com/aridofflimits/Whisper-Speech-Recognition/assets/147245715/74d0905e-ef6d-4b72-b1bc-2aa86d8c9258)

### 4. Spectrogram Analysis
Spectrograms offer a visual representation of the spectrum of frequencies in the audio signal as they vary with time, which is valuable for feature extraction and understanding the signal's frequency content.

![image](https://github.com/aridofflimits/Whisper-Speech-Recognition/assets/147245715/9c6d0165-a39a-4209-a703-483e4cca58de)

### 5. MFCCs (Mel-Frequency Cepstral Coefficients)
MFCCs are commonly used features in audio processing. Visualizing MFCCs can help in understanding the timbral and spectral features of the audio signals.

![image](https://github.com/aridofflimits/Whisper-Speech-Recognition/assets/147245715/2442216e-efe1-4825-8210-8004e42424e9)

### 6. FFT (Fast Fourier Transform)
FFT analysis converts the signal from the time domain to the frequency domain, providing insights into the dominant frequencies present in the audio signal.

![image](https://github.com/aridofflimits/Whisper-Speech-Recognition/assets/147245715/3f4249a7-d54e-4e44-80c9-706f12ae1678)


### 7. Class Distribution Visualization
Plotting the class distribution helps in visualizing the frequency of each intent class within the dataset, highlighting potential imbalances.

![image](https://github.com/aridofflimits/Whisper-Speech-Recognition/assets/147245715/f099b864-32b3-4d17-96c9-1a66467a1a28)

Each step involves specific code snippets and visualizations that contribute to a comprehensive exploratory data analysis, aiding in the understanding and preprocessing of the dataset for model training.

---
## Import Pretrained Model and Load its Preprocessor

### 1. Importing Pretrained Model
The model is instantiated using a pretrained Whisper model, specifically the "openai/whisper-medium" version, which is designed for speech-to-text tasks.

### 2. Load Preprocessor
A preprocessor is loaded to match the Whisper model's expected input format. It's set to process English language audio for transcription tasks.

---
## Preprocessing

### 1. Train Test Split
The dataset is divided into training and test sets, with 20% of the data reserved for testing, to evaluate the model's performance on unseen data.

### 2. Resampling Input Data
Audio data is resampled to match the Whisper model's expected sampling rate, ensuring uniformity across all inputs and optimizing model performance.

### 3. Building Preprocess Function
A custom function prepares the dataset by applying the preprocessor to each audio sample, converting them into a format suitable for the model and calculating the duration of each audio clip in seconds.

### 4. Mapping Preprocess Function to Dataset
The preprocess function is applied to the entire dataset, removing irrelevant columns and transforming the audio data into a model-ready format.

### 5. Filtering Data
Audio samples longer than 30 seconds are filtered out to maintain consistency and manage computational load, focusing on shorter, potentially more informative audio segments.

### 6. Data Collator
A specialized data collator is defined to handle the unique requirements of padding speech-to-text model inputs and labels, ensuring proper tensor shapes and masking padding in labels for accurate loss calculation.

### 7. Instantiating the Data Collator
The data collator is initialized with the preprocessor, ready to collate batches of processed audio data for training the model, optimizing the data flow into the model during training sessions.

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



