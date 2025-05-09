# InstructAware
This has all training Colabs, dataset links and other related resources related to InstructAware project

# Transformer-Based Instructional Narrative Generation (From Scratch)  
## Notebook: `Option1_Transformer_trained_From_Scratch_With_Metrics.ipynb`

This notebook demonstrates how to build and train a Transformer model **from scratch** for the task of generating instructional narratives from scene input data. It represents **Option 1** in the modeling strategies of the InstructAware project.

---

### 🛠️ Key Features

- Implements a custom Transformer architecture using PyTorch or TensorFlow.
- Trains the model end-to-end using preprocessed CSV datasets (`INPUT TEXT` → `OUTPUT TEXT`).
- Includes custom training loop and attention mechanisms.
- Tracks training and validation performance over epochs.
- Visualizes loss, learning rate, and accuracy using Matplotlib and TensorBoard (if enabled).

---

### 📂 Dataset Format
Expected CSV file input format:
datasets/
- train_dataset_cleaned.csv  
- validation_dataset_cleaned.csv  
- test_dataset_cleaned.csv

  
Each file must contain:
- `INPUT TEXT`: bounding box + OCR text.
- `OUTPUT TEXT`: Instructional text describing the scene.

# T5-Based Instructional Narrative Generation
## Notebook: `Option2_T5_with_Final_Dataset.ipynb`

This notebook presents the complete training and evaluation pipeline for fine-tuning a **T5-small** model on a custom instructional scene understanding dataset. It forms **Option 2** of the modeling strategies explored in the InstructAware project.

### 🔍 Overview
- Fine-tunes a pre-trained T5 model for converting scene descriptions into instructional narratives.
- Designed for tasks involving vision-language understanding, where structured image metadata is mapped to natural language instructions.
  
---

### 📂 Dataset Details
You can find the csv files from the drive link given below for CSV dataset
The notebook expects CSV-formatted datasets to be organized as follows:

datasets/
- train_dataset_cleaned.csv  
- validation_dataset_cleaned.csv  
- test_dataset_cleaned.csv


Each file should contain the following columns:
- `INPUT TEXT`: Bounding Box + OCR text.
- `OUTPUT TEXT`: Ground-truth narrative instructions describing the scene.

# GPT-3.5 Fine-Tuning via OpenAI API  
## Notebook: `Option_3_Retraining_Using_GPT_3_5.ipynb`

This notebook demonstrates how to fine-tune OpenAI’s `gpt-3.5-turbo` model using custom JSONL datasets for the task of generating narrative descriptions from scene-based inputs. This is **Option 3** in the InstructAware model comparison.

---

### 🚀 Key Features

- Uploads `.jsonl` datasets to OpenAI via API.
- Initiates and monitors fine-tuning jobs on `gpt-3.5-turbo`.
- Evaluates model performance on a held-out test set.
- Supports generation of predictions and saving them to `.jsonl` or `.csv` formats.

---

### 🔐 API Key Setup (for Google Colab)

This notebook uses `userdata.get("OpenAiKey")` to access your OpenAI API key securely.

#### 👉 How to add your API key in Colab:

1. Click the **🔐 key icon** in the left sidebar to open **"Secrets"**.

3. Add a new secret:
   - **Key:** `OpenAiKey`  
   - **Value:** *your OpenAI API key* (get it from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys))





- Dataset used for Detection - https://drive.google.com/drive/folders/1hzg9zE7_syzb83Le37Kzc8k87WpzKE7v?usp=sharing
- Dataset used for Narrative Generation - https://drive.google.com/drive/folders/1ubRAzrbPvVPL2TcnK1H6fM6NXE9HL-Nz?usp=sharing
- Narrative Dataset file for Training Transformer Models (CSV) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing
- Narrative Dataset file for Training Transformer Models (JSONL) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing

