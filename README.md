# InstructAware
### InstructAware is a System to assist low-vision people with Intelligent Sign Awareness. 
[Demo of InstructAware App -](https://youtu.be/N5SmVK5J02c?si=dMQxnNRq3QCTlcYQ) (see seperate [github](https://github.com/rakshitshah280701/SmartSignNavigation-Android.git] ) )

This has all training Colabs, dataset links and other related resources related to the creation of Models Tests & Used in the InstructAware project.

### There were Models created and Compared:   Transformer-Based from Scratch, T5, GPT-3.5, DeepSeek R1 Distill Llama 8B

# Dataset for ALL models:
The following is an example of the data in our dataset.  It consists of


### Image

### Sign Info
{
  "task": "Generate a natural language description based on detected text and bounding boxes.",
  "detected_signs": [
    {
      "text": "CITY JEWELRY 10.14.18. K ORO HOUR NAME PLATES BIG SALE",
      "coordinates": [0.8871527777777778, 0.7033333333333334, 0.10590277777777778, 0.0961111111111111]
    },
    {
      "text": "14k ORO BIG SALE",
      "coordinates": [0.5253472222222222, 0.7027777777777777, 0.04201388888888889, 0.07277777777777777]
    },
    {
      "text": "HOUR NAME PLATE BIG SALE",
      "coordinates": [0.5163194444444444, 0.8155555555555556, 0.06875, 0.051111111111111114]
    },
    {
      "text": "JEWELRY CITY",
      "coordinates": [0.48194444444444445, 0.49777777777777776, 0.26944444444444443, 0.09277777777777778]
    },
    {
      "text": "NAME PLATES 10/14K GOLD WE BUY GOLD",
      "coordinates": [0.596875, 0.7027777777777777, 0.06701388888888889, 0.07388888888888889]
    },
    {
      "text": "CITY JEWELRY",
      "coordinates": [0.3645833333333333, 0.7094444444444444, 0.0375, 0.034444444444444444]
    },
    {
      "text": "1 HOUR REPAIR WE BUY GOLD",
      "coordinates": [0.45208333333333334, 0.8044444444444444, 0.035069444444444445, 0.04777777777777778]
    }
  ],
  "description": ""
}

### Narrative
You’re in front of "CITY JEWELRY," where there are big sales on gold and name plates. They also offer quick repairs for gold items. You can find signs that say they buy gold too.


---

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
   - 
### 📂 Dataset Details
The notebook expects input in .jsonl format with OpenAI’s fine-tuning schema, you can find the jsonl dataset file at the drive link given below
```
{
  "messages": [
    {"role": "system", "content": "You generate detailed narratives from text."},
    {"role": "user", "content": "<input_text_here>"},
    {"role": "assistant", "content": "<target_output_text>"}
  ]
}
```

# Option 4: Training DeepSeek-R1 using Unsloth  
## Notebook: `Option4_Training_DeepSeek_using_Unsloth_FixedSplitData.ipynb`

This notebook showcases how to fine-tune the **DeepSeek-R1-Distill-LLaMA-8B** model using the [Unsloth](https://github.com/unslothai/unsloth) library, which enables highly efficient fine-tuning of large language models on consumer hardware such as Google Colab.

---

### ⚙️ Key Features

- Utilizes `unsloth/DeepSeek-R1-Distill-Llama-8B`, a compact and instruction-tuned model.
- Implements **PEFT with LoRA adapters** to reduce training memory requirements.
- Employs `FastLanguageModel` API for model loading, formatting, and training.
- Loads fixed split data from CSV files (train/validation).
- Supports logging and checkpoint saving to Google Drive.

---

### 📂 Dataset Structure

Expected directory:
directory/
- train_dataset.csv
- validation_dataset.csv

  
Each CSV must contain:
- `INPUT TEXT`: structured scene input (e.g., object labels and coordinates)
- `OUTPUT TEXT`: human-written instructional narrative

---

### 🔐 API Key Setup (for Google Colab)

The notebook retrieves your Hugging Face access token securely using Colab secrets:

#### Steps:
1. Click the 🔐 "key" icon on the left sidebar in Colab.
2. Add a new secret:
   - **Key**: `DeepSeek`  
   - **Value**: your Hugging Face token  
     → You can get one from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

The key is accessed in the notebook using:
```python
hf_token = userdata.get("DeepSeek")
```

- Dataset used for Detection - https://drive.google.com/drive/folders/1hzg9zE7_syzb83Le37Kzc8k87WpzKE7v?usp=sharing
- Dataset used for Narrative Generation - https://drive.google.com/drive/folders/1ubRAzrbPvVPL2TcnK1H6fM6NXE9HL-Nz?usp=sharing
- Narrative Dataset file for Training Transformer Models (CSV) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing
- Narrative Dataset file for Training Transformer Models (JSONL) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing

