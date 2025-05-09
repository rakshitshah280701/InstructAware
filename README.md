# InstructAware
This has all training Colabs, dataset links and other related resources related to InstructAware project

# T5-Based Instructional Narrative Generation
## Notebook: `Option2_T5_with_Final_Dataset.ipynb`

This notebook presents the complete training and evaluation pipeline for fine-tuning a **T5-small** model on a custom instructional scene understanding dataset. It forms **Option 2** of the modeling strategies explored in the InstructAware project.

### 🔍 Overview
- Fine-tunes a pre-trained T5 model for converting scene descriptions into instructional narratives.
- Designed for tasks involving vision-language understanding, where structured image metadata is mapped to natural language instructions.
  
---

### 📂 Dataset Details
You can find the csv files from the drive link given below for CSV dataset
The notebook expects CSV-formatted datasets located in the following structure:
<pre> ``` datasets/ ├── train_dataset_cleaned.csv ├── validation_dataset_cleaned.csv └── test_dataset_cleaned.csv ``` </pre>

Each file should contain the following columns:
- `INPUT TEXT`: Preprocessed textual input (e.g., detected object labels, bounding box positions).
- `OUTPUT TEXT`: Ground-truth narrative instructions describing the scene.






Dataset used for Detection - https://drive.google.com/drive/folders/1hzg9zE7_syzb83Le37Kzc8k87WpzKE7v?usp=sharing
Dataset used for Narrative Generation - https://drive.google.com/drive/folders/1ubRAzrbPvVPL2TcnK1H6fM6NXE9HL-Nz?usp=sharing
Narrative Dataset file for Training Transformer Models (CSV) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing
Narrative Dataset file for Training Transformer Models (JSONL) - https://drive.google.com/drive/folders/1zFOqAvPMl39hQgIty116fUdmpZrm9dkj?usp=sharing
Option 1 - Training Colab from Scratch - https://colab.research.google.com/drive/1orAc8W9sziHX3ZqXsgQgrK4s7U0JCUiS?usp=sharing
