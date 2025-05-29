
# 🦖 DINO-based Object Detection using Hugging Face Dataset

This project demonstrates how to perform **object detection** using the **DINO (DETR with Improved DeNoising Training)** model, fine-tuned on a dataset imported from the **Hugging Face Datasets Hub**.

## 🚀 Overview

- ✅ Model: [DINO (Facebook Research)](https://github.com/IDEA-Research/DINO)
- 📦 Dataset: Imported from Hugging Face (`coco2017` or any custom dataset)
- 🧠 Framework: PyTorch + Hugging Face + Transformers
- 🎯 Task: Object detection with bounding boxes and class labels
- 📊 Metrics: mAP (mean Average Precision), IoU
- 🖼️ Visualization: Matplotlib or OpenCV

---

## 📁 Project Structure

```

📦 dino-object-detection
├── 📂 data/                # Dataset (auto-downloaded from Hugging Face)
├── 📂 src/                 # Core training, evaluation, and utils
├── 📂 results/             # Sample predictions with bounding boxes
├── train.py               # Fine-tuning script
├── evaluate.py            # Evaluation & mAP calculation
├── visualize.py           # Visualization utility
├── requirements.txt       # Python dependencies
└── README.md              # You're here!

````

---

## 📚 Dataset

We use the `keremberke/coco2017` dataset from Hugging Face (or you can replace with a custom dataset).

```python
from datasets import load_dataset
dataset = load_dataset("keremberke/coco2017")
````

---

## 🏗️ Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/BadavathArjun/DINO.git
cd DINO
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 🔹 Training the Model

```bash
python train.py --dataset coco2017 --epochs 10 --lr 1e-4
```

### 🔹 Evaluation

```bash
python evaluate.py --model_path ./checkpoints/dino_best.pth
```

### 🔹 Visualization

```bash
python visualize.py --model_path ./checkpoints/dino_best.pth --num_images 5
```

---

## 📊 Results

* ✅ Achieved mAP of **XX.XX** on the validation set
* 🔍 Visual inspection confirms good bounding box localization and classification

---

## 💡 Future Work

* Add real-time webcam detection
* Extend to custom object detection datasets
* Build a Streamlit web app for demo

---

## 📚 References

* [DINO GitHub (IDEA Research)](https://github.com/IDEA-Research/DINO)
* [Hugging Face Datasets](https://huggingface.co/datasets)
* [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

---

## 🤝 Contributing

Pull requests are welcome! If you find bugs or have suggestions, feel free to open an issue.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

