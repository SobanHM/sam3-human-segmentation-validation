# SAM3 vs Human Segmentation Validation
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Segmentation-red)
![Human in the Loop](https://img.shields.io/badge/Human--in--the--Loop-AI%20Validation-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)


This repository presents a **human-in-the-loop segmentation validation system** that quantitatively compares **SAM3 (Segment Anything v3)** automatic segmentation results with **human-annotated ground truth masks**.

The goal of this project is to **measure trust, reliability, and alignment between AI-generated segmentation and human perception**, rather than assuming correctness visually.

---

## 🚀 Key Features

- ✅ Automatic object segmentation using **SAM3**
- ✍️ Manual human annotation (LabelMe-style)
- 📊 Quantitative evaluation using:
  - Intersection over Union (IoU)
  - Precision
  - Recall
  - F1-Score
  - Average confidence score
- 🧠 Human-in-the-loop validation workflow
- 🖥️ Interactive interface for visualization and comparison

---

## 🧪 Methodology

1. An image is provided as input
2. SAM3 generates multiple segmentation masks
3. A human manually annotates the same object
4. The system selects the SAM3 mask with **maximum IoU**
5. Pixel-level comparison is performed between:
   - Human annotation (Ground Truth)
   - SAM3 prediction
6. Quantitative metrics are computed and displayed

Human annotations are treated as **ground truth**, following standard practices in computer vision research.

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate segmentation quality:

- **Intersection over Union (IoU)**
- **Dice Coefficient (F1-score)**
- **Precision**
- **Recall**

These metrics provide an objective measure of how closely the model aligns with human perception.

---

## 🌍 Why This Matters

- Visual inspection alone is insufficient for real-world AI systems
- Large-scale datasets increasingly rely on auto-labeling
- Poor segmentation quality propagates errors downstream
- Trustworthy AI requires **quantitative validation**

This system helps answer:
- How reliable is SAM3 in complex real-world scenes?
- Can SAM3 replace human annotation at scale?
- Where does AI segmentation fail compared to humans?

---

## 🛠️ Tech Stack

- Python
- PyTorch
- SAM3
- NumPy
- Scikit-learn
- Gradio (UI)

---

## 🔮 Applications

- Smart city vision systems
- Autonomous perception
- Traffic and toll-plaza analysis
- Medical imaging validation
- Dataset quality assurance
- Human-in-the-loop AI pipelines

---

## 📌 Future Work

- Support for multi-class segmentation
- Dataset-level benchmarking
- Integration with trajectory-based behavior analysis
- Edge-device optimization
- Active learning loop with expert feedback

---

## 👤 Author

**Soban Hussain**  
AI & Deep Learning/VLMs Researcher  
Email: sobanhussainmahesar@gmail.com

If you find this work useful, feel free to ⭐ the repository.
