# 🌿 Plant Disease Classification using ConvNeXt-Tiny

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)
![Model](https://img.shields.io/badge/Model-ConvNeXt--Tiny-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A deep learning project that classifies plant leaf diseases using a pretrained ConvNeXt-Tiny model on the PlantVillage dataset.  
> It applies data augmentation and class balancing to achieve accurate, efficient, and automated disease detection for 38 plant categories.


## 🚀 Project Overview

Early and accurate detection of plant diseases is crucial for protecting crops and improving yield.  
This project uses **transfer learning** with **ConvNeXt-Tiny** to automatically classify leaf diseases from the **PlantVillage** dataset.

🔹 Balanced training with `WeightedRandomSampler`  
🔹 Advanced augmentations for better generalization  
🔹 Transfer learning from ImageNet weights  
🔹 Visualizations for performance tracking and predictions  

---

## 🧠 Model Architecture

**ConvNeXt-Tiny** (pretrained on ImageNet) was used as a frozen feature extractor.  
Only the classifier head was retrained for the 38 disease categories:

```python
num_classes = 38
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
````

🧩 Optimizer: `Adam`
🎯 Loss Function: `CrossEntropyLoss`
📆 Epochs: 5

---

## 🌱 Dataset

**Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
Contains **color images of healthy and diseased plant leaves** organized into 38 categories.

### Folder Structure

```
PlantVillage/
├── train/
│   ├── Apple___Black_rot/
│   ├── Apple___healthy/
│   └── ...
├── val/
│   ├── Apple___Black_rot/
│   ├── Apple___healthy/
│   └── ...
└── test/
    ├── Apple___Black_rot/
    ├── Apple___healthy/
    └── ...
```

---

## ⚖️ Handling Data Imbalance

The dataset is **imbalanced** across classes.
A `WeightedRandomSampler` was used to balance sample frequencies during training:

```python
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
```

---

## 📊 Evaluation

Visualize training results:

```python
plot_loss_curves(results)
```

Run random test predictions:

```python
python test_random_image.py
```

---

## 🧾 Results Summary

| Metric            | Description                            |
| ----------------- | -------------------------------------- |
| **Architecture**  | ConvNeXt-Tiny (Pretrained on ImageNet) |
| **Optimizer**     | Adam                                   |
| **Loss Function** | CrossEntropyLoss                       |
| **Epochs**        | 5                                      |
| **Classes**       | 38                                     |
| **Model Size**    | ~110 MB                                |

📈 Includes training/validation curves and sample predictions showing **actual vs. predicted** labels.

---

## 🧩 Key Files

| File                                 | Description                         |
| ------------------------------------ | ----------------------------------- |
| `engine_enhanced.py`                 | Training loop implementation        |
| `data_setup.py`                      | Handles dataset loading & splitting |
| `helper_functions.py`                | Utility tools (e.g., seed setting)  |
| `utils.py`                           | Save/load model utilities           |
| `models/ConvNext-Tiny_First_Try.pth` | Trained model checkpoint            |

---

## 💡 Future Work

* Fine-tune full ConvNeXt model
* Experiment with Vision Transformers (ViT)
* Build real-time detection app with Streamlit or Flask

---

## 👨‍💻 Author

**Owais**
📍 University of Jordan – College of Information Technology


---

⭐ **If you found this project useful, consider giving it a star!**

```
