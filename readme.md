# 🧠 Spiking Neural Network (SNN) on MNIST

This project implements a **fully connected Spiking Neural Network (SNN)** using [snntorch](https://snntorch.readthedocs.io) to classify handwritten digits from the **MNIST dataset**.  

Unlike traditional ANNs and CNNs, SNNs use **spikes (binary events over time)** instead of continuous activations, making them more biologically inspired and energy-efficient.

---

## 🚀 Project Structure
```
spiking-mnist-snn/
│── spiking_mnist.py      # Main training script
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

---

## ⚙️ Installation

Clone this repo:
```bash
git clone https://github.com/your-username/spiking-mnist-snn.git
cd spiking-mnist-snn
```

Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Training the Model

Run the script:
```bash
python spiking_mnist.py
```

This will:
- Load the MNIST dataset  
- Define a fully connected Spiking Neural Network  
- Train it using spike-based backpropagation  
- Print training loss and accuracy  

---

## 📈 Results
- The model achieves **~90% test accuracy** on MNIST (depending on training settings).  
- Spiking activity can be visualized over time for better interpretability.  

---

## 🛠 Requirements
- Python 3.8+  
- snntorch  
- torch  
- torchvision  
- matplotlib  

Install them with:
```bash
pip install torch torchvision snntorch matplotlib
```

---

## ✨ Future Work
- Add Convolutional SNN for better accuracy  
- Visualize spike rasters  
- Compare SNN vs ANN performance  

---

## 📌 Reference
- [snntorch Documentation](https://snntorch.readthedocs.io)  
- *Eshraghian, J.K. et al. "Training Spiking Neural Networks Using Lessons from Deep Learning."*  

---

👨‍💻 Author: [ADITYA RAJ](https://github.com/muddycode-tech)  
⭐ If you like this project, don’t forget to star the repo!
