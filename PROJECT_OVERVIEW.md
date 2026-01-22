# Neural Network Models - Project Overview

This project implements four different neural network models from scratch using NumPy, demonstrating various machine learning concepts from basic perceptrons to recurrent neural networks.

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Models Overview](#models-overview)
- [Installation](#installation)
- [Running the Analysis](#running-the-analysis)
- [Detailed Model Descriptions](#detailed-model-descriptions)

---

## 📁 Project Structure

```
Machine Learning Files/
├── nn.py                           # Neural network framework (nodes, operations, backprop)
├── models.py                       # Four model implementations
├── backend.py                      # Dataset handling and visualization
├── autograder.py                   # Testing framework
├── analysis_and_visualization.py  # Comprehensive analysis script (NEW)
├── requirements.txt               # Python dependencies (NEW)
├── data/
│   ├── mnist.npz                  # MNIST handwritten digits dataset
│   └── lang_id.npz                # Language identification dataset
└── analysis_outputs/              # Generated visualizations (created when running analysis)
    ├── 01_perceptron_analysis.png
    ├── 02_regression_analysis.png
    ├── 03_digit_classification_analysis.png
    ├── 04_digit_predictions_samples.png
    ├── 05_language_identification_analysis.png
    └── 06_summary_report.png
```

---

## 🧠 Models Overview

### 1. **Perceptron Model** - Binary Classification
- **Purpose**: Classify 2D points into two classes (+1 or -1)
- **Architecture**: Single-layer perceptron
- **Key Concepts**: Linear classification, decision boundaries
- **Training**: Perceptron learning rule (update on misclassifications)

### 2. **Regression Model** - Function Approximation
- **Purpose**: Approximate sin(x) function on interval [-2π, 2π]
- **Architecture**: 1 → 512 (ReLU) → 1
- **Key Concepts**: Universal function approximation, non-linear activation
- **Loss Function**: Mean Squared Error (Square Loss)

### 3. **Digit Classification Model** - MNIST
- **Purpose**: Classify handwritten digits (0-9)
- **Architecture**: 784 → 200 (ReLU) → 10
- **Dataset**: 28×28 grayscale images (60,000 training, 10,000 test)
- **Key Concepts**: Multi-class classification, softmax
- **Loss Function**: Softmax Cross-Entropy Loss

### 4. **Language Identification Model** - RNN
- **Purpose**: Identify language from words (English, Spanish, Finnish, Dutch, Polish)
- **Architecture**: RNN (200 hidden) → 200 (ReLU) → 200 (ReLU) → 5
- **Key Concepts**: Recurrent neural networks, sequence processing, hidden state
- **Loss Function**: Softmax Cross-Entropy Loss

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda install numpy matplotlib seaborn
```

### Step 2: Verify Installation

```bash
# Test that numpy is installed
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test that matplotlib is installed
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
```

---

## 🚀 Running the Analysis

### Option 1: Run Complete Analysis (Recommended)

This will train all four models and generate comprehensive visualizations:

```bash
python analysis_and_visualization.py
```

**Expected Runtime:**
- Perceptron: ~1-5 seconds
- Regression: ~10-30 seconds
- Digit Classification: ~2-5 minutes
- Language ID: ~5-10 minutes
- **Total: ~10-15 minutes**

**Output:** 6 PNG files in `analysis_outputs/` directory

### Option 2: Run Individual Models

```bash
# Run the main backend script (trains all models with visualization)
python backend.py

# Or import and test specific models
python -c "
import models
import backend

# Test just the perceptron
model = models.PerceptronModel(3)
dataset = backend.PerceptronDataset(model)
model.train(dataset)
"
```

### Option 3: Run Autograder Tests

```bash
# Test all models
python autograder.py

# Test specific question
python autograder.py -q q1  # Perceptron
python autograder.py -q q2  # Regression
python autograder.py -q q3  # Digit Classification
python autograder.py -q q4  # Language ID

# Run without graphics
python autograder.py --no-graphics
```

---

## 📊 Detailed Model Descriptions

### Model 1: Perceptron (Binary Linear Classifier)

**Mathematical Formulation:**
```
score(x) = w · x
prediction = +1 if score(x) ≥ 0, else -1
```

**Training Algorithm:**
1. Initialize weights randomly
2. For each misclassified point (x, y):
   - Update: w ← w + y·x
3. Repeat until convergence

**Visualizations Generated:**
- Decision boundary plot
- Training data points (colored by class)
- Weight vector and training statistics

---

### Model 2: Regression (Function Approximation)

**Architecture Details:**
```
Input (1D) → Linear(512) → AddBias → ReLU → Linear(1) → AddBias → Output
```

**Mathematical Formulation:**
```
h₁ = ReLU(x·W₁ + b₁)
y = h₁·W₂ + b₂
loss = 0.5·mean((y_pred - y_true)²)
```

**Training:**
- Batch Size: 200
- Learning Rate: 0.05
- Stopping Criterion: Loss < 0.02

**Visualizations Generated:**
- True function vs. predicted function
- Training loss curve (log scale)
- Residual plot (error analysis)
- Network architecture diagram

---

### Model 3: Digit Classification (MNIST)

**Architecture Details:**
```
Input (784D) → Linear(200) → AddBias → ReLU → Linear(10) → AddBias → Output (logits)
```

**Dataset Details:**
- Training: 60,000 images
- Validation: 5,000 images (from test set)
- Test: 5,000 images (from test set)
- Image Size: 28×28 pixels (flattened to 784D vector)

**Training:**
- Batch Size: 100
- Learning Rate: 0.5
- Stopping Criterion: Validation accuracy ≥ 97.5%

**Output Classes:**
- 10 classes (digits 0-9)
- One-hot encoded labels
- Softmax activation for probabilities

**Visualizations Generated:**
- Confusion matrix (10×10)
- Training accuracy curve
- Per-digit accuracy bar chart
- Sample predictions (15 random examples)
- Correct vs. incorrect predictions comparison

---

### Model 4: Language Identification (RNN)

**Architecture Details:**
```
Input (47 chars, one-hot) 
  → RNN Layer (hidden=200)
  → Dense Layer 1 (200, ReLU)
  → Dense Layer 2 (200, ReLU)
  → Output Layer (5 languages)
```

**RNN Formulation:**
```
For each character c in word:
    z = c·Wₓ + h·Wₕ
    h = ReLU(z + b)

h₁ = ReLU(h·W₁ + b₁)
h₂ = ReLU(h₁·W₂ + b₂)
logits = h₂·W_out + b_out
```

**Dataset Details:**
- 5 Languages: English, Spanish, Finnish, Dutch, Polish
- 47 unique characters in combined alphabet
- Variable-length words (batched by length)

**Training:**
- Batch Size: 100
- Initial Learning Rate: 0.1 (decays by 0.9 every 1000 steps)
- Stopping Criterion: Step > 2000 AND validation accuracy ≥ 82%

**Visualizations Generated:**
- Language confusion matrix (5×5)
- Training progress curve
- Per-language accuracy
- RNN architecture diagram
- Sample word predictions with confidence scores

---

## 🔬 Neural Network Framework (`nn.py`)

The project includes a custom neural network framework implemented from scratch:

### Node Types:
- **Parameter**: Trainable weights
- **Constant**: Fixed data (inputs, labels)
- **FunctionNode**: Computed values (operations)

### Operations Implemented:
- `Add`: Element-wise addition
- `AddBias`: Broadcasting bias addition
- `DotProduct`: Batched dot product
- `Linear`: Matrix multiplication (W·x)
- `ReLU`: Rectified Linear Unit (max(0, x))
- `SquareLoss`: Mean squared error
- `SoftmaxLoss`: Cross-entropy loss with softmax

### Backpropagation:
- `gradients(loss, parameters)`: Automatic differentiation
- Computes gradients using reverse-mode autodiff (backprop)
- Returns gradient for each parameter

---

## 📈 Expected Results

### Perceptron
- **Accuracy**: 100% (should converge perfectly on linearly separable data)
- **Epochs**: Typically 5-15 epochs

### Regression
- **Final Loss**: < 0.02
- **Visual Fit**: Should closely match sin(x) curve

### Digit Classification
- **Validation Accuracy**: ≥ 97.5%
- **Test Accuracy**: ~97%
- **Common Errors**: Often confuses 4↔9, 3↔5, 7↔9

### Language Identification
- **Validation Accuracy**: ≥ 82%
- **Test Accuracy**: ~82-85%
- **Challenges**: Short words, shared Latin alphabet

---

## 🎨 Visualization Features

The analysis script generates high-quality visualizations including:

1. **Decision Boundaries** - Visual representation of learned separating hyperplanes
2. **Loss Curves** - Training progress over time
3. **Confusion Matrices** - Error analysis for classification tasks
4. **Sample Predictions** - Visual inspection of model outputs
5. **Architecture Diagrams** - Network structure visualization
6. **Accuracy Metrics** - Per-class and overall performance
7. **Residual Plots** - Regression error analysis

All plots are saved at 300 DPI in PNG format, suitable for reports and presentations.

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError: No module named 'numpy'
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: Training takes too long
**Solution:** 
- Digit classification and language ID can take several minutes
- You can test individual models instead of running all at once
- Use `autograder.py -q q1` to test just the perceptron

### Issue: Graphics not displaying
**Solution:** 
- The analysis script automatically disables live graphics
- All outputs are saved to `analysis_outputs/` directory
- Use `--no-graphics` flag with autograder if needed

### Issue: Low accuracy
**Solution:**
- Ensure models.py is correctly implemented
- Check that gradients are being computed correctly
- Verify learning rates are appropriate

---

## 📚 Learning Objectives

This project demonstrates:

1. ✅ **Neural Network Fundamentals**: Forward/backward propagation
2. ✅ **Optimization**: Gradient descent, learning rate tuning
3. ✅ **Classification**: Binary and multi-class problems
4. ✅ **Regression**: Function approximation
5. ✅ **Recurrent Networks**: Sequence processing with hidden states
6. ✅ **Evaluation**: Accuracy, loss, confusion matrices
7. ✅ **Visualization**: Data analysis and model interpretation

---

## 📄 License

This is an educational project. Feel free to use for learning purposes.

---

## 🤝 Credits

- **Custom NN Framework**: Implements autodiff from scratch
- **MNIST Dataset**: Classic handwritten digit database
- **Language Dataset**: Multi-language word classification

---

## 📞 Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify Python version (3.7+)
3. Check file permissions in data/ directory
4. Review error messages carefully

---

**Happy Learning! 🎓**

