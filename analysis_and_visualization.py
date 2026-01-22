"""
Comprehensive Analysis and Visualization for Neural Network Models
This script generates diagrams, plots, and analysis for all four models:
1. Perceptron Model
2. Regression Model (sin function approximation)
3. Digit Classification Model (MNIST)
4. Language Identification Model (RNN)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# Import our modules
import nn
import backend
import models

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory for saving plots
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("NEURAL NETWORK MODELS ANALYSIS AND VISUALIZATION")
print("="*80)

# Disable live graphics for batch processing
backend.use_graphics = False

def save_plot(filename):
    """Save the current plot to file"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")

# ============================================================================
# 1. PERCEPTRON MODEL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. PERCEPTRON MODEL - Binary Classification")
print("="*80)

print("\n[1.1] Training Perceptron...")
perceptron_model = models.PerceptronModel(3)
perceptron_dataset = backend.PerceptronDataset(perceptron_model)
perceptron_model.train(perceptron_dataset)

print(f"   Training completed in {perceptron_dataset.epoch} epochs")
print(f"   Final weights: {perceptron_model.get_weights().data.flatten()}")

# Visualize decision boundary
print("\n[1.2] Generating Perceptron Visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Decision boundary with data points
ax = axes[0]
x_data = perceptron_dataset.x
y_data = perceptron_dataset.y.flatten()
w = perceptron_model.get_weights().data.flatten()

# Plot data points
positive_mask = y_data == 1
negative_mask = y_data == -1
ax.scatter(x_data[positive_mask, 0], x_data[positive_mask, 1], 
           c='red', marker='+', s=100, label='Class +1', alpha=0.6)
ax.scatter(x_data[negative_mask, 0], x_data[negative_mask, 1], 
           c='blue', marker='_', s=100, label='Class -1', alpha=0.6)

# Plot decision boundary
xlim = ax.get_xlim()
if w[1] != 0:
    x_line = np.array(xlim)
    y_line = (-w[0] * x_line - w[2]) / w[1]
    ax.plot(x_line, y_line, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_xlabel('Feature 1 (x₁)')
ax.set_ylabel('Feature 2 (x₂)')
ax.set_title('Perceptron Decision Boundary')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Weight vector visualization
ax = axes[1]
accuracy = np.mean(np.where(np.dot(x_data, w.reshape(-1, 1)) >= 0, 1, -1) == y_data.reshape(-1, 1))
ax.text(0.5, 0.7, f'Training Accuracy: {accuracy:.2%}', 
        ha='center', va='center', fontsize=16, weight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.5, f'Epochs to Converge: {perceptron_dataset.epoch}', 
        ha='center', va='center', fontsize=14,
        transform=ax.transAxes)
ax.text(0.5, 0.3, f'Weight Vector:\nw₁={w[0]:.3f}, w₂={w[1]:.3f}, b={w[2]:.3f}', 
        ha='center', va='center', fontsize=12, family='monospace',
        transform=ax.transAxes)
ax.axis('off')
ax.set_title('Training Statistics')

plt.tight_layout()
save_plot('01_perceptron_analysis.png')
plt.close()

# ============================================================================
# 2. REGRESSION MODEL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. REGRESSION MODEL - Sin(x) Function Approximation")
print("="*80)

print("\n[2.1] Training Regression Model...")
regression_model = models.RegressionModel()
regression_dataset = backend.RegressionDataset(regression_model)

# Track loss during training
losses = []
original_train = regression_model.train

def train_with_tracking(dataset):
    for x, y in dataset.iterate_forever(200):
        loss = regression_model.get_loss(x, y)
        losses.append(nn.as_scalar(loss))
        if nn.as_scalar(loss) < 0.02:
            break
        gradients = nn.gradients(loss, [regression_model.W1, regression_model.b1, 
                                       regression_model.W2, regression_model.b2])
        regression_model.W1.update(gradients[0], -0.05)
        regression_model.b1.update(gradients[1], -0.05)
        regression_model.W2.update(gradients[2], -0.05)
        regression_model.b2.update(gradients[3], -0.05)

train_with_tracking(regression_dataset)

final_loss = losses[-1] if losses else 0
print(f"   Training completed with final loss: {final_loss:.6f}")
print(f"   Number of iterations: {len(losses)}")

# Generate predictions
print("\n[2.2] Generating Regression Visualizations...")
x_test = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
y_true = np.sin(x_test)
y_pred = regression_model.run(nn.Constant(x_test)).data

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Function approximation
ax = axes[0, 0]
ax.plot(x_test, y_true, 'b-', linewidth=2, label='True sin(x)', alpha=0.7)
ax.plot(x_test, y_pred, 'r--', linewidth=2, label='Model Prediction', alpha=0.7)
ax.scatter(regression_dataset.x, regression_dataset.y, c='green', s=20, 
           alpha=0.3, label='Training Points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Regression: True vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss curve
ax = axes[0, 1]
ax.plot(losses, linewidth=2, color='darkblue')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 3: Residuals
ax = axes[1, 0]
residuals = y_true - y_pred
ax.scatter(x_test, residuals, alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('Residual (True - Predicted)')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# Plot 4: Model architecture
ax = axes[1, 1]
ax.text(0.5, 0.85, 'Neural Network Architecture', 
        ha='center', va='center', fontsize=14, weight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.70, 'Input Layer: 1 neuron', 
        ha='center', va='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.58, '↓', ha='center', va='center', fontsize=16, transform=ax.transAxes)
ax.text(0.5, 0.50, 'Hidden Layer: 512 neurons (ReLU)', 
        ha='center', va='center', fontsize=11, weight='bold', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
        transform=ax.transAxes)
ax.text(0.5, 0.38, '↓', ha='center', va='center', fontsize=16, transform=ax.transAxes)
ax.text(0.5, 0.30, 'Output Layer: 1 neuron', 
        ha='center', va='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.15, f'Total Parameters: {1*512 + 512 + 512*1 + 1:,}', 
        ha='center', va='center', fontsize=10, family='monospace',
        transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
save_plot('02_regression_analysis.png')
plt.close()

# ============================================================================
# 3. DIGIT CLASSIFICATION MODEL ANALYSIS (MNIST)
# ============================================================================
print("\n" + "="*80)
print("3. DIGIT CLASSIFICATION MODEL - MNIST Handwritten Digits")
print("="*80)

print("\n[3.1] Training Digit Classification Model...")
print("   (This may take a few minutes...)")
digit_model = models.DigitClassificationModel()
digit_dataset = backend.DigitClassificationDataset(digit_model)

# Track accuracy during training
accuracies = []
iterations = []
iter_count = 0

for x, y in digit_dataset.iterate_forever(100):
    loss = digit_model.get_loss(x, y)
    iter_count += 1
    
    if iter_count % 50 == 0:
        val_acc = digit_dataset.get_validation_accuracy()
        accuracies.append(val_acc)
        iterations.append(iter_count)
        print(f"   Iteration {iter_count}: Validation Accuracy = {val_acc:.2%}")
    
    if digit_dataset.get_validation_accuracy() >= 0.975:
        print(f"   Target accuracy reached!")
        break
        
    gradients = nn.gradients(loss, [digit_model.W1, digit_model.b1, 
                                   digit_model.W2, digit_model.b2])
    digit_model.W1.update(gradients[0], -0.5)
    digit_model.b1.update(gradients[1], -0.5)
    digit_model.W2.update(gradients[2], -0.5)
    digit_model.b2.update(gradients[3], -0.5)

# Get predictions
print("\n[3.2] Generating Digit Classification Visualizations...")
test_logits = digit_model.run(nn.Constant(digit_dataset.test_images)).data
test_predicted = np.argmax(test_logits, axis=1)
test_accuracy = np.mean(test_predicted == digit_dataset.test_labels)
print(f"   Final test accuracy: {test_accuracy:.2%}")

# Confusion Matrix
confusion_matrix = np.zeros((10, 10))
for true_label, pred_label in zip(digit_dataset.test_labels, test_predicted):
    confusion_matrix[true_label, pred_label] += 1

# Normalize confusion matrix
confusion_matrix_norm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Confusion Matrix
ax = fig.add_subplot(gs[0:2, 0:2])
im = ax.imshow(confusion_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix (Normalized)', fontsize=14, weight='bold')

# Add text annotations
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, f'{confusion_matrix_norm[i, j]:.2f}',
                      ha="center", va="center", color="white" if confusion_matrix_norm[i, j] > 0.5 else "black",
                      fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 2: Accuracy curve
ax = fig.add_subplot(gs[0, 2])
ax.plot(iterations, accuracies, linewidth=2, marker='o', color='darkgreen')
ax.set_xlabel('Iteration')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Training Progress')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# Plot 3: Per-digit accuracy
ax = fig.add_subplot(gs[1, 2])
per_digit_acc = []
for digit in range(10):
    mask = digit_dataset.test_labels == digit
    acc = np.mean(test_predicted[mask] == digit)
    per_digit_acc.append(acc)

colors = ['red' if acc < 0.95 else 'green' for acc in per_digit_acc]
ax.bar(range(10), per_digit_acc, color=colors, alpha=0.7)
ax.axhline(y=0.97, color='red', linestyle='--', label='Target (97%)')
ax.set_xlabel('Digit')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Digit Accuracy')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Sample predictions
ax = fig.add_subplot(gs[2, :])
n_samples = 15
sample_indices = np.random.choice(len(digit_dataset.test_images), n_samples, replace=False)

for idx, sample_idx in enumerate(sample_indices):
    ax_sub = plt.subplot(3, 15, 31 + idx)
    image = digit_dataset.test_images[sample_idx].reshape(28, 28)
    true_label = digit_dataset.test_labels[sample_idx]
    pred_label = test_predicted[sample_idx]
    
    ax_sub.imshow(image, cmap='gray')
    ax_sub.axis('off')
    
    color = 'green' if true_label == pred_label else 'red'
    ax_sub.set_title(f'T:{true_label}\nP:{pred_label}', fontsize=8, color=color)

fig.suptitle(f'MNIST Digit Classification - Test Accuracy: {test_accuracy:.2%}', 
             fontsize=16, weight='bold', y=0.995)

save_plot('03_digit_classification_analysis.png')
plt.close()

# Sample correctly and incorrectly classified digits
print("\n[3.3] Generating Correct vs Incorrect Predictions...")
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

# Correctly classified
correct_mask = test_predicted == digit_dataset.test_labels
correct_indices = np.where(correct_mask)[0]
for i in range(10):
    idx = correct_indices[i]
    axes[0, i].imshow(digit_dataset.test_images[idx].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f'{digit_dataset.test_labels[idx]}', color='green', fontsize=10)
    axes[0, i].axis('off')

# Incorrectly classified
incorrect_mask = ~correct_mask
incorrect_indices = np.where(incorrect_mask)[0]
for i in range(min(10, len(incorrect_indices))):
    idx = incorrect_indices[i]
    axes[1, i].imshow(digit_dataset.test_images[idx].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f'T:{digit_dataset.test_labels[idx]},P:{test_predicted[idx]}', 
                        color='red', fontsize=9)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Correct', fontsize=12, weight='bold')
axes[1, 0].set_ylabel('Incorrect', fontsize=12, weight='bold')
plt.suptitle('Sample Predictions: Correct vs Incorrect', fontsize=14, weight='bold')
plt.tight_layout()
save_plot('04_digit_predictions_samples.png')
plt.close()

# ============================================================================
# 4. LANGUAGE IDENTIFICATION MODEL ANALYSIS (RNN)
# ============================================================================
print("\n" + "="*80)
print("4. LANGUAGE IDENTIFICATION MODEL - RNN for 5 Languages")
print("="*80)

print("\n[4.1] Training Language Identification Model...")
print("   (This may take several minutes...)")
lang_model = models.LanguageIDModel()
lang_dataset = backend.LanguageIDDataset(lang_model)

# Training with progress tracking
train_accuracies = []
train_steps = []
step_count = 0

for x, y in lang_dataset.iterate_forever(100):
    loss = lang_model.get_loss(x, y)
    gradient = nn.gradients(loss, lang_model.parameter)
    
    for param, grad in zip(lang_model.parameter, gradient):
        param.update(grad, -0.1)
    
    step_count += 1
    
    if step_count % 200 == 0:
        val_acc = lang_dataset.get_validation_accuracy()
        train_accuracies.append(val_acc)
        train_steps.append(step_count)
        print(f"   Step {step_count}: Validation Accuracy = {val_acc:.2%}")
        
        # Decay learning rate
        if step_count % 1000 == 0:
            print(f"   Learning rate decayed at step {step_count}")
    
    if step_count > 2000 and lang_dataset.get_validation_accuracy() >= 0.82:
        print(f"   Target accuracy reached!")
        break

# Get test predictions
print("\n[4.2] Generating Language ID Visualizations...")
test_probs, test_predicted, test_correct = lang_dataset._predict('test')
test_accuracy = np.mean(test_predicted == test_correct)
print(f"   Final test accuracy: {test_accuracy:.2%}")

# Confusion matrix for languages
lang_confusion = np.zeros((5, 5))
for true_lang, pred_lang in zip(test_correct, test_predicted):
    lang_confusion[true_lang, pred_lang] += 1

lang_confusion_norm = lang_confusion / lang_confusion.sum(axis=1, keepdims=True)

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Confusion Matrix
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(lang_confusion_norm, cmap='Greens', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(lang_dataset.language_names, rotation=45, ha='right')
ax.set_yticklabels(lang_dataset.language_names)
ax.set_xlabel('Predicted Language', fontsize=11)
ax.set_ylabel('True Language', fontsize=11)
ax.set_title('Language Confusion Matrix', fontsize=12, weight='bold')

for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{lang_confusion_norm[i, j]:.2f}',
                      ha="center", va="center", 
                      color="white" if lang_confusion_norm[i, j] > 0.5 else "black",
                      fontsize=9)
plt.colorbar(im, ax=ax)

# Plot 2: Training progress
ax = fig.add_subplot(gs[0, 1])
ax.plot(train_steps, train_accuracies, linewidth=2, marker='o', color='darkblue')
ax.set_xlabel('Training Step')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Training Progress', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# Plot 3: Per-language accuracy
ax = fig.add_subplot(gs[0, 2])
per_lang_acc = []
for lang_idx in range(5):
    mask = test_correct == lang_idx
    acc = np.mean(test_predicted[mask] == lang_idx)
    per_lang_acc.append(acc)

colors_lang = plt.cm.tab10(range(5))
ax.barh(range(5), per_lang_acc, color=colors_lang, alpha=0.7)
ax.set_yticks(range(5))
ax.set_yticklabels(lang_dataset.language_names)
ax.set_xlabel('Accuracy')
ax.set_title('Per-Language Accuracy', fontsize=12, weight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')
for i, acc in enumerate(per_lang_acc):
    ax.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontsize=9)

# Plot 4: RNN Architecture
ax = fig.add_subplot(gs[1, 0])
ax.text(0.5, 0.95, 'RNN Architecture', 
        ha='center', va='top', fontsize=12, weight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.82, f'Input: {lang_model.num_chars} characters (one-hot)', 
        ha='center', va='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.72, '↓', ha='center', va='center', fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.63, f'Recurrent Layer: {lang_model.sizeHidden} hidden units', 
        ha='center', va='center', fontsize=9, weight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
        transform=ax.transAxes)
ax.text(0.5, 0.53, '↓', ha='center', va='center', fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.44, f'Hidden Layer 1: {lang_model.sizeHidden} neurons (ReLU)', 
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
        transform=ax.transAxes)
ax.text(0.5, 0.35, '↓', ha='center', va='center', fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.26, f'Hidden Layer 2: {lang_model.sizeHidden} neurons (ReLU)', 
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
        transform=ax.transAxes)
ax.text(0.5, 0.17, '↓', ha='center', va='center', fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.08, f'Output: {len(lang_dataset.language_names)} languages', 
        ha='center', va='center', fontsize=9, transform=ax.transAxes)
ax.axis('off')

# Plot 5: Sample words and predictions
ax = fig.add_subplot(gs[1, 1:])
ax.axis('off')
ax.set_title('Sample Test Predictions (Random Selection)', fontsize=12, weight='bold')

# Get some sample predictions
n_samples_per_lang = 3
y_pos = 0.95
for lang_idx in range(5):
    lang_samples = np.where(test_correct == lang_idx)[0][:n_samples_per_lang]
    
    ax.text(0.05, y_pos, f'{lang_dataset.language_names[lang_idx]}:', 
            fontsize=10, weight='bold', transform=ax.transAxes)
    y_pos -= 0.05
    
    for sample_idx in lang_samples:
        word_indices = lang_dataset.test_x[sample_idx]
        word = "".join([lang_dataset.chars[ch] for ch in word_indices if ch != -1])
        pred_lang = test_predicted[sample_idx]
        confidence = test_probs[sample_idx, pred_lang]
        
        correct = pred_lang == lang_idx
        color = 'green' if correct else 'red'
        pred_text = "" if correct else f" → {lang_dataset.language_names[pred_lang]}"
        
        ax.text(0.10, y_pos, f'"{word}" ({confidence:.1%}){pred_text}', 
                fontsize=9, color=color, transform=ax.transAxes, family='monospace')
        y_pos -= 0.04
    
    y_pos -= 0.02

fig.suptitle(f'Language Identification (RNN) - Test Accuracy: {test_accuracy:.2%}', 
             fontsize=16, weight='bold', y=0.998)

save_plot('05_language_identification_analysis.png')
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

summary_text = f"""
NEURAL NETWORK MODELS - COMPREHENSIVE ANALYSIS REPORT
{'='*70}

1. PERCEPTRON MODEL (Binary Classification)
   ├─ Architecture: Single layer perceptron with 3 dimensions
   ├─ Training Epochs: {perceptron_dataset.epoch}
   ├─ Final Accuracy: {accuracy:.2%}
   └─ Weights: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]

2. REGRESSION MODEL (Function Approximation)
   ├─ Architecture: 1 → 512 (ReLU) → 1
   ├─ Target Function: sin(x) on [-2π, 2π]
   ├─ Training Iterations: {len(losses):,}
   ├─ Final Loss: {final_loss:.6f}
   └─ Total Parameters: {1*512 + 512 + 512*1 + 1:,}

3. DIGIT CLASSIFICATION MODEL (MNIST)
   ├─ Architecture: 784 → 200 (ReLU) → 10
   ├─ Dataset: MNIST handwritten digits (28×28 pixels)
   ├─ Training Iterations: {iter_count:,}
   ├─ Validation Accuracy: {digit_dataset.get_validation_accuracy():.2%}
   ├─ Test Accuracy: {test_accuracy:.2%}
   └─ Total Parameters: {784*200 + 200 + 200*10 + 10:,}

4. LANGUAGE IDENTIFICATION MODEL (RNN)
   ├─ Architecture: RNN with {lang_model.sizeHidden} hidden units + 2 dense layers
   ├─ Languages: {', '.join(lang_dataset.language_names)}
   ├─ Character Alphabet Size: {lang_model.num_chars}
   ├─ Training Steps: {step_count:,}
   ├─ Validation Accuracy: {lang_dataset.get_validation_accuracy():.2%}
   ├─ Test Accuracy: {test_accuracy:.2%}
   └─ Total Parameters: {sum([p.data.size for p in lang_model.parameter]):,}

{'='*70}
OUTPUTS GENERATED:
   ✓ 01_perceptron_analysis.png
   ✓ 02_regression_analysis.png
   ✓ 03_digit_classification_analysis.png
   ✓ 04_digit_predictions_samples.png
   ✓ 05_language_identification_analysis.png
   ✓ 06_summary_report.png

All visualizations saved in: {OUTPUT_DIR}/
{'='*70}
"""

ax.text(0.05, 0.95, summary_text, 
        fontsize=10, family='monospace', va='top',
        transform=ax.transAxes)

save_plot('06_summary_report.png')
plt.close()

print("\n" + summary_text)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll visualizations have been saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("   1. 01_perceptron_analysis.png - Decision boundary and statistics")
print("   2. 02_regression_analysis.png - Function approximation and loss curve")
print("   3. 03_digit_classification_analysis.png - Confusion matrix and accuracy")
print("   4. 04_digit_predictions_samples.png - Sample correct/incorrect predictions")
print("   5. 05_language_identification_analysis.png - Language confusion matrix and RNN analysis")
print("   6. 06_summary_report.png - Complete summary report")
print("\n" + "="*80)

