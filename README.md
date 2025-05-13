# CNN Comparison on CIFAR-100 Subset

This project investigates the effectiveness of various convolutional neural network (CNN) architectures—including LeNet, AlexNet, VGG11, and a custom-built `MyCNN`—on a selected 20-class subset of the CIFAR-100 dataset. The study also includes experiments on overfitting control techniques and transfer learning using pretrained models.

---

##  References

- LeCun et al. (1989): *Handwritten Digit Recognition with a Back-Propagation Network*
- Krizhevsky et al. (2012): *ImageNet Classification with Deep Convolutional Neural Networks*
- Simonyan & Zisserman (2014): *Very Deep Convolutional Networks for Large-Scale Image Recognition*
- [Dive into Deep Learning - LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)
- [D2L - AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
- [D2L - VGG](https://d2l.ai/chapter_convolutional-modern/vgg.html)

---

##  Dataset

A fixed 20-class subset of CIFAR-100, filtered using a team seed. Images are resized to 224x224 for compatibility with modern CNNs. Preprocessing includes normalization, class remapping, and TFRecord considerations.

---

##  Models Trained

- **LeNet**: Classic architecture with ~60K parameters.
- **AlexNet**: Deeper model with ReLU and dropout, ~60M parameters.
- **VGG11**: Uses stacked 3x3 filters, ~138M parameters.
- **MyCNN**: Custom lightweight CNN with dropout.

---

##  Evaluation Metric

**Macro F1-score** was used to evaluate classification performance due to class imbalance and the multi-class nature of the problem.

---

##  Results Summary

| Model       | Final Test F1 |
|-------------|---------------|
| LeNet       | 0.4338        |
| AlexNet     | 0.4583        |
| VGG11       | 0.5038        |
| MyCNN       | 0.5157        |
| MyCNN + DA + Dropout 0.5 | **0.5762** |

---

## 🔧 Regularization Experiments

- Dropout rates: `0.3`, `0.5`, `0.7`
- Data Augmentation levels: `light`, `moderate`, `aggressive`
- Best performance: `Moderate Augmentation + Dropout 0.5`

---

##  Transfer Learning Results

| Model Variant                         | Final Test F1 |
|--------------------------------------|----------------|
| VGG19 (head only)                    | 0.7682         |
| VGG19 (last conv layers trainable)   | 0.8440         |
| EfficientNetB0 (head only)           | 0.8770         |
| EfficientNetB0 (last 20 layers trainable) | **0.9136** |

---

##  Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Scikit-learn

---

##  Key Takeaways

- Larger CNNs overfit easily without proper regularization.
- Custom models can outperform classic architectures with the right design.
- Transfer learning (especially with EfficientNetB0) significantly boosts performance.
- Moderate data augmentation + dropout yields robust generalization.

---



