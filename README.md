🧠 EEG Motor Imagery Classification – A Systematic Pipeline Benchmarking Study
Project Title:
A Systematic Investigation of EEG Motor Imagery Classification using the BCI Competition IV-2a Dataset

Overview:
This repository presents a comprehensive and methodical exploration of EEG-based motor imagery classification, focused on identifying the most robust and generalizable machine learning pipeline. Using the BCI Competition IV-2a dataset (4-class MI: left hand, right hand, feet, tongue), we benchmarked classical feature extraction methods and deep learning architectures through a Leave-One-Subject-Out Cross-Validation framework.

🎯 Objective:
To systematically identify the most effective pipeline for decoding motor imagery from EEG using rigorous benchmarking and data-driven experimentation.

🚀 Highlights
✅ Dataset: BCI Competition IV-2a

✅ Validation: Leave-One-Subject-Out (LOSO-CV) for true generalization

✅ Best Accuracy Achieved: 32.08% (Riemannian + SVC)

🔍 Explored Approaches
1. Classical Pipelines
CSP + SVC / RF: Solid baseline (~31%), but limited by assumptions on variance.

Riemannian Geometry + SVC: Top performer, capturing spatial covariance in tangent space.

2. End-to-End Deep Learning
EEGNetv4: Promising, but suffered from overfitting and underfitting trade-offs. Performance capped at ~31.6%.

3. Hyperparameter Tuning
Epoch Duration: Optimal window = 2.5s. Longer windows degraded performance significantly.

4. Advanced Feature Engineering (Ongoing)
Filter-Bank Riemannian + SVC: Combines multiple frequency bands with tangent space mapping. Currently under evaluation.

📊 Key Takeaways
Classical methods, when fine-tuned, can outperform deep learning on small EEG datasets.
Feature representation via Riemannian geometry provides superior discriminative power.
Epoch duration critically influences signal-to-noise ratio.
Deep learning models require extensive regularization and are data-hungry.

📌 Citation & References
https://pubmed.ncbi.nlm.nih.gov/39028609/
https://pubmed.ncbi.nlm.nih.gov/40009879/
https://www.cai.sk/ojs/index.php/cai/article/view/2023_3_741
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865594/full
BCI Competition IV Dataset 2a: https://www.bbci.de/competition/iv/
EEGNet (Lawhern et al., 2018)


Screenshots to be updated soon
