## License

[Apache-License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---
# Esophageal Cancer Detection Using Tensorflow and EfficientNetV2L
A deep learning project that predicts esophageal cancer risk from endoscopy images using transfer learning with EfficientNetV2L. This repository includes the training pipeline, single-image inference, and a simple web UI for demonstrations.

##  Acknowledgments

This work extends a 2024 student project contributed by  [@harshaygadekar](https://www.github.com/harshaygadekar),
[@kiranss7](https://github.com/kiranss7), [@manoj8541](https://www.github.com/manoj8541),[@atharv6411](https://www.github.com/atharv6411) , Kishan. Their dataset curation, baseline modeling, and early experimentation made this updated release possible. The 2025 revision modernizes the backbone (EfficientNetV2L), adds a user-facing UI, and streamlines training for demos and reproducibility.

##  Features Include

- Binary classification (cancer/no-cancer).
- EfficientNetV2L backbone (pretrained on ImageNet).
- Data augmentation using ImageDataGenerator
- Visualization of training and validation metrics
- Model saving for future use
- Simple web UI for uploading images and getting predictions on cancer.

## Tech Stack Used

- Language: Python.
- Libraries: Tensorflow, Keras APIs, Pandas, Numpy, Matplotlib.
- Model: EfficientNetV2L.
- Environment: Jupyter Notebook.
---
## Installation

### Requirements:

Download the dataset from https://www.kaggle.com/datasets/chopinforest/esophageal-endoscopy-images

### Install Python: 
Download and install Python from https://www.python.org/downloads/

### Install Jupyter Notebook:
Open a terminal or command prompt and run:

```bash
pip install jupyter
```

### Install Required Dependencies:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python
```


### Execution:

- Download the Jupyter Notebook from this repository.
- Specify the directory path and image path
- Specify the number of images to train the model.
- Run each code cells from the beginning.
---
## FAQ

### Q: Why EfficientNetV2L instead of older EfficientNetB7?
A: V2 models are more compatible with modern TensorFlow/Keras and offer strong accuracy with easier setup.

### Q: Why do I get the “PyDataset/fit workers” warning in Keras 3.x?
A: In Keras 3.x, the arguments `workers`, `use_multiprocessing`, and `max_queue_size` passed to `model.fit()` are ignored. You can safely ignore this warning or remove those parameters from `fit()`. If you need parallelism, configure it in the input pipeline (e.g., `tf.data`) rather than via `fit()`.

### Q: What dataset structure is expected?
A:
data/Endoscopy-esophagus/                                                                                                                                                                       
├─ esophagus/ # positive class images                                                                                                                                                   
└─ no-esophagus/ # negative class images

### Q: How do I hide IPs in Gradio logs?
A: Use `demo.launch(quiet=True, show_api=False)` or `demo.launch(quiet=False)` and avoid committing terminal screenshots/logs.

### Q: Can I run this on GPU?
A: 
- NVIDIA: Yes, with standard TensorFlow (ensure proper CUDA/cuDNN versions).
- AMD (Windows): TensorFlow GPU support may require DirectML, which depends on Python/TF versions and isn’t always available. CPU works fine for demos; consider Colab/Kaggle GPUs for training speed.

---
#  🔐 Ethics & Disclaimer:
- This is a research/education project for risk prediction, not a diagnostic device.
- Use alongside medical expertise and validated clinical protocols.
- Be transparent about data sources, model limitations, and biases.
