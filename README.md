Here’s a **README.md** tailored for your Plant Disease Detection project with Streamlit and DVC:

---

```markdown
# 🌿 Plant Disease Detection

A deep learning–powered application to detect plant leaf diseases using **CNN** and a simple **Streamlit web interface**.  
The project uses **DVC** for dataset and model versioning, ensuring reproducibility and efficient storage.

---

## 📌 Features
- Detects plant leaf diseases from uploaded images.
- Interactive Streamlit web app.
- Real-time prediction confidence levels.
- Feedback system to collect user opinions.
- Dataset and trained model tracked with **DVC**.

---

## 📂 Project Structure
```

Plant\_Disease\_Detection/
│
├── app.py                             # Streamlit web app
├── plant\_diseases.ipynb               # Jupyter notebook for training
├── plant\_diseases\_cnn.h5.dvc          # DVC tracked model file
├── plant\_dieases\_recognation\_dataset.dvc # DVC tracked dataset
├── requirements.txt                   # Project dependencies
├── feedback.csv                       # User feedback (ignored in git)
├── README.md                          # Project documentation
└── .gitignore                         # Ignored files and folders

````

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/HarekrishnaHK/Plant_Disease_Detection.git
cd Plant_Disease_Detection
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Pull Dataset & Model from DVC Remote

```bash
dvc pull
```

---

## 🚀 Usage

### Run Streamlit App

```bash
streamlit run app.py
```

### Training the Model

Open `plant_diseases.ipynb` in Jupyter Notebook or Jupyter Lab and run all cells.

---

## 📊 Dataset

* **Source:** [Kaggle Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
* Organized into **Train**, **Validation**, and **Test** folders.
* Each image is resized to **128x128** during preprocessing.

---

## 🧠 Model Architecture

* **3 Convolutional Layers** with ReLU activation
* **MaxPooling** after each convolution
* **Dense layer** with Dropout for regularization
* **Softmax output layer** for multi-class classification

---

## 📈 Training Details

* **Optimizer:** Adam
* **Loss:** Categorical Crossentropy
* **Epochs:** 20
* **Augmentation:** Rotation, shift, zoom, horizontal flip

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify it.

---

## 🙌 Acknowledgements

* [Kaggle](https://www.kaggle.com) for providing the dataset
* [TensorFlow](https://www.tensorflow.org/) for deep learning framework
* [Streamlit](https://streamlit.io/) for the interactive UI
* [DVC](https://dvc.org/) for dataset and model versioning


## 📸 App Preview

Here’s a preview of the Plant Disease Detection app in action:

![App Screenshot](app_preview.png)

```
