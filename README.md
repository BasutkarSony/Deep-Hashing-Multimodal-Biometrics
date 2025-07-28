# Deep Hashing for Secure Multimodal Biometrics

A Python-based deep learning framework for secure multimodal biometric authentication, combining fingerprint and vein images using feature-level fusion via deep hashing and convolutional neural networks. The system preserves user privacy through cancelable biometrics and provides a user-friendly Tkinter GUI for enrollment and authentication.

---

## 🚀 Features

- Multimodal authentication: combines fingerprint and vein images.
- Deep hashing for secure, efficient feature fusion.
- Cancelable biometrics: generates revocable, privacy-preserving templates.
- VGG19-based CNN integrated with a Tkinter GUI.
- Tested on a dataset of 20 users, achieving high classification accuracy.

---

## 🛠️ Tech Stack

- Python 3.7
- VGG19 
- Tkinter (GUI)
- NumPy, Pandas, Matplotlib

---

## 📝 Prerequisites

- Python 3.7.x (required for compatibility)
- Visual Studio Code or any Python IDE
- Git (optional, for version control)
- Internet connection for installing dependencies

---

## ⚡ Installation & Setup

Clone the repository:

<pre>git clone https://github.com/BasutkarSony/Deep-Hashing-Multimodal-Biometrics.git
cd Deep-Hashing-Multimodal-Biometrics</pre>


Create a virtual environment:

```
python -m venv .venv
```


Activate the virtual environment:  
- **Windows CMD:**

    ```
    .venv\Scripts\activate.bat
    ```

- **Windows PowerShell:**

    ```
    .venv\Scripts\Activate.ps1
    ```

- **macOS/Linux:**

    ```
    source .venv/bin/activate
    ```

Upgrade pip and setuptools:

```
python -m pip install --upgrade pip setuptools wheel
```


Install required dependencies:
```
pip install protobuf==3.20.3
pip install -r requirements.txt
```


---

## 🗂 Project Structure
```
deep-hashing-for-secure-multimodal-biometrics/
├── src/
│ ├── MultimodalBiometrics.py
│ └── testtrain.py
├── model/
├── Dataset/
│ ├── User0/
│ ├── User1/
│ ├── User2/
│ └── ...
├── TestImages/
│ ├── sample1/
│ ├── sample2/
│ └── ...
├── docs/
│ ├── DEEP HASHING FOR SECURE MULTIMODAL BIOMETRICS.docx
│ └── Basepaper.pdf
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```


**Folder Descriptions:**  
- `src/`: Source code for application logic and scripts  
- `model/`: Trained models and checkpoints (usually git-ignored)  
- `Dataset/`: User-wise biometric image folders (fingerprint & vein images)  
- `TestImages/`: Sample images for testing and demonstration  
- `docs/`: Documentation, reports, and reference papers  

---

## ▶️ How to Run

After activating your virtual environment, run the main application script:

```
python src/MultimodalBiometrics.py
```

---

## ℹ️ Usage Notes

- Always activate the virtual environment before running the code or installing dependencies.
- Use Python 3.7 for best compatibility.
- Run scripts from the repository root folder to avoid import errors.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Thank you for checking out the project!*  
*If this is useful to you, please consider starring the repository ⭐.*
