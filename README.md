# Deep Hashing for Secure Multimodal Biometrics

---

## Project Overview

This project presents a **Python-based deep learning framework for secure multimodal biometric authentication**, combining **fingerprint and vein images** through feature-level fusion using deep hashing and convolutional neural networks (CNNs). It incorporates **cancelable biometrics** to generate revocable and non-invertible biometric templates, preserving user privacy.

The core of the system is a **VGG19-based CNN model** trained to learn robust deep hash codes from biometric data. A custom **Tkinter GUI** provides an interface for enrollment and authentication workflows.

Tested on a dataset of 20 users, the system achieves **high classification accuracy** while maintaining strong privacy protections.

---

## Features

- Multimodal biometric authentication using fingerprint and vein images.
- Deep hashing for secure and efficient feature-level fusion.
- Cancelable biometrics for privacy and template protection.
- VGG19-based CNN training integrated with a user-friendly GUI.
- Python 3.7 environment with all dependencies managed via `requirements.txt`.

---

## Prerequisites

- **Python 3.7.x** (required; later versions may cause compatibility issues)
- [Visual Studio Code](https://code.visualstudio.com/) or another Python IDE
- Windows Command Prompt or PowerShell (macOS/Linux terminal also supported)
- [Git](https://git-scm.com/), optional but recommended for version control
- Internet connection for dependency installation

---

## Setup & Installation

1. **Clone the repository**

git clone https://github.com/BasutkarSony/Deep-Hashing-Multimodal-Biometrics.git
cd Deep-Hashing-Multimodal-Biometrics

text

2. **Create a virtual environment**

python -m venv .venv

text

3. **Activate the virtual environment**

- Windows CMD:

  ```
  .venv\Scripts\activate.bat
  ```

- Windows PowerShell:

  ```
  .venv\Scripts\Activate.ps1
  ```

- macOS/Linux:

  ```
  source .venv/bin/activate
  ```

4. **Upgrade pip and setuptools**

python -m pip install --upgrade pip setuptools wheel

text

5. **Install dependencies**

pip install protobuf==3.20.3
pip install -r requirements.txt

text

---

## Project Structure

deep-hashing-for-secure-multimodal-biometrics/
├── src/ # Python source code
│ ├── MultimodalBiometrics.py # Main application script
│ └── testtrain.py # Additional scripts
├── model/ # Model files and checkpoints (ignored in git)
Dataset/ # Dataset Folder
├── User0/
├── User1/
├── User2/
├── ...           # Multiple user folders, each containing that user's fingerprint and vein images
TestImages/ # Test Images Folder
├── sample1/
├── sample2/
├── ...           # Multiple test sample folders for demo/testing purposes
├── docs/ # Documentation and screenshots
│ ├── DEEP HASHING FOR SECURE MULTIMODAL BIOMETRICS.docx
│ └── Basepaper.pdf
├── requirements.txt # Python dependencies with pinned versions
├── .gitignore # Files/folders ignored by git
├── README.md # This file
└── LICENSE # Project license

text

---

## Running the Project

After activating your virtual environment, run:

python src\MultimodalBiometrics.py

---
## Usage Notes

- Use Python 3.7 for best compatibility.
- Always activate your virtual environment before running or installing.
- Run scripts from the project root to avoid import errors.

---

## License

This project is distributed under the [MIT License](LICENSE).

---

*Thank you for checking out the project! If this is helpful, please consider ⭐ starring the repository.*