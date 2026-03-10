\# IoMT Cybersecurity ML: Attack Detection and Classification with CICIoMT2024



\## Overview



This project presents an end-to-end machine learning workflow for cybersecurity analysis in the Internet of Medical Things (IoMT) using the CICIoMT2024 dataset. The project covers the full pipeline from raw CSV integration and exploratory data analysis to preprocessing, imbalanced classification, explainability, and deployment through a Streamlit application.



The main goal is to detect and classify malicious IoMT network traffic using flow-based features extracted from benchmark traffic data. Three supervised learning tasks were developed:



1\. Binary intrusion detection: Benign vs Attack  

2\. Grouped multiclass classification: Benign, DDoS, DoS, MQTT Attack, Recon, Spoofing  

3\. Full multiclass classification: detailed attack-type recognition  



The best-performing model family across all tasks was Random Forest.



---



\## Dataset



Dataset used: CICIoMT2024  

Source: Kaggle version of the CIC IoMT benchmark dataset



The dataset contains network-flow features and labeled traffic instances representing benign IoMT traffic and multiple cyberattack scenarios. The raw data were organized into separate CSV files by traffic or attack type and then merged into a unified modeling dataset.



Example classes included:

\- Benign Traffic

\- DDoS ICMP Flood

\- DDoS UDP Flood

\- DoS TCP Flood

\- DoS UDP Flood

\- DoS ICMP Flood

\- MITM ARP Spoofing

\- MQTT DDoS Publish Flood

\- MQTT DoS Connect Flood

\- MQTT DoS Publish Flood

\- MQTT Malformed

\- Recon OS Scan

\- Recon Ping Sweep

\- Recon Port Scan

\- Recon Vulnerability Scan



Important dataset characteristics:

\- No missing values

\- No infinite values

\- Severe class imbalance

\- Strong feature redundancy and multicollinearity in several flow-based variables



Note: Raw dataset files are not included in this repository. Please download them directly from Kaggle and place them in `data/raw/`.



---



\## Project Objectives



The project was designed around the following objectives:



\- Build a reproducible cybersecurity ML pipeline for IoMT traffic

\- Detect malicious traffic in a binary intrusion-detection setting

\- Classify broader attack families in a grouped multiclass setting

\- Classify fine-grained attack subtypes in a full multiclass setting

\- Compare linear and tree-based baselines

\- Evaluate performance under severe class imbalance

\- Interpret the strongest models using feature importance, permutation importance, and SHAP

\- Deploy the best models in an interactive Streamlit application



---



\## Repository Structure



```text

iomt-cybersecurity-ml/

├── app/

│   ├── streamlit\_app.py

│   ├── utils.py

│   ├── binary\_input\_template.csv

│   ├── grouped\_input\_template.csv

│   ├── full\_input\_template.csv

│   ├── binary\_input\_example.csv

│   ├── grouped\_input\_example.csv

│   ├── full\_input\_example.csv

│   └── sample\_input\_template.csv

│

├── data/

│   ├── raw/

│   ├── interim/

│   └── processed/

│

├── models/

│   ├── binary\_random\_forest.joblib

│   ├── grouped\_random\_forest.joblib

│   └── full\_random\_forest.joblib

│

├── notebooks/

│   ├── 01\_data\_loading\_and\_inspection.ipynb

│   ├── 02\_eda.ipynb

│   ├── 03\_preprocessing.ipynb

│   ├── 04\_binary\_classification.ipynb

│   ├── 05\_grouped\_multiclass\_classification.ipynb

│   ├── 06\_full\_multiclass\_classification.ipynb

│   ├── 07\_explainability\_and\_final\_model\_selection.ipynb

│   └── 08\_create\_app\_sample\_files.ipynb

│

├── reports/

│   ├── figures/

│   └── results/

│

├── src/

├── README.md

├── requirements.txt

├── .gitignore

└── LICENSE

