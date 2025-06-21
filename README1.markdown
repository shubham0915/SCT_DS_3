# 🌟 Bank Marketing Analysis Project

![Bank Marketing Banner](https://img.shields.io/badge/Project-Bank%20Marketing%20Analysis-blueviolet?style=for-the-badge)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)  
![Date](https://img.shields.io/badge/Date-June%2021%202025-orange?style=for-the-badge)

## 📖 Overview
This project is part of **Task 3** for my **SkillCraft Technology Internship - Data Science** role. The goal was to analyze the UCI Bank Marketing dataset to identify key factors influencing whether a client subscribes to a term deposit (`y`). Through exploratory data analysis (EDA) and visualizations, I uncovered actionable insights to guide marketing strategies. 🚀

---

## 🗃️ Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **File**: `bank-full.csv`
- **Size**: 45,211 instances, 17 attributes
- **Features**:
  - **Input (16)**: `age` (numeric), `job` (categorical), `marital`, `education`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
  - **Output (1)**: `y` (binary: "yes"/"no")
- **Description**: See `bank-names.txt` for full details.

---

## 🔍 Methodology
### 1. Data Loading & Initial Exploration
- Loaded the dataset using **Pandas** with semicolon separation (`sep=";"`).
- Checked shape, data types, and missing values (none found).

### 2. Exploratory Data Analysis (EDA)
- Analyzed numeric features (`age`, `balance`, `duration`) by subscription status.
- Examined categorical features (`job`, `marital`) using crosstabs with normalization.
- Identified class imbalance: 88.3% "no" vs. 11.7% "yes".

### 3. Visualizations
- Created four plots to visualize insights:
  - Histogram: Age distribution by subscription.
  - Bar charts: Subscription rates by `job` and `marital` status.
  - Box plot: Call `duration` by subscription status.
- Saved as `graphs.png`.

---

## 📊 Key Findings
- **Class Imbalance**: 39,922 "no" (88.3%) vs. 5,289 "yes" (11.7%).
- **Numeric Insights**:
  - Clients who subscribed had longer call durations: **537s** ("yes") vs. **221s** ("no") ⏳.
  - Higher average balance for subscribers: **1804** ("yes") vs. **1303** ("no") 💰.
  - Age difference is minor: **41.67** ("yes") vs. **40.84** ("no") 👴.
- **Categorical Insights**:
  - **Job**: Highest subscription rates for `student` (28.68%) and `retired` (22.79%); lowest for `blue-collar` (7.27%) 👨‍🎓.
  - **Marital Status**: `single` (14.95%) and `divorced` (11.95%) more likely to subscribe than `married` (10.12%) 💍.

---

## 🛠️ Tools & Techniques Used
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python)  
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458?style=flat-square&logo=pandas)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-FF6F61?style=flat-square&logo=python)  
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-FF9F00?style=flat-square&logo=python)  
![EDA](https://img.shields.io/badge/Technique-EDA-00CED1?style=flat-square)  
![Data Visualization](https://img.shields.io/badge/Technique-Data%20Visualization-4682B4?style=flat-square)

---

## 📂 Repository Structure
- `bank-full.csv`: The dataset used for analysis.
- `bank_analysis.py`: Python script for EDA and visualizations.
- `graphs.png`: Output of the visualizations.
- `bank-names.txt`: Dataset description.

---

## 🚀 How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Bank-Marketing-Analysis.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install pandas matplotlib seaborn
   ```
3. **Run the Script**:
   ```bash
   python bank_analysis.py
   ```
4. **View Results**:
   - Check the console for EDA output.
   - Open `graphs.png` for visualizations.

---

## 💡 Future Improvements
- Apply machine learning models (e.g., logistic regression, random forest) to predict subscriptions.
- Handle class imbalance using techniques like SMOTE.
- Explore more features (e.g., `poutcome`, `contact`) for deeper insights.

---

## 👩‍💻 Author
**Arous**  
SkillCraft Technology Intern - Data Science  
📅 Completed: June 21, 2025  
📧 Contact: [Your Email/LinkedIn]  

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.