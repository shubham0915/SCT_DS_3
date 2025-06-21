# 🌟 Bank Marketing Analysis Project

![Bank Marketing Banner](https://img.shields.io/badge/Project-Bank%20Marketing%20Analysis-blueviolet?style=for-the-badge)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)  
![Date](https://img.shields.io/badge/Date-June%2021%202025-orange?style=for-the-badge)

## 📖 Overview
This project is part of **Task 3** for my **SkillCraft Technology Internship - Data Science** role. The goal was to analyze the UCI Bank Marketing dataset and build a decision tree classifier to predict whether a client will subscribe to a term deposit (`y`) based on demographic and behavioral data. Through exploratory data analysis (EDA), visualizations, and machine learning, I uncovered actionable insights to guide marketing strategies. 🚀

## 🗃️ Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **File**: `bank-full.csv`
- **Size**: 45,211 instances, 17 attributes
- **Features**:
  - **Input (16)**: `age` (numeric), `job` (categorical), `marital`, `education`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
  - **Output (1)**: `y` (binary: "yes"/"no")
- **Description**: See `bank-names.txt` for full details.

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

### 4. Decision Tree Classifier
- Preprocessed data: Encoded categorical variables using one-hot encoding, converted `y` to binary (0/1).
- Trained a depth-limited decision tree classifier (`max_depth=5`) with balanced class weights to handle imbalance.
- Evaluated the model on a 20% test set.

## 📊 Key Findings
- **Class Imbalance**: 39,922 "no" (88.3%) vs. 5,289 "yes" (11.7%).
- **Numeric Insights**:
  - Clients who subscribed had longer call durations: **537s** ("yes") vs. **221s** ("no") ⏳.
  - Higher average balance for subscribers: **1804** ("yes") vs. **1303** ("no") 💰.
  - Age difference is minor: **41.67** ("yes") vs. **40.84** ("no") 👴.
- **Categorical Insights**:
  - **Job**: Highest subscription rates for `student` (28.68%) and `retired` (22.79%); lowest for `blue-collar` (7.27%) 👨‍🎓.
  - **Marital Status**: `single` (14.95%) and `divorced` (11.95%) more likely to subscribe than `married` (10.12%) 💍.

## 🤖 Model Results
- **Accuracy**: 75.6% (depth-limited tree to prevent overfitting).
- **Classification Metrics**:
  - **Class 0 ("no")**: Precision: 0.98, Recall: 0.74, F1-score: 0.84 (7,985 samples).
  - **Class 1 ("yes")**: Precision: 0.31, Recall: 0.86, F1-score: 0.45 (1,058 samples).
  - Improved recall for "yes" (86%) ensures better identification of potential subscribers, though precision (31%) indicates some false positives.
- **Feature Importance** (Top 5):
  - `duration`: 0.581 – Call duration is the strongest predictor.
  - `poutcome_success`: 0.183 – Previous campaign success is highly influential.
  - `contact_unknown`: 0.119 – Unknown contact method negatively impacts subscriptions.
  - `housing_yes`: 0.077 – Having a housing loan slightly affects the outcome.
  - `month_oct`: 0.023 – Seasonal effect (October campaigns more successful).

## 🛠️ Project Workflow and Decisions
### Step 1: Understanding the Dataset
- **Objective**: Load and explore `bank-full.csv` to understand its structure.
- **Actions**:
  - Used Pandas to load the dataset with `sep=";"` due to its semicolon-separated format.
  - Confirmed the dataset has 45,211 rows and 17 columns, matching the description in `bank-names.txt`.
  - Identified numeric (`age`, `balance`, `duration`, etc.) and categorical (`job`, `marital`, etc.) features.
  - Found no missing values, simplifying preprocessing.

### Step 2: Exploratory Data Analysis (EDA)
- **Objective**: Uncover patterns influencing term deposit subscriptions.
- **Actions**:
  - Calculated summary statistics for numeric features (e.g., mean `age` ~40.9, `balance` range -8019 to 102,127).
  - Analyzed the target variable `y`: 88.3% "no" vs. 11.7% "yes", indicating significant class imbalance.
  - Grouped numeric features by `y` to find trends:
    - `duration` showed a strong correlation (537s for "yes" vs. 221s for "no").
    - `balance` was higher for "yes" (1804 vs. 1303).
    - `age` had a minor difference (41.67 vs. 40.84).
  - Used crosstabs to analyze categorical features:
    - `job`: `student` (28.68%) and `retired` (22.79%) had the highest subscription rates.
    - `marital`: `single` (14.95%) and `divorced` (11.95%) were more likely to subscribe than `married` (10.12%).

### Step 3: Visualizations
- **Objective**: Visualize insights to support the analysis.
- **Actions**:
  - Used Matplotlib and Seaborn for plotting.
  - Created four plots in a 2x2 grid:
    - Histogram of `age` by subscription to explore age distribution.
    - Bar charts for subscription rates by `job` and `marital` status to highlight demographic trends.
    - Box plot of `duration` by `y` to confirm its strong predictive power.
  - Saved the plots as `graphs.png` for inclusion in the repository.

### Step 4: Initial Decision Tree Classifier
- **Objective**: Build a model to predict `y` based on demographic and behavioral data.
- **Actions**:
  - Preprocessed the data:
    - Converted `y` to binary (0 for "no", 1 for "yes").
    - Used one-hot encoding for categorical variables (`job`, `marital`, etc.) to prepare them for modeling.
  - Split the data into 80% training and 20% test sets, using `stratify=y` to maintain the class imbalance ratio.
  - Trained a decision tree classifier with `class_weight="balanced"` to address the imbalance.
  - Evaluated the model:
    - Accuracy: 87.8%.
    - "yes" class: Precision: 0.48, Recall: 0.45, F1-score: 0.46.
  - **Observation**: The model performed well on the majority class ("no") but struggled with the minority class ("yes"), with low recall (45%), meaning it missed many actual subscribers.

### Step 5: Depth-Limited Decision Tree Classifier
- **Objective**: Improve the model’s performance on the minority class and prevent overfitting.
- **Actions**:
  - Limited the tree depth to `max_depth=5` to reduce overfitting, as the initial tree was likely too complex and fitting noise in the training data.
  - **Why Depth-Limited?**:
    - Decision trees can grow very deep, especially with imbalanced datasets, leading to overfitting where the model memorizes the training data rather than generalizing.
    - A depth of 5 was chosen as a starting point to balance model complexity and performance, ensuring the tree captures key patterns (e.g., `duration`) without overfitting to less important features.
  - Re-evaluated the model:
    - Accuracy: 75.6% (lower due to reduced complexity, but more reliable).
    - "yes" class: Precision: 0.31, Recall: 0.86, F1-score: 0.45.
  - **Improvement**: Recall for "yes" improved significantly (45% to 86%), meaning the model now identifies most actual subscribers, which is crucial for marketing applications. However, precision dropped (48% to 31%), indicating more false positives, a trade-off we accepted to prioritize recall.
  - Analyzed feature importance:
    - `duration` (0.581) was the top predictor, confirming our EDA findings.
    - `poutcome_success` (0.183) and `contact_unknown` (0.119) were also influential, suggesting that previous campaign outcomes and contact method play a role.

### Key Decisions and Rationale
- **Class Weighting**: Used `class_weight="balanced"` to give more importance to the minority class ("yes") during training, addressing the 88.3%/11.7% imbalance.
- **One-Hot Encoding**: Chose this method for categorical variables to ensure the model can interpret them without assuming ordinal relationships.
- **Stratified Split**: Ensured the train/test split maintained the class imbalance ratio, preventing skewed evaluation metrics.
- **Focus on Recall for "yes"**: Prioritized higher recall for the minority class to identify more potential subscribers, even at the cost of lower precision, as false positives are less costly than missing actual subscribers in a marketing context.

## 🛠️ Tools & Techniques Used
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python)  
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458?style=flat-square&logo=pandas)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-FF6F61?style=flat-square&logo=python)  
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-FF9F00?style=flat-square&logo=python)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-F7931E?style=flat-square&logo=scikit-learn)  
![EDA](https://img.shields.io/badge/Technique-EDA-00CED1?style=flat-square)  
![Data Visualization](https://img.shields.io/badge/Technique-Data%20Visualization-4682B4?style=flat-square)  
![Decision Tree](https://img.shields.io/badge/Technique-Decision%20Tree-9B59B6?style=flat-square)

## 📂 Repository Structure
- `bank-full.csv`: The dataset used for analysis.
- `bank_analysis.py`: Python script for EDA, visualizations, and modeling.
- `graphs.png`: Output of the visualizations.
- `bank-names.txt`: Dataset description.

## 🚀 How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Bank-Marketing-Analysis.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. **Run the Script**:
   ```bash
   python bank_analysis.py
   ```
4. **View Results**:
   - Check the console for EDA and model evaluation output.
   - Open `graphs.png` for visualizations.

## 💡 Future Improvements
- Apply advanced techniques like Random Forest or XGBoost for better performance.
- Use SMOTE to further address class imbalance.
- Explore hyperparameter tuning for the decision tree (e.g., adjusting `max_depth`, `min_samples_split`).

## 👩‍💻 Author
**Arous**  
SkillCraft Technology Intern - Data Science  
📅 Completed: June 21, 2025  
📧 Contact: [Your Email/LinkedIn]  

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.