# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
#

# data = pd.read_csv("bank-full.csv", sep=";")
#

# print("Dataset Shape:", data.shape)
# print("\nData Types:\n", data.dtypes)
# print("\nFirst 5 Rows:\n", data.head())
#

# print("\nSummary Statistics:\n", data.describe())
#

# print("\nSubscription (y) Distribution:\n", data["y"].value_counts())
#

# print("\nMissing Values:\n", data.isnull().sum())
#

# print("\nAverage Age by Subscription:")
# print(data.groupby("y")["age"].mean())
# print("\nAverage Balance by Subscription:")
# print(data.groupby("y")["balance"].mean())
# print("\nAverage Duration by Subscription:")
# print(data.groupby("y")["duration"].mean())
#

# print("\nJob Distribution by Subscription:")
# print(pd.crosstab(data["job"], data["y"], normalize="index"))
# print("\nMarital Status Distribution by Subscription:")
# print(pd.crosstab(data["marital"], data["y"], normalize="index"))
#

# plt.figure(figsize=(12, 8))
#
# # Histogram for age
# plt.subplot(2, 2, 1)
# sns.histplot(data=data, x="age", hue="y", multiple="stack")
# plt.title("Age Distribution by Subscription")
# plt.xlabel("Age")
# plt.ylabel("Count")
#
# # Bar chart for job
# plt.subplot(2, 2, 2)
# job_cross = pd.crosstab(data["job"], data["y"], normalize="index")["yes"]
# job_cross.plot(kind="bar")
# plt.title("Subscription Rate by Job")
# plt.xlabel("Job")
# plt.ylabel("Subscription Rate")
# plt.xticks(rotation=45)
#
# # Bar chart for marital status
# plt.subplot(2, 2, 3)
# marital_cross = pd.crosstab(data["marital"], data["y"], normalize="index")["yes"]
# marital_cross.plot(kind="bar")
# plt.title("Subscription Rate by Marital Status")
# plt.xlabel("Marital Status")
# plt.ylabel("Subscription Rate")
#
# # Box plot for duration
# plt.subplot(2, 2, 4)
# sns.boxplot(data=data, x="y", y="duration")
# plt.title("Duration by Subscription")
# plt.xlabel("Subscribed")
# plt.ylabel("Duration (seconds)")
#
# plt.tight_layout()
# plt.savefig("graphs.png")
# plt.close()
#
# # Data Preprocessing
# # Convert target variable 'y' to binary
# data["y"] = data["y"].map({"no": 0, "yes": 1})
#
# # Encode categorical variables
# categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
# data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
#
# # Define features (X) and target (y)
# X = data.drop("y", axis=1)
# y = data["y"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Train a Decision Tree Classifier with limited depth
# clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
# clf.fit(X_train, y_train)
#
# # Make predictions
# y_pred = clf.predict(X_test)
#
# # Evaluate the model
# print("\nModel Evaluation (Depth-Limited Tree):")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# # Feature Importance
# feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
# feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
# print("\nTop 10 Feature Importance:\n", feature_importance.head(10))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("bank-full.csv", sep=";")


print("Dataset Shape:", data.shape)
print("\nData Types:\n", data.dtypes)
print("\nFirst 5 Rows:\n", data.head())


print("\nSummary Statistics:\n", data.describe())


print("\nSubscription (y) Distribution:\n", data["y"].value_counts())


print("\nMissing Values:\n", data.isnull().sum())

# EDA: Numeric features analysis
print("\nAverage Age by Subscription:")
print(data.groupby("y")["age"].mean())
print("\nAverage Balance by Subscription:")
print(data.groupby("y")["balance"].mean())
print("\nAverage Duration by Subscription:")
print(data.groupby("y")["duration"].mean())


print("\nJob Distribution by Subscription:")
print(pd.crosstab(data["job"], data["y"], normalize="index"))
print("\nMarital Status Distribution by Subscription:")
print(pd.crosstab(data["marital"], data["y"], normalize="index"))


plt.figure(figsize=(12, 8))

# Histogram for age
plt.subplot(2, 2, 1)
sns.histplot(data=data, x="age", hue="y", multiple="stack")
plt.title("Age Distribution by Subscription")
plt.xlabel("Age")
plt.ylabel("Count")

# Bar chart for job
plt.subplot(2, 2, 2)
job_cross = pd.crosstab(data["job"], data["y"], normalize="index")["yes"]
job_cross.plot(kind="bar")
plt.title("Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate")
plt.xticks(rotation=45)

# Bar chart for marital status
plt.subplot(2, 2, 3)
marital_cross = pd.crosstab(data["marital"], data["y"], normalize="index")["yes"]
marital_cross.plot(kind="bar")
plt.title("Subscription Rate by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Subscription Rate")

# Box plot for duration
plt.subplot(2, 2, 4)
sns.boxplot(data=data, x="y", y="duration")
plt.title("Duration by Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Duration (seconds)")

plt.tight_layout()
plt.savefig("graphs.png")
plt.close()


# Convert target variable 'y' to binary
data["y"] = data["y"].map({"no": 0, "yes": 1})

# Encode categorical variables
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = data.drop("y", axis=1)
y = data["y"]



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Decision Tree Classifier with limited depth
clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)





# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nModel Evaluation (Depth-Limited Tree):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print("\nTop 10 Feature Importance:\n", feature_importance.head(10))