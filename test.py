import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv("C:\\Users\\STUDENT\\Desktop\\New folder (2)\\diabetes_prediction_dataset.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
le = LabelEncoder()
df["smoking_history"] = le.fit_transform(df["smoking_history"])
X = df.drop(["diabetes", "gender"], axis=1)   # gender removed
y = df["diabetes"]

print("\nFinal training features:", X.columns.tolist())
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\nModel training completed.")
print("Model expects features:", model.feature_names_in_)
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Base accuracy
majority_class = df["diabetes"].mode()[0]
base_accuracy = sum(df["diabetes"] == majority_class) / len(df)
print("Base Accuracy:", base_accuracy)
sample_df = pd.read_csv("C:\\Users\\STUDENT\\Desktop\\New folder (2)\\sample.csv")
names = sample_df["name"]

sample_inputs = sample_df.drop("name", axis=1)
sample_inputs["smoking_history"] = le.transform(sample_inputs["smoking_history"])

sample_predictions = model.predict(sample_inputs)

print("\nPredictions for sample.csv:")
for name, pred in zip(names, sample_predictions):
    status = "Diabetes" if pred == 1 else "No Diabetes"
    print(f"{name}: {status}")
total_diabetes = sum(sample_predictions)
total_no_diabetes = len(sample_predictions) - total_diabetes

print("\nSummary:")
print("People predicted with Diabetes:", total_diabetes)
print("People predicted with No Diabetes:", total_no_diabetes)
output_df = pd.DataFrame({
    "name": names,
    "diabetes_prediction": ["Diabetes" if p == 1 else "No Diabetes" for p in sample_predictions]
})
output_df.to_csv("prediction_results.csv", index=False)
print("\nResults exported to prediction_results.csv")
plt.bar(["No Diabetes", "Diabetes"], [total_no_diabetes, total_diabetes])
plt.title("Diabetes Prediction Summary")
plt.xlabel("Category")
plt.ylabel("Number of People")
plt.show()
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel and Label Encoder saved successfully!")