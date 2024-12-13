# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import argparse
import mlflow


# %%

parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()


# Load the dataset
# file_path = "path_to_your_dataset.csv"  # Replace with your actual dataset path
dataset = pd.read_csv(args.trainingdata)

# Preprocess the data
# Separate features and target
X = dataset.drop(columns=["Activity", "subject"])  # Drop target and subject columns
y = dataset["Activity"]

# Encode the target variable (if it's categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=45)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Training a Logistic Regression model
LR_model = LogisticRegression(max_iter=800, random_state=45)
LR_model.fit(X_train, y_train)

# Make predictions on the test set
# y_prediction = rf_classifier.predict(X_test)

y_prediction_lr_model = LR_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
# print(classification_report(y_test, y_prediction, target_names=label_encoder.classes_))
print(classification_report(y_test, y_prediction_lr_model, target_names=label_encoder.classes_))

print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction_lr_model))



