import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# STEP 1: Read the dataset
data = pd.read_csv("dataset.csv")

# STEP 2: Extract features from the URL
data["length"] = data["url"].apply(lambda x: len(x))
data["has_at"] = data["url"].apply(lambda x: 1 if "@" in x else 0)
data["has_dash"] = data["url"].apply(lambda x: 1 if "-" in x else 0)
data["has_https"] = data["url"].apply(lambda x: 1 if "https" in x else 0)
data["dot_count"] = data["url"].apply(lambda x: x.count("."))

# STEP 3: Select input and output
X = data[["length", "has_at", "has_dash", "has_https", "dot_count"]]
y = data["label"]

# STEP 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# STEP 5: Predict user input
def predict_url(url):
    features = [
        len(url),
        1 if "@" in url else 0,
        1 if "-" in url else 0,
        1 if "https" in url else 0,
        url.count(".")
    ]
    result = model.predict([features])
    return "⚠️ Phishing" if result[0] == 1 else "✅ Safe"

# Run: ask user to type a URL
url = input("Enter a website link to check: ")
print(predict_url(url))
