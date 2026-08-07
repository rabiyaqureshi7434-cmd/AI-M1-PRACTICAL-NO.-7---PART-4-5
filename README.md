4. Implement k-fold cross-validation on a classification model.
Code
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Create classification model
model = KNeighborsClassifier()
# Apply 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy scores for each fold:")
print(scores)
print("Average Accuracy:", round(scores.mean() * 100, 2), "%")
Output

NetFlix dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv("C:/Users/info/Downloads/NetFlix - NetFlix.csv ")
# Select required columns
df = df[['type', 'release_year', 'rating']]
# Remove missing values
df = df.dropna()
# Convert text to numbers
le_type = LabelEncoder()
le_rating = LabelEncoder()
df['type'] = le_type.fit_transform(df['type'])
df['rating'] = le_rating.fit_transform(df['rating'])
# Features and Target
X = df[['release_year', 'rating']]
y = df['type']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Create model
model = DecisionTreeClassifier(random_state=42)
# Train model
model.fit(X_train, y_train)
# Accuracy
print("Training Accuracy:", model.score(X_train, y_train) * 100)
print("Testing Accuracy:", model.score(X_test, y_test) * 100)
# Prediction
prediction = model.predict([[2020, 3]])
print("Predicted Type:", le_type.inverse_transform(prediction))
5. Implement a Decision Tree classifier using the Iris or any relevant dataset.
output

5. Implement a Decision Tree classifier using the Iris or any relevant dataset.
Code
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)
# Create model
model = DecisionTreeClassifier()
# Train model
model.fit(X_train, y_train)
# Display accuracy
print("Training Accuracy:", model.score(X_train, y_train) * 100)
print("Testing Accuracy:", model.score(X_test, y_test) * 100)
# Prediction
prediction = model.predict([X_test[0]])
print("Predicted Class:", prediction[0])
Output

NetFlix dataset
Code
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv("C:/Users/info/Downloads/NetFlix - NetFlix.csv ")
# Select required columns
df = df[['type', 'release_year', 'rating']]
# Remove missing values
df = df.dropna()
# Encode categorical columns
le_type = LabelEncoder()
le_rating = LabelEncoder()
df['type'] = le_type.fit_transform(df['type'])
df['rating'] = le_rating.fit_transform(df['rating'])
# Features and Target
X = df[['release_year', 'rating']]
y = df['type']
# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Apply 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy Scores:")
print(scores)
print("Average Accuracy:", round(scores.mean() * 100, 2), "%")
Output


