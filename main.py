#Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Load the data
data = pd.read_csv('diabetes_data.csv')

#exploring
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())

#Preprocessing
# Remove unnecessary columns
data.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# Convert categorical variables to numerical variables
data['race'] = data['race'].map({'Caucasian': 0, 'AfricanAmerican': 1, 'Hispanic': 2, 'Other': 3, '?': 4})
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1, 'Unknown/Invalid': 2})
data['age'] = data['age'].map({'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
                               '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9})

# Separate features and target
X = data.drop(['readmitted'], axis=1)
y = data['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Model training & Eval
# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict the target variable for test data
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

joblib.dump(model, 'model.pk1')
python app.py