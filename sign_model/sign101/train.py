import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data = data_dict['data']
raw_labels = data_dict['labels']

# Filter only entries where each data sample has exactly 42 elements
filtered_data = []
filtered_labels = []

for i in range(len(raw_data)):
    if len(raw_data[i]) == 42:
        filtered_data.append(raw_data[i])
        filtered_labels.append(raw_labels[i])
    else:
        print(f"Skipping index {i} due to invalid length: {len(raw_data[i])}")

# Convert to numpy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
