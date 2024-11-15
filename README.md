import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
columns = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
data = pd.read_csv(url, sep="\s+", names=columns)

# Step 2: Preprocess the data
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values - 1  # Labels (adjusting to 0-based indexing)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the labels
y = to_categorical(y, num_classes=3)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the neural network
model = Sequential([
    Dense(10, input_dim=X.shape[1], activation='relu'),  # Hidden layer with 10 neurons
    Dense(8, activation='relu'),                        # Another hidden layer with 8 neurons
    Dense(3, activation='softmax')                      # Output layer with 3 neurons (one per class)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 6: Predict on new data (optional)
predictions = np.argmax(model.predict(X_test), axis=-1)
true_classes = np.argmax(y_test, axis=-1)

# Print some results
print(f"Predicted classes: {predictions[:5]}")
print(f"True classes: {true_classes[:5]}")
