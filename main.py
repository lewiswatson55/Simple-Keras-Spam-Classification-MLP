import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.init(project="Email-Spam-Classification-MLP-For-ENUSEC-Talk")

# Load our csv into a dataframe with column names from header
df = pd.read_csv('emails.csv', header=0)

# Remove Email No. column from df
df.pop('Email No.')

# extract the last column as the labels
labels = df.pop('Prediction') # -  1 for spam, 0 for not spam
labels = pd.get_dummies(labels) # Not really required but useful if changing from binary to multi classification

print(df.head())
print(labels.head())

# Convert to numpy arrays
X = df.values
y = labels.values


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# print sizes
print("Train Size: ", len(X_train))
print("Test Size: ",len(X_test))
print("Validation size: ", len(X_val))


# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(3000, activation='relu', input_shape=(3000,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model!
model.fit(X_train, y_train,
          batch_size=25,
          epochs=50,
          validation_data=(X_val, y_val),
          callbacks=[WandbMetricsLogger()])


# Evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

