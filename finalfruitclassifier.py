# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part 1: Customize the Dataset
# Features: [weight (grams), shape (0=round, 1=long, 2=heart-shaped), color (0=green, 1=yellow, 2=red, 3=orange)]
# Labels: 0=apple, 1=banana, 2=cherry, 3=lemon, 4=avocado, 5=grape, 6=guava

x = np.array([
    [150, 0, 2],  # apple
    [120, 1, 1],  # banana
    [10, 2, 2],   # cherry
    [130, 0, 2],  # apple
    [110, 1, 1],  # banana
    [5, 2, 2],    # cherry
    [200, 0, 1],  # lemon
    [70, 0, 0],   # guava
    [180, 0, 1],  # lemon
    [170, 1, 0],  # avocado
    [6, 0, 2],    # grape
    [90, 0, 0],   # guava
    [8, 0, 2],    # grape
    [160, 1, 0]   # avocado
])

y = np.array([0, 1, 2, 0, 1, 2, 3, 6, 3, 4, 5, 6, 5, 4])  # Fruit labels

# Step 2: Create and Train the Model
model = DecisionTreeClassifier()
model.fit(x, y)
print("Model Training Completed Succesfully.")

# Step 3: Make Predictions
test_fruits = np.array([
    [160, 0, 2],  # 160 grams, round, red (likely apple)
    [180, 0, 1],  # 115 grams, round, yellow (likely lemon)
    [100, 1, 1]   # 100 grams, long, yellow (likely banana)
])
labels_predicted = model.predict(test_fruits)

fruit_names = {0: "Apple", 1: "Banana", 2: "Cherry", 3: "Lemon", 4: "Avocado", 5: "Grape", 6: "Guava"}
for i, fruits in enumerate(labels_predicted):
    print(f"testFRUIT {i+1} is Predicted To Be: {fruit_names[fruits]}")

# Step 4: Accuracy Score
# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
splitmodel = DecisionTreeClassifier()
splitmodel.fit(x_train, y_train)
y_predict = splitmodel.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
