import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the reshaped data from the pickle file
with open("./reshaped_data.pickle", "rb") as f:
    data_dict = pickle.load(f)

# Check if the loaded data_dict is a dictionary and contains the expected keys
if isinstance(data_dict, dict) and "data" in data_dict and "labels" in data_dict:
    data = data_dict["data"]
    labels = data_dict["labels"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Create and train the Random Forest model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_predict = model.predict(x_test)

    # Calculate accuracy
    score = accuracy_score(y_test, y_predict)

    print("{}% of samples were classified correctly!".format(score * 100))

    # Save the trained model to a file
    with open("model.p", "wb") as f:
        pickle.dump({"model": model}, f)
else:
    print(
        "The loaded pickle file does not contain 'data' and 'labels' keys, or it is not a dictionary."
    )
