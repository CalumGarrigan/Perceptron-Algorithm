# Name: Calum Garrigan, Student Number: 201379070

# Binary perceptron implementation
import numpy as np

# Establishing a class for binary perceptron training and testing
class Perceptron:
    def __init__(self, max_epoch, class_a, class_b, train_set):
        self.max_epoch = max_epoch
        self.train_set = train_set

        self.class_a = class_a
        self.class_b = class_b

        self.num_instances, self.num_features = self.train_set.shape
        self.weights = np.zeros(self.num_features - 1, np.float16)
        self.b = 0

    def train(self, l2_reg):
        # Labels and features should be split
        y_true, features = self.split_labels_features()
        y_true = [1 if label == self.class_a else -1 for label in y_true]
        
        # A necessary step before the regularised update
        a = float(1 - (2 * l2_reg))

        for epoch in range(self.max_epoch):
            for sw in range(self.num_instances):
                activation = np.dot(self.weights, features[sw]) + self.b
                # Misclassification
                if (y_true[sw] * activation) <= 0:
                    self.update_weights(features[sw], y_true[sw], a)

    def split_labels_features(self):
        class_label = self.train_set[:, 4]
        features = self.train_set[:, :4]
        return class_label, features

    def update_weights(self, feature, y_true, a):
        self.weights = (self.weights * a) + feature * y_true
        self.b += y_true


    def test(self, test_set, rest_flag):
        # For 1vs1 appraoch
        if not rest_flag:
            # Extracting just the features
            features = test_set[:, :4]

            activation = np.dot(features, self.weights) + self.b

            # Obtain the predicted labels
            y_pred = np.sign(activation)
            return y_pred
        # For 1vsrest approach
        else:
            features = test_set
            activation = np.dot(self.weights, features) + self.b
            return activation

    def accuracy(self, dataset, y_pred):
        # The class label being extracted
        classlabel = dataset[:, 4]

        y_true = np.array([1 if label == self.class_a else -1 for label in classlabel])

        # Number of data samples that are correctly classified
        positive = np.sum(y_pred == y_true)

        # Calculation of accuracy
        accuracy = (positive / len(y_true)) * 100

        return accuracy


# Assistance Functions
def masking1v1(dataset, class_combinations):
    # Extracting the class label
    mask = dataset[:, 4]

    class_datasets = []

    # Separates the dataset for each of the class combinations for 1vs1 approach
    for combination in class_combinations:
        class_a = combination[0]
        class_b = combination[1]
        mask_class = ((mask == class_a) | (mask == class_b))
        data_split = dataset[mask_class]
        class_datasets.append(data_split)

    return class_datasets


def overall_accuracy(dataset, y_pred):
    # Extracting the class label
    y_true = dataset[:, 4]

    # Number of correctly classified data samples
    positive = np.sum(y_pred == y_true)

    accuracy = (positive / len(y_true)) * 100

    return accuracy

# The function q3_run is used to perform the experiments for the 3 different class combinations
def q3_run(class_combinations, idx, train_data_list, test_data_list, max_epoch):
    # It does not include a 1vsrest approach
    rest_flag = False
    l2_reg = 0


    # Regarding the various comparison combinations
    for combination in class_combinations:
        class_a = combination[0]
        class_b = combination[1]
        print(f"Class {class_a} vs Class {class_b}")

        train_set = train_data_list[idx]
        test_set = test_data_list[idx]

        perceptron3 = Perceptron(max_epoch=max_epoch, class_a=class_a, class_b=class_b,
                                 train_set=train_set)

        # Training Set
        perceptron3.train(l2_reg=l2_reg)
        y_pred_train = perceptron3.test(test_set=train_set, rest_flag=rest_flag)
        train_accuracy = perceptron3.accuracy(dataset=train_set, y_pred=y_pred_train)
        print(f"Train Accuracy: {train_accuracy:.1f}")

        # Testing Set
        y_pred_test = perceptron3.test(test_set=test_set, rest_flag=rest_flag)
        test_accuracy = perceptron3.accuracy(dataset=test_set, y_pred=y_pred_test)
        print(f"Test Accuracy: {test_accuracy:.1f}")
        print("----------")
        idx += 1

def multi_classification(perceptron_list, data, rest_flag):
    predictions = []
    features = data[:, :4]
    num_data = data.shape[0]

    for j in range(num_data):
        activation_scores = [
            perceptron_list[0].test(features[j], rest_flag=rest_flag),
            perceptron_list[1].test(features[j], rest_flag=rest_flag),
            perceptron_list[2].test(features[j], rest_flag=rest_flag)
        ]

        # Identify the class by its maximum activation score
        predictions.append(float(np.argmax(activation_scores) + 1))

    return predictions


def one_vs_rest(categories, idx, train_set, test_set, max_epoch, l2_reg):
    # It is a 1vsrest approach
    rest_flag = True

    # List of perceptron classes
    perceptron_list = [m for m in range(len(categories))]

    # Training 3 separate perceptrons
    for category in categories:
        idx = int(category - 1)

        perceptron_list[idx] = Perceptron(
            max_epoch=max_epoch, class_a=category, class_b=None, train_set=train_set
        )

        # Train perceptron with regularization
        perceptron_list[idx].train(l2_reg=l2_reg)

    # Obtaining predictions from each perceptron
    # Training set
    train_predictions = multi_classification(perceptron_list=perceptron_list, data=train_set, rest_flag=rest_flag)

    # Obtaining predictions from each perceptron
    # Test set
    test_predictions = multi_classification(perceptron_list=perceptron_list, data=test_set, rest_flag=rest_flag)

    general_train_accuracy = overall_accuracy(dataset=train_set, y_pred=train_predictions)
    print("Multiclass-Training Accuracy: %.1f" % general_train_accuracy)

    general_test_accuracy = overall_accuracy(dataset=test_set, y_pred=test_predictions)
    print("Multiclass-Testing Accuracy: %.1f" % general_test_accuracy)


def main():
    # Data file reading
    with open("train.data", "r") as file:
        train_data = [[
            float(value.split("-")[1]) if value.startswith("class-") else float(value)
            for value in line.strip().split(",")
        ] for line in file]

    with open("test.data", "r") as file:
        test_data = [[
            float(value.split("-")[1]) if value.startswith("class-") else float(value)
            for value in line.strip().split(",")
        ] for line in file]

    # Convert the list of lists to a numpy array
    training_data = np.array(train_data)

    # Rearrange the dataset
    np.random.seed(2)
    np.random.shuffle(training_data)


    # Convert the list of lists to a numpy array
    test_data = np.array(test_data)

    class_combinations = [(1.0, 2.0), (2.0, 3.0), (1.0, 3.0)]
    categories = [1.0, 2.0, 3.0]
    reg_list = [0.01, 0.1, 1.0, 10.0, 100.0]

    # List of data-List of each classCombinations
    train_data_list = masking1v1(training_data, class_combinations)
    test_data_list = masking1v1(test_data, class_combinations)

    max_epoch = 20

    # Questions Regarding Assignment Execution
    execute_question_3(class_combinations, train_data_list, test_data_list, max_epoch)
    execute_question_4(categories, training_data, test_data, max_epoch)
    execute_question_5(categories, training_data, test_data, max_epoch)

def execute_question_3(class_combinations, train_data_list, test_data_list, max_epoch):
    print("Question 3")
    q3_run(class_combinations=class_combinations, 
            idx=0,
            train_data_list=train_data_list, 
            test_data_list=test_data_list,
            max_epoch=max_epoch)

def execute_question_4(categories, training_data, test_data, max_epoch):
    print("Question 4")
    one_vs_rest(categories=categories, 
                 idx=0, 
                 train_set=training_data, 
                 test_set=test_data,
                 max_epoch=max_epoch, 
                 l2_reg=0)

def execute_question_5(categories, training_data, test_data, max_epoch):
    reg_list = [0.01, 0.1, 1.0, 10.0, 100.0]
    print("Question 5")
    for reg_term in reg_list:
        print(f'Regularisation term = {reg_term}')
        one_vs_rest(categories=categories, 
                     idx=0, 
                     train_set=training_data, 
                     test_set=test_data,
                     max_epoch=max_epoch, 
                     l2_reg=reg_term)

if __name__ == "__main__":
    main()
