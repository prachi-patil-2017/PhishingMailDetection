# Class for common methods used in the project

import pathlib
import pickle
import re
from itertools import chain
import spacy
import pandas as pd
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score

result_list = []

# Checks if the wordlist file exists or not. If the file does not exist, it creates the file
def check_wordslist():
    path = "../wordslist/threatening_words.txt"
    file = pathlib.Path(path)
    if file.exists():
        return True
    else:
        # Opens the file to write in the path specified
        f = open(path, "w+")
        for word in create_threatening_wordslist():
            f.write(word + "\n")
        f.close()


# Create a list of words
def check_threatening_words(text):
    # Creates a list of threatening words
    if isinstance(text, float):
        return 1
    else:
        # Checks if the wordlist file exists or not. If the file does not exist, it creates the file
        check_wordslist()

    # The words are fetched from the file
    wordlist = open("../wordslist/threatening_words.txt", "r").read().split("\n")
    threatening_synonyms = set(wordlist)

    common_words = list(threatening_synonyms.intersection(text.split(" ")))
    if len(common_words) > 0:
        return 1
    else:
        return 0

# Creates a list of threatening words
def create_threatening_wordslist():
    threatening_synonyms = set()
    # Gets the synonyms of the word  passed
    cancel_synonyms = wordnet.synsets("cancel")
    # Adds the synonyms to the set
    threatening_synonyms = set(chain.from_iterable([word.lemma_names() for word in cancel_synonyms]))

    suspend_synonyms = wordnet.synsets("suspend")
    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in suspend_synonyms]))  # append new synonyms to the set

    block_synonyms = wordnet.synsets("blocked")
    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in block_synonyms]))

    update_synonyms = wordnet.synsets("update")
    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in update_synonyms]))

    delete_synonyms = wordnet.synsets("delete")
    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in delete_synonyms]))

    upgrade_synonyms = wordnet.synsets("upgrade")
    threatening_synonyms = threatening_synonyms | set(chain.from_iterable([word.lemma_names() for word in upgrade_synonyms]))

    stp_synonyms = wordnet.synsets("stop")
    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in stp_synonyms]))

    threatening_synonyms = threatening_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in wordnet.synsets("disable")]))
    return threatening_synonyms


# Performs evaluation of the models
def evaluate_model(target_var, predict_var, model):
    # Calculates accuracy
    a = accuracy_score(target_var, predict_var) * 100
    # Calculate confusion matrix result
    tn, fp, fn, tp = create_confusion_matrix(target_var, predict_var)
    # Calculates precision
    precision = precision_score(target_var, predict_var) * 100
    # Calculates recall
    recall = recall_score(target_var, predict_var) * 100
    # Calculates F1 score
    f1 = f1_score(target_var, predict_var) * 100
    # Creates a dictionary of the result
    result_dict = {"Model": model, "tn": tn, "tp": tp, "fp": fp, "fn": fn, "Accuracy": a, "Precision": precision,
                   "Recall": recall, "F1": f1}
    # Prints the result
    print_model_result(model, tn, fp, fn, tp, a, precision, recall, f1)
    # Appends the result of each model to global result_list
    result_list.append(result_dict)
    pass


# Performs Model training and testing
def get_result_from_model(classifier, model, without_target_train, without_target_test, target_train, target_test,
                          filename):
    # Checks the classifiers trained file is available or not.
    # If the trained model is not available then
    if not pathlib.Path(filename).exists():
        # Model training
        classifier.fit(without_target_train, target_train)
        # Save the trained model
        pickle.dump(classifier, open(filename, 'wb'))
    else:
        # If the trained model is available then the model is loaded
        classifier = pickle.load(open(filename, 'rb'))
    # Predict result
    predict = classifier.predict(without_target_test)
    # Get result for the predicted data
    evaluate_model(target_test, predict, model)
    pass


# Splits the data into training and testing data
# Initialise the model for classification
def predict_results(dataframe,type):
    from sklearn.model_selection import train_test_split
    X = dataframe.drop(['phish'], axis=1)  # axis 1 drops columns, drop target variable
    Y = dataframe['phish']
    Y = Y.astype('int')  # cast pandas object to int
    without_target_train, without_target_test, target_train, target_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    from sklearn.ensemble import RandomForestClassifier  # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier()  # Train the model on training data
    model = "RF"
    filename = "../trained_models/" +type+"_"+ model + ".sav"
    get_result_from_model(rf, model, without_target_train, without_target_test, target_train, target_test, filename)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    model = "LR"
    filename = "../trained_models/" +type+"_"+ model + ".sav"
    get_result_from_model(lr, model, without_target_train, without_target_test, target_train, target_test, filename)

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    model = "KNN"
    filename = "../trained_models/" +type+"_"+ model + ".sav"
    get_result_from_model(neigh, model, without_target_train, without_target_test, target_train, target_test, filename)

    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', gamma='auto')
    model = "SVC"
    filename = "../trained_models/" +type+"_" + model + ".sav"
    get_result_from_model(svm, model, without_target_train, without_target_test, target_train, target_test, filename)

    return pd.DataFrame(result_list)


# Creates confusion mattrix
def create_confusion_matrix(target_var, predicted_var):
    # calculates the value of confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_var, predicted_var).ravel()
    return tn, fp, fn, tp


def cleaning_txt(text):
    # Loads spacy small english dictionary
    eng_dict = spacy.load("en_core_web_sm")
    # Loads stop words from the english dictionary
    all_stopwords = eng_dict.Defaults.stop_words

    text = re.sub(r'<.*?>', '', text)  # remove all <>
    text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1", text)
    # () is capturing group
    # [a-zA-Z] selects the letter, + selects multiple letters,
    # \d selects digits, d+ selects multiple digits
    # r"\1" selects only first group
    # Above regex replaces "have20" with only "have", removing digit 20.
    punctuations = r'!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    # Each character in punctuations, is set as a key in translate_dict with " "(space as a value)
    translate_dict = dict((c, " ") for c in punctuations)

    # maketrans method replaces the punctuation sign by its ascii value
    # maketrans(str1,str2) --> str1: characters to replace ; str2: replacement character
    translate_map = str.maketrans(translate_dict)

    # translate method replaces the keys of translate map found in the text, with corresponding value of the key
    text = text.translate(translate_map)

    # Removing stop words
    tokens = [word for word in text.split() if word.lower() not in all_stopwords]
    text = ' '.join(tokens)

    # To lemmatize each word, the word should be passed to an english dictionary, to get its information
    text = " ".join(word.lemma_ for word in eng_dict(text))
    # text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    # text = spell(text)
    return text


# Prints the result of the models
def print_model_result(model, tn, fp, fn, tp, a, pre,recall, f1):
    print("| Result for model: ", model)
    print("| Accuracy: ", a)
    print("| True negative: ", tn)
    print("| False negative: ", fn)
    print("| False positive: ", fp)
    print("| True positive: ", tp)
    print("| Precision: ", pre)
    print("| Recall: ", recall)
    print("| F1 score: ", f1)

    print("\n")


# Create a csv from the list passed
def create_csv(dataframe, path):
    print("File created in: ",path)
    # create a csv file to store the dataframe in the path passed to the function
    pd.DataFrame.to_csv(dataframe, path_or_buf=path, header=True, index=False)


# Create a dataframe and csv from the list passed
def create_df_csv(data_list, path_to_csv):
    # Create dataframe from the list
    df = pd.DataFrame(data_list)
    # Create CSV file from the list
    create_csv(df, path=path_to_csv)
    return df


