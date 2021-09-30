# Performs body analysis

import pathlib
import time
import pandas as pd
from nltk.corpus import wordnet
from itertools import chain
import Utils
import text2emotion as txt2emo


# Function for Urgency detection based on words such as urgent, soon, now
def urgent(text):
    # Create a list of words related to urgent action
    urgent_synonyms = set(chain.from_iterable([word.lemma_names() for word in wordnet.synsets("now")]))
    urgent_synonyms = urgent_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in wordnet.synsets("urgent")]))
    urgent_synonyms = urgent_synonyms | set(
        chain.from_iterable([word.lemma_names() for word in wordnet.synsets("soon")]))
    if any(word in urgent_synonyms for word in text.split()):
        return 1
    else:
        return 0
# Prachi Patil


# Time constraint
# Time not always mentioned in hours or days, date is at times specified.
# Function to check for time in mails
def time_f(text):
    time_feature = {"hours", "hour", "day", "minutes", "seconds", "today", "yesterday"}
    if any(word in time_feature for word in text.split()):
        return 1
    else:
        return 0


# phish mail corpus is taken from user Jose https://monkey.org/~jose/phishing/, name : Jose and
# legitimate mail corpus is taken from personal college mails - name: Prachi Patil.
# Function to check for either of the names in mails
def name(text):
    # all the words are in lower case
    name = {"prachi", "jose"}
    if any(word in name for word in text.split()):
        return 1
    else:
        return 0

# Create body features
def create_body_features():
    path = "../data/all_mails.csv"
    body_feature_list = []
    mails_df = pd.read_csv(path)
    for index, entry in mails_df.iterrows():
        body = entry["body"]
        phish = entry["phish"]
        if isinstance(body, float):
            body_row = {'urgent': 1, 'name': 1, 'time': 1,
                        'threatening': 1,
                        'phish': 1, "Happy": 0, "Sad": 0, "Fear": 1,
                        "Angry": "0", "Surprise": 0}
        else:
            # Creates a dictionary of respective features
            body_row = {'urgent': urgent(body), 'name': name(body), 'time': time_f(body),
                        'threatening': Utils.check_threatening_words(body),
                        'phish': phish}

            # calculate emotion in text, return dict
            emo_dict = txt2emo.get_emotion(body)
            # update row with emotions
            body_row.update(emo_dict)
        # Append the dictionary of each mail to the list
        body_feature_list.append(body_row)
    return body_feature_list


# Performs Body without emotion analysis
def no_emotion_analysis(body_features_df):
    print("********************     No Emotion analysis     ****************************")
    # Drops the emotions coloums from features dataset
    X = body_features_df.drop(['Angry'], axis=1)
    X = X.drop(['Happy'], axis=1)
    X = X.drop(['Sad'], axis=1)
    X = X.drop(['Surprise'], axis=1)
    X = X.drop(['Fear'], axis=1)
    Utils.result_list = []
    # Train, test the models based on the features, return a list of results of performance of all the models.
    result = Utils.predict_results(X, "no_emotion_body")
    # Store the result for further processing
    Utils.create_csv(result, "../data/no_emotion_body.csv")
    pass


def body_analysis():
    start_time = time.time()
    print("Body Start: ", start_time)
    print("***********************          Performing body Analysis         ***********************")
    path = "../data/body_features.csv"
    file = pathlib.Path(path)
    # Checks for feature file
    if file.exists():
        # Data from features file to Datafrmae
        body_features_df = pd.read_csv(path)
        # no_emotion_analysis(body_features_df)
    else:
        # Create feature file
        features_list = create_body_features()
        # Features list to dataframe
        body_features_df = pd.DataFrame(features_list)
        # Store feature file
        Utils.create_csv(body_features_df, path)

    results = Utils.predict_results(body_features_df, "body_features")
    Utils.create_df_csv(results, "../data/body_result.csv")
    end_time = time.time()
    print("Body end time: ", end_time)
    print("Total time for body processing in minutes : ", (end_time - start_time) / 60)
    # Perform body analysis by removing emotions from the features extrcted
    no_emotion_analysis(body_features_df)


if __name__ == "__main__":
    body_analysis()

