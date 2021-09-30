import pathlib
import time
import HeaderAnalysis
import BodyAnalysis
import pandas as pd
import Intial_Setup
from main_directory import Utils, VisualiseResult


def combined_analysis():
    print("***********************          Performing header + body Analysis         ***********************")
    header_df = pd.read_csv("../data/header_features.csv")
    # Drop the common columns from both datasets
    # axis 1 drops columns, drop target variable which is common in both the feature files
    X = header_df.drop(['phish'], axis=1)
    body_df = pd.read_csv("../data/body_features.csv")
    # Concats the header and body features
    all_features = pd.concat([X, body_df], axis=1)

    # call the function to train, test and evaluate the model
    all_features_result = Utils.predict_results(dataframe=all_features, type="All_features_h_b")
    Utils.create_csv(all_features_result, "../data/all_features_h_b_results.csv")
    pass


if __name__ == "__main__":

    start_time = time.time()
    print("Start: time for complete mail analysis", start_time)
    # Check for all_mail_file which contains separated data

    if not pathlib.Path("../data/all_mails_new.csv").exists():
        start_time = time.time()
        print("Start: time for complete mail analysis", start_time)
        # Call the function to create all_mails file
        Intial_Setup.main()
        end_time = time.time()
        print("end time: ", end_time)
        print("Total time for complete mail analysis in minutes: ", (end_time - start_time)/60)

    # Checks if header result file is created
    # If not created then performs header analysis
    if not pathlib.Path("../data/header_results.csv").exists():
        HeaderAnalysis.header_analysis()

    # Checks if body result file is created
    # If not created then performs body analysis
    if not pathlib.Path("../data/body_result.csv").exists():
        BodyAnalysis.body_analysis()

    # Checks if all feature result file is created
    # If not created then performs combined header and body analysis
    if not pathlib.Path("../data/all_features_h_b_results.csv").exists():
        combined_analysis()
    end_time = time.time()
    print("end time: ", end_time)
    print("Total time for complete mail analysis in minutes: ", (end_time - start_time)/60)

    VisualiseResult.plot_graphs()

