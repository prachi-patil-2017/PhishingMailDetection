import pandas as pd
import matplotlib.pyplot as plt


def plot_graphs():
    # Plot evaluation parameters
    model_performance_df = pd.read_csv(r"..\data\header_results.csv")
    header_eval_params_df = model_performance_df[['Model', 'Accuracy', 'Recall', "Precision", "F1"]]
    ax = header_eval_params_df.plot(x='Model', kind='bar', stacked=False,
                                      title='EVALUATION METRIC BASED ON HEADER FEATURES',
                                      figsize=(10, 10), colormap='summer')
    ax.set_xlabel("CLASSIFIERS", fontsize=10)
    ax.set_ylabel("SCORE VALUES", fontsize=10)

    # Plot confusion matrix
    model_performance_df.drop(['Accuracy', 'Precision', 'Recall', 'F1'], inplace=True, axis=1)
    header_eval_params_df.set_index('Model', inplace=True)
    header_eval_params_df.plot(kind="barh", figsize=(10, 10), colormap='tab20c')
    plt.title("Header result analysis", fontsize=20)
    plt.xlabel("Parameter", fontsize=10)
    plt.ylabel("Model", fontsize=10)

    # Plot evaluation parameters
    model_performance_df = pd.read_csv(
        r"..\data\body_result.csv")
    body_eval_params_df = model_performance_df[['Model', 'Accuracy', 'Recall', "Precision", "F1"]]
    ax = body_eval_params_df.plot(x='Model', kind='bar', stacked=False,
                                      title='EVALUATION METRIC BASED ON BODY FEATURES',
                                      figsize=(10, 10), colormap='summer')
    ax.set_xlabel("CLASSIFIERS", fontsize=10)
    ax.set_ylabel("SCORE VALUES", fontsize=10)

    # Plot confusion matrix
    model_performance_df.drop(['Accuracy', 'Precision', 'Recall', 'F1'], inplace=True, axis=1)
    body_eval_params_df.set_index('Model', inplace=True)
    body_eval_params_df.plot(kind="barh", figsize=(10, 10), colormap='tab20c')
    plt.title("Body result analysis", fontsize=20)
    plt.xlabel("Parameter", fontsize=10)
    plt.ylabel("Model", fontsize=10)

    # Plot evaluation parameters
    model_performance_df = pd.read_csv(r"..\data\all_features_h_b_results.csv")
    all_eval_params_df = model_performance_df[['Model', 'Accuracy', 'Recall', "Precision", "F1"]]
    ax = all_eval_params_df.plot(x='Model', kind='bar', stacked=False,
                                      title='EVALUATION METRIC BASED ON ALL FEATURES',
                                      figsize=(10, 10), colormap='summer')
    ax.set_xlabel("CLASSIFIERS", fontsize=10)
    ax.set_ylabel("SCORE VALUES", fontsize=10)

    # Plot confusion matrix
    model_performance_df.drop(['Accuracy', 'Precision', 'Recall', 'F1'], inplace=True, axis=1)
    all_eval_params_df.set_index('Model', inplace=True)
    all_eval_params_df.plot(kind="barh", figsize=(10, 10), colormap='tab20c')
    plt.title("All features result analysis", fontsize=20)
    plt.xlabel("Parameter", fontsize=10)
    plt.ylabel("Model", fontsize=10)

    plt.show()




if __name__ == "__main__":
    plot_graphs()