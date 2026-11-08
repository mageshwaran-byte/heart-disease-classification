

# ==========================================================
# HEART DISEASE CLASSIFICATION DASHBOARD
# ==========================================================

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Classification Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CONSTANTS
# ==========================================================

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

MODEL_FILES = {
    "Logistic Regression":
        "models/logistic_regression.pkl",

    "Decision Tree":
        "models/decision_tree.pkl",

    "KNN":
        "models/knn.pkl",

    "Naive Bayes":
        "models/naive_bayes.pkl",

    "Support Vector Machine":
        "models/svm.pkl"
}

METRIC_COLUMNS = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]


# ==========================================================
# LOAD SAVED MODELS
# ==========================================================

@st.cache_resource
def load_saved_artifacts():
    """
    Load the saved scaler and trained models.

    Streamlit caches the results so model files are not
    repeatedly loaded during widget interactions.
    """

    loaded_scaler = joblib.load(
        "models/scaler.pkl"
    )

    loaded_models = {}

    for model_name, model_path in MODEL_FILES.items():

        loaded_models[model_name] = (
            joblib.load(model_path)
        )

    return loaded_scaler, loaded_models


# ==========================================================
# CALCULATE REQUIRED METRICS
# ==========================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):
    """
    Calculate all metrics required by the assignment.

    AUC cannot be calculated when actual target values contain
    only one class. In that case, AUC is returned as NaN.
    """

    if pd.Series(y_true).nunique() == 2:

        auc_value = roc_auc_score(
            y_true,
            probabilities
        )

    else:

        auc_value = np.nan

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "AUC":
            auc_value,

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                predictions
            )
    }


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ## ❤️ Heart Disease Classification

    ### Available Models

    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Gaussian Naive Bayes
    - Support Vector Machine
    """
)

st.sidebar.markdown("---")

st.sidebar.success(
    """
    👨‍🎓 **Student Details**

    **Name:** MAGESHWARAN P

    **BITS ID:** 2026AK05470

    **Course:** Machine Learning

    **Assignment:** ML Assignment 2
    """
)

st.sidebar.markdown("---")


# ==========================================================
# SAMPLE DATASET DOWNLOAD
# ==========================================================

if os.path.exists("heart.csv"):

    with open("heart.csv", "rb") as sample_file:

        sample_dataset_bytes = sample_file.read()

    st.sidebar.download_button(
        label="📥 Download Sample heart.csv",
        data=sample_dataset_bytes,
        file_name="heart.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.sidebar.warning(
        "heart.csv was not found in the project folder."
    )


# ==========================================================
# MAIN HEADER
# ==========================================================

st.title(
    "❤️ Heart Disease Classification Dashboard"
)

st.markdown(
    """
    Upload a labelled Heart Disease CSV file to evaluate all
    five trained machine learning models on the same dataset.

    The application automatically:

    - Evaluates every trained model
    - Computes all required classification metrics
    - Ranks models by F1 Score
    - Identifies the winning model
    - Compares any selected model with the winner
    - Displays confusion matrices and predictions
    """
)


# ==========================================================
# LOAD MODEL ARTIFACTS
# ==========================================================

models_loaded = True
model_loading_error = None

try:

    scaler, models = load_saved_artifacts()

except Exception as error:

    models_loaded = False
    model_loading_error = str(error)


if not models_loaded:

    st.error(
        "The trained models or scaler could not be loaded."
    )

    st.code(
        model_loading_error
    )

    st.info(
        """
        Ensure the following files are available:

        models/scaler.pkl

        models/logistic_regression.pkl

        models/decision_tree.pkl

        models/knn.pkl

        models/naive_bayes.pkl

        models/svm.pkl
        """
    )


# ==========================================================
# DATASET UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a labelled Heart Disease CSV file",
    type=["csv"],
    help=(
        "The CSV must contain the 13 feature columns and "
        "a binary target column named target."
    )
)


# ==========================================================
# BEFORE DATASET UPLOAD
# ==========================================================

if uploaded_file is None:

    st.info(
        "Upload `test_data.csv`, or download and upload "
        "the sample `heart.csv` dataset."
    )

    st.subheader(
        "Required CSV Columns"
    )

    required_columns_df = pd.DataFrame(
        {
            "Column":
                FEATURE_COLUMNS + ["target"],

            "Purpose":
                (
                    ["Input feature"]
                    * len(FEATURE_COLUMNS)
                    + ["Actual class label"]
                )
        }
    )

    st.dataframe(
        required_columns_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PROCESS UPLOADED DATASET
# ==========================================================

elif models_loaded:

    # ------------------------------------------------------
    # READ CSV
    # ------------------------------------------------------

    try:

        data = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"The uploaded CSV could not be read: {error}"
        )

        st.stop()

    st.success(
        "Dataset uploaded successfully."
    )


    # ------------------------------------------------------
    # VALIDATE COLUMNS
    # ------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS + ["target"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    extra_columns = [
        column
        for column in data.columns
        if column not in required_columns
    ]

    if missing_columns:

        st.error(
            "The uploaded dataset is missing required columns."
        )

        st.write(
            "Missing columns:",
            missing_columns
        )

        st.stop()

    if extra_columns:

        st.warning(
            "Extra columns were detected and will be ignored."
        )

        st.write(
            "Ignored columns:",
            extra_columns
        )


    # ------------------------------------------------------
    # PREPARE MODEL DATA
    # ------------------------------------------------------

    model_data = data[
        required_columns
    ].copy()

    if model_data.empty:

        st.error(
            "The uploaded dataset contains no records."
        )

        st.stop()


    # ------------------------------------------------------
    # CHECK MISSING VALUES
    # ------------------------------------------------------

    missing_value_count = int(
        model_data
        .isnull()
        .sum()
        .sum()
    )

    if missing_value_count > 0:

        st.error(
            "The uploaded dataset contains missing values."
        )

        missing_summary = (
            model_data
            .isnull()
            .sum()
        )

        missing_summary = (
            missing_summary[
                missing_summary > 0
            ]
        )

        st.dataframe(
            missing_summary.rename(
                "Missing Values"
            )
        )

        st.stop()


    # ------------------------------------------------------
    # CHECK TARGET VALUES
    # ------------------------------------------------------

    invalid_targets = (
        set(
            model_data["target"]
            .unique()
        )
        -
        {0, 1}
    )

    if invalid_targets:

        st.error(
            "The target column must contain only 0 and 1."
        )

        st.write(
            "Invalid target values:",
            sorted(invalid_targets)
        )

        st.stop()


    # ------------------------------------------------------
    # FEATURES AND TARGET
    # ------------------------------------------------------

    X = model_data[
        FEATURE_COLUMNS
    ].copy()

    y = (
        model_data["target"]
        .astype(int)
    )


    # ------------------------------------------------------
    # APPLY SAVED SCALER
    # ------------------------------------------------------

    try:

        X_scaled = scaler.transform(
            X
        )

    except Exception as error:

        st.error(
            f"Feature scaling failed: {error}"
        )

        st.info(
            "Ensure the uploaded feature columns match heart.csv."
        )

        st.stop()


    # ======================================================
    # EVALUATE ALL FIVE MODELS
    # ======================================================

    all_results = []
    model_outputs = {}

    for model_name, model in models.items():

        try:

            predictions = model.predict(
                X_scaled
            )

            # ----------------------------------------------
            # PROBABILITY SCORES
            # ----------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        X_scaled
                    )[:, 1]
                )

            elif hasattr(
                model,
                "decision_function"
            ):

                raw_scores = (
                    model.decision_function(
                        X_scaled
                    )
                )

                score_min = (
                    raw_scores.min()
                )

                score_max = (
                    raw_scores.max()
                )

                if score_max > score_min:

                    probabilities = (
                        raw_scores
                        -
                        score_min
                    ) / (
                        score_max
                        -
                        score_min
                    )

                else:

                    probabilities = (
                        np.zeros_like(
                            raw_scores,
                            dtype=float
                        )
                    )

            else:

                probabilities = (
                    predictions.astype(float)
                )


            # ----------------------------------------------
            # MODEL METRICS
            # ----------------------------------------------

            metrics = calculate_metrics(
                y,
                predictions,
                probabilities
            )

            all_results.append(
                {
                    "Model":
                        model_name,

                    **metrics
                }
            )

            model_outputs[
                model_name
            ] = {
                "predictions":
                    predictions,

                "probabilities":
                    probabilities,

                "metrics":
                    metrics
            }

        except Exception as error:

            st.error(
                f"{model_name} evaluation failed: {error}"
            )


    if not all_results:

        st.error(
            "None of the trained models could be evaluated."
        )

        st.stop()


    # ======================================================
    # CREATE LEADERBOARD
    # ======================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "F1",
                "MCC",
                "AUC"
            ],
            ascending=False,
            na_position="last"
        )
        .reset_index(
            drop=True
        )
    )

    results_df.insert(
        0,
        "Rank",
        range(
            1,
            len(results_df) + 1
        )
    )

    winner_row = (
        results_df.iloc[0]
    )

    winner_name = (
        winner_row["Model"]
    )

    winner_output = (
        model_outputs[
            winner_name
        ]
    )

    winner_metrics = (
        winner_output[
            "metrics"
        ]
    )


    # ======================================================
    # WINNER CONCLUSION AT TOP
    # ======================================================

    st.markdown("---")

    st.subheader(
        "🏆 Overall Model Conclusion"
    )

    winner_col1, winner_col2, winner_col3 = (
        st.columns(3)
    )

    winner_col1.metric(
        "Winning Model",
        winner_name
    )

    winner_col2.metric(
        "Winner F1 Score",
        f"{winner_row['F1']:.4f}"
    )

    winner_col3.metric(
        "Winner MCC",
        f"{winner_row['MCC']:.4f}"
    )

    st.success(
        (
            f"**{winner_name}** achieved the highest "
            f"F1 Score on the uploaded dataset."
        )
    )

    if pd.isna(
        winner_row["AUC"]
    ):

        st.warning(
            "AUC is unavailable because the uploaded target "
            "contains only one class."
        )


    # ======================================================
    # MODEL LEADERBOARD
    # ======================================================

    st.subheader(
        "📋 Model Leaderboard"
    )

    display_results_df = (
        results_df.copy()
    )

    display_results_df[
        METRIC_COLUMNS
    ] = (
        display_results_df[
            METRIC_COLUMNS
        ]
        .round(4)
    )

    st.dataframe(
        display_results_df,
        use_container_width=True,
        hide_index=True
    )

    leaderboard_csv = (
        display_results_df
        .to_csv(
            index=False
        )
    )

    st.download_button(
        label="📥 Download Model Comparison",
        data=leaderboard_csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )


    # ======================================================
    # SELECT MODEL FOR DETAILED COMPARISON
    # ======================================================

    st.markdown("---")

    model_names = list(
        models.keys()
    )

    selected_model_name = (
        st.selectbox(
            "Select a model for detailed comparison",
            model_names,
            index=model_names.index(
                winner_name
            )
        )
    )

    selected_output = (
        model_outputs[
            selected_model_name
        ]
    )

    selected_metrics = (
        selected_output[
            "metrics"
        ]
    )

    selected_is_winner = (
        selected_model_name
        ==
        winner_name
    )


    # ======================================================
    # WINNER SELECTED
    # ======================================================

    if selected_is_winner:

        st.subheader(
            f"🏆 {selected_model_name} is the Winning Model"
        )

        st.success(
            (
                f"**{selected_model_name}** is already the "
                "highest-ranked model. A winner-versus-winner "
                "comparison is not required."
            )
        )

        winner_summary_df = (
            pd.DataFrame(
                {
                    "Metric":
                        METRIC_COLUMNS,

                    "Score":
                        [
                            winner_metrics[
                                metric
                            ]
                            for metric
                            in METRIC_COLUMNS
                        ]
                }
            )
        )

        winner_summary_df[
            "Score"
        ] = (
            winner_summary_df[
                "Score"
            ]
            .round(4)
        )

        st.dataframe(
            winner_summary_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # WINNER PERFORMANCE CHART
        # --------------------------------------------------

        chart_values = [
            winner_metrics[
                metric
            ]
            for metric
            in METRIC_COLUMNS
        ]

        fig_winner, ax_winner = (
            plt.subplots(
                figsize=(10, 5)
            )
        )

        bars = ax_winner.bar(
            METRIC_COLUMNS,
            chart_values,
            color=[
                "#2563eb",
                "#0ea5e9",
                "#14b8a6",
                "#22c55e",
                "#84cc16",
                "#f59e0b"
            ]
        )

        ax_winner.set_title(
            f"{winner_name} Performance"
        )

        ax_winner.set_xlabel(
            "Metric"
        )

        ax_winner.set_ylabel(
            "Score"
        )

        ax_winner.set_ylim(
            -1.0,
            1.05
        )

        ax_winner.grid(
            axis="y",
            alpha=0.25
        )

        labels = []

        for value in chart_values:

            if pd.isna(value):
                labels.append("N/A")
            else:
                labels.append(
                    f"{value:.4f}"
                )

        ax_winner.bar_label(
            bars,
            labels=labels,
            padding=3
        )

        st.pyplot(
            fig_winner
        )

        plt.close(
            fig_winner
        )


    # ======================================================
    # NON-WINNER SELECTED
    # ======================================================

    else:

        st.subheader(
            (
                f"⚖️ {selected_model_name} "
                f"vs {winner_name}"
            )
        )

        st.info(
            (
                f"Comparing **{selected_model_name}** "
                f"against the winning model, "
                f"**{winner_name}**."
            )
        )

        # --------------------------------------------------
        # COMPARISON TABLE
        # --------------------------------------------------

        comparison_df = (
            pd.DataFrame(
                {
                    "Metric":
                        METRIC_COLUMNS,

                    "Selected Model":
                        [
                            selected_metrics[
                                metric
                            ]
                            for metric
                            in METRIC_COLUMNS
                        ],

                    "Winning Model":
                        [
                            winner_metrics[
                                metric
                            ]
                            for metric
                            in METRIC_COLUMNS
                        ]
                }
            )
        )

        comparison_display_df = (
            comparison_df.copy()
        )

        comparison_display_df[
            [
                "Selected Model",
                "Winning Model"
            ]
        ] = (
            comparison_display_df[
                [
                    "Selected Model",
                    "Winning Model"
                ]
            ]
            .round(4)
        )

        comparison_display_df = (
            comparison_display_df
            .rename(
                columns={
                    "Selected Model":
                        selected_model_name,

                    "Winning Model":
                        winner_name
                }
            )
        )

        st.dataframe(
            comparison_display_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DIFFERENCE FROM WINNER
        # --------------------------------------------------

        difference_df = pd.DataFrame(
            {
                "Metric":
                    METRIC_COLUMNS,

                "Selected minus Winner":
                    [
                        (
                            selected_metrics[
                                metric
                            ]
                            -
                            winner_metrics[
                                metric
                            ]
                        )
                        for metric
                        in METRIC_COLUMNS
                    ]
            }
        )

        difference_df[
            "Selected minus Winner"
        ] = (
            difference_df[
                "Selected minus Winner"
            ]
            .round(4)
        )

        st.markdown(
            "#### Difference from Winning Model"
        )

        st.dataframe(
            difference_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # COMPARISON CHART
        # --------------------------------------------------

        comparison_plot_df = (
            comparison_df
            .set_index(
                "Metric"
            )
            [
                [
                    "Selected Model",
                    "Winning Model"
                ]
            ]
            .rename(
                columns={
                    "Selected Model":
                        selected_model_name,

                    "Winning Model":
                        winner_name
                }
            )
        )

        fig_compare, ax_compare = (
            plt.subplots(
                figsize=(11, 5)
            )
        )

        comparison_plot_df.plot(
            kind="bar",
            ax=ax_compare,
            color=[
                "#2563eb",
                "#16a34a"
            ]
        )

        ax_compare.set_title(
            "Selected Model Compared with Winning Model"
        )

        ax_compare.set_xlabel(
            "Metric"
        )

        ax_compare.set_ylabel(
            "Score"
        )

        ax_compare.set_ylim(
            -1.0,
            1.05
        )

        ax_compare.tick_params(
            axis="x",
            rotation=0
        )

        ax_compare.grid(
            axis="y",
            alpha=0.25
        )

        ax_compare.legend(
            title="Model"
        )

        st.pyplot(
            fig_compare
        )

        plt.close(
            fig_compare
        )


    # ======================================================
    # DETAILED TABS
    # ======================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Selected Model Metrics",
            "📉 Confusion Matrix",
            "🔮 Predictions",
            "📂 Dataset"
        ]
    )


    # ======================================================
    # TAB 1: SELECTED MODEL METRICS
    # ======================================================

    with tab1:

        st.subheader(
            f"{selected_model_name} Performance"
        )

        metric_row1 = st.columns(3)

        metric_row1[0].metric(
            "Accuracy",
            f"{selected_metrics['Accuracy']:.4f}"
        )

        if pd.isna(
            selected_metrics["AUC"]
        ):

            selected_auc_text = "N/A"

        else:

            selected_auc_text = (
                f"{selected_metrics['AUC']:.4f}"
            )

        metric_row1[1].metric(
            "AUC",
            selected_auc_text
        )

        metric_row1[2].metric(
            "Precision",
            f"{selected_metrics['Precision']:.4f}"
        )

        metric_row2 = st.columns(3)

        metric_row2[0].metric(
            "Recall",
            f"{selected_metrics['Recall']:.4f}"
        )

        metric_row2[1].metric(
            "F1 Score",
            f"{selected_metrics['F1']:.4f}"
        )

        metric_row2[2].metric(
            "MCC",
            f"{selected_metrics['MCC']:.4f}"
        )

        if selected_is_winner:

            st.success(
                "The selected model is the overall winner."
            )

        else:

            f1_difference = (
                winner_metrics["F1"]
                -
                selected_metrics["F1"]
            )

            st.info(
                (
                    f"The selected model's F1 Score "
                    f"is {f1_difference:.4f} below "
                    f"{winner_name}."
                )
            )


    # ======================================================
    # TAB 2: CONFUSION MATRIX
    # ======================================================

    with tab2:

        selected_cm = confusion_matrix(
            y,
            selected_output[
                "predictions"
            ]
        )

        if selected_is_winner:

            st.subheader(
                (
                    f"{selected_model_name} "
                    "Confusion Matrix"
                )
            )

            fig_selected_cm, ax_selected_cm = (
                plt.subplots(
                    figsize=(6, 5)
                )
            )

            sns.heatmap(
                selected_cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                cbar=False,
                xticklabels=[
                    "No Disease",
                    "Disease"
                ],
                yticklabels=[
                    "No Disease",
                    "Disease"
                ],
                ax=ax_selected_cm
            )

            ax_selected_cm.set_xlabel(
                "Predicted"
            )

            ax_selected_cm.set_ylabel(
                "Actual"
            )

            ax_selected_cm.set_title(
                f"Winner: {selected_model_name}"
            )

            st.pyplot(
                fig_selected_cm
            )

            plt.close(
                fig_selected_cm
            )

        else:

            winner_cm = confusion_matrix(
                y,
                winner_output[
                    "predictions"
                ]
            )

            cm_col1, cm_col2 = st.columns(2)

            with cm_col1:

                st.markdown(
                    f"### {selected_model_name}"
                )

                fig_selected_cm, ax_selected_cm = (
                    plt.subplots(
                        figsize=(5, 4)
                    )
                )

                sns.heatmap(
                    selected_cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=[
                        "No Disease",
                        "Disease"
                    ],
                    yticklabels=[
                        "No Disease",
                        "Disease"
                    ],
                    ax=ax_selected_cm
                )

                ax_selected_cm.set_xlabel(
                    "Predicted"
                )

                ax_selected_cm.set_ylabel(
                    "Actual"
                )

                st.pyplot(
                    fig_selected_cm
                )

                plt.close(
                    fig_selected_cm
                )

            with cm_col2:

                st.markdown(
                    f"### Winner: {winner_name}"
                )

                fig_winner_cm, ax_winner_cm = (
                    plt.subplots(
                        figsize=(5, 4)
                    )
                )

                sns.heatmap(
                    winner_cm,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    cbar=False,
                    xticklabels=[
                        "No Disease",
                        "Disease"
                    ],
                    yticklabels=[
                        "No Disease",
                        "Disease"
                    ],
                    ax=ax_winner_cm
                )

                ax_winner_cm.set_xlabel(
                    "Predicted"
                )

                ax_winner_cm.set_ylabel(
                    "Actual"
                )

                st.pyplot(
                    fig_winner_cm
                )

                plt.close(
                    fig_winner_cm
                )


    # ======================================================
    # TAB 3: PREDICTIONS
    # ======================================================

    with tab3:

        prediction_df = (
            model_data.copy()
        )

        if selected_is_winner:

            prediction_df[
                f"{winner_name} Prediction"
            ] = (
                winner_output[
                    "predictions"
                ]
            )

            prediction_df[
                f"{winner_name} Probability"
            ] = (
                winner_output[
                    "probabilities"
                ]
            )

        else:

            prediction_df[
                f"{selected_model_name} Prediction"
            ] = (
                selected_output[
                    "predictions"
                ]
            )

            prediction_df[
                f"{selected_model_name} Probability"
            ] = (
                selected_output[
                    "probabilities"
                ]
            )

            prediction_df[
                f"{winner_name} Prediction"
            ] = (
                winner_output[
                    "predictions"
                ]
            )

            prediction_df[
                f"{winner_name} Probability"
            ] = (
                winner_output[
                    "probabilities"
                ]
            )

        st.dataframe(
            prediction_df.head(25),
            use_container_width=True
        )

        if selected_is_winner:

            distribution_df = pd.DataFrame(
                {
                    winner_name:
                        pd.Series(
                            winner_output[
                                "predictions"
                            ]
                        )
                        .value_counts()
                }
            )

        else:

            distribution_df = pd.DataFrame(
                {
                    selected_model_name:
                        pd.Series(
                            selected_output[
                                "predictions"
                            ]
                        )
                        .value_counts(),

                    winner_name:
                        pd.Series(
                            winner_output[
                                "predictions"
                            ]
                        )
                        .value_counts()
                }
            )

        distribution_df = (
            distribution_df
            .fillna(0)
            .reindex(
                [0, 1],
                fill_value=0
            )
        )

        distribution_df.index = [
            "No Disease",
            "Disease"
        ]

        fig_distribution, ax_distribution = (
            plt.subplots(
                figsize=(8, 4)
            )
        )

        if selected_is_winner:

            distribution_colors = [
                "#16a34a"
            ]

        else:

            distribution_colors = [
                "#2563eb",
                "#16a34a"
            ]

        distribution_df.plot(
            kind="bar",
            ax=ax_distribution,
            color=distribution_colors
        )

        ax_distribution.set_title(
            "Prediction Distribution"
        )

        ax_distribution.set_xlabel(
            "Predicted Class"
        )

        ax_distribution.set_ylabel(
            "Number of Records"
        )

        ax_distribution.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(
            fig_distribution
        )

        plt.close(
            fig_distribution
        )

        predictions_csv = (
            prediction_df
            .to_csv(
                index=False
            )
        )

        st.download_button(
            label="📥 Download Detailed Predictions",
            data=predictions_csv,
            file_name=(
                "heart_disease_predictions.csv"
            ),
            mime="text/csv"
        )


    # ======================================================
    # TAB 4: DATASET
    # ======================================================

    with tab4:

        st.subheader(
            "Dataset Summary"
        )

        (
            dataset_col1,
            dataset_col2,
            dataset_col3,
            dataset_col4
        ) = st.columns(4)

        dataset_col1.metric(
            "Rows",
            model_data.shape[0]
        )

        dataset_col2.metric(
            "Input Features",
            len(FEATURE_COLUMNS)
        )

        dataset_col3.metric(
            "Missing Values",
            int(
                model_data
                .isnull()
                .sum()
                .sum()
            )
        )

        dataset_col4.metric(
            "Target Classes",
            int(
                y.nunique()
            )
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            model_data.head(20),
            use_container_width=True
        )

        st.subheader(
            "Feature Statistics"
        )

        st.dataframe(
            model_data
            .describe()
            .T,
            use_container_width=True
        )

        st.subheader(
            "Target Distribution"
        )

        target_distribution = (
            y.value_counts()
            .sort_index()
            .rename(
                index={
                    0: "No Disease",
                    1: "Disease"
                }
            )
        )

        st.bar_chart(
            target_distribution
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Machine Learning Assignment 2 | "
    "MAGESHWARAN P | "
    "BITS ID: 2026AK05470"
)
