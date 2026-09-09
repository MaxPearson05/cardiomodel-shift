from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


st.set_page_config(
    page_title="CardioModel Shift",
    page_icon="🫀",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

GITHUB_URL = "https://github.com/MaxPearson05/cardiomodel-shift"

MODEL_FEATURES = [
    "age",
    "resting_blood_pressure",
    "serum_cholesterol",
    "max_heart_rate",
    "st_depression",
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "resting_ecg",
    "exercise_induced_angina",
]

MODEL_FILES = {
    ("Logistic Regression", "Cleveland"):
        "logistic_regression_cleveland.joblib",
    ("Random Forest", "Cleveland"):
        "random_forest_cleveland.joblib",
    ("Logistic Regression", "Hungary"):
        "logistic_regression_hungary.joblib",
    ("Random Forest", "Hungary"):
        "random_forest_hungary.joblib",
}

CATEGORY_LABELS = {
    "sex": {
        0: "Female",
        1: "Male",
    },
    "chest_pain_type": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fasting_blood_sugar": {
        0: "≤ 120 mg/dL",
        1: "> 120 mg/dL",
    },
    "resting_ecg": {
        0: "Normal",
        1: "ST-T abnormality",
        2: "LV hypertrophy",
    },
    "exercise_induced_angina": {
        0: "No",
        1: "Yes",
    },
}

FEATURE_NAMES = {
    "age": "Age",
    "resting_blood_pressure": "Resting blood pressure",
    "serum_cholesterol": "Serum cholesterol",
    "max_heart_rate": "Maximum heart rate",
    "st_depression": "ST depression",
    "sex": "Sex",
    "chest_pain_type": "Chest pain type",
    "fasting_blood_sugar": "Fasting blood sugar",
    "resting_ecg": "Resting ECG",
    "exercise_induced_angina": "Exercise-induced angina",
}

RESULT_NAMES = {
    (0, 0): "True negative",
    (0, 1): "False positive",
    (1, 0): "False negative",
    (1, 1): "True positive",
}


@st.cache_resource
def load_saved_model(file_name):
    return load(MODELS_DIR / file_name)


@st.cache_data
def load_demo_patients():
    return pd.read_csv(DEMO_DIR / "patient_examples.csv")


def format_feature_value(feature, value):
    if pd.isna(value):
        return "Missing"

    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature].get(
            int(value),
            f"Code {int(value)}",
        )

    if feature == "age":
        return f"{int(value)} years"

    if feature == "resting_blood_pressure":
        return f"{value:.0f} mmHg"

    if feature == "serum_cholesterol":
        return f"{value:.0f} mg/dL"

    if feature == "max_heart_rate":
        return f"{value:.0f} bpm"

    if feature == "st_depression":
        return f"{value:.1f}"

    return str(value)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(
            120deg,
            #0d3040 0%,
            #176b87 55%,
            #1d849f 100%
        );
        box-shadow: 0 14px 35px rgba(13, 48, 64, 0.18);
        margin-bottom: 1.2rem;
    }

    .hero-label {
        color: #ffd29c;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        font-size: 3rem;
        margin: 0.35rem 0 0.5rem 0;
    }

    .hero p {
        color: #e5f4f8;
        font-size: 1.1rem;
        max-width: 800px;
        margin-bottom: 0;
    }

    [data-testid="stMetric"] {
        background: rgba(23, 107, 135, 0.08);
        border: 1px solid rgba(23, 107, 135, 0.20);
        border-radius: 14px;
        padding: 1rem;
    }

    .context-box {
        padding: 1rem 1.2rem;
        border-left: 5px solid #da7b29;
        border-radius: 8px;
        background: rgba(218, 123, 41, 0.10);
        margin-bottom: 1.2rem;
    }

    .footer {
        color: #6b7280;
        text-align: center;
        font-size: 0.85rem;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Interactive machine-learning audit</div>
        <h1>CardioModel Shift</h1>
        <p>
            Explore how heart-disease classification models behave
            when transferred between the Cleveland and Hungarian
            patient cohorts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational demonstration using public historical data. "
    "This application is not a diagnostic system, medical device "
    "or personal risk calculator."
)

demo_patients = load_demo_patients()

st.sidebar.header("Model explorer")

training_cohort = st.sidebar.selectbox(
    "Training cohort",
    ["Cleveland", "Hungary"],
)

evaluation_cohort = (
    "Hungary"
    if training_cohort == "Cleveland"
    else "Cleveland"
)

model_name = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Random Forest"],
)

available_examples = demo_patients.loc[
    demo_patients["cohort"] == evaluation_cohort
].copy()

example_id = st.sidebar.selectbox(
    f"Example patient from {evaluation_cohort}",
    available_examples["example_id"].tolist(),
)

st.sidebar.caption(
    "The selected patient comes from the opposite cohort "
    "to the model's training population."
)

selected_rows = available_examples.loc[
    available_examples["example_id"] == example_id
]

patient_row = selected_rows.iloc[0]
patient_features = selected_rows[MODEL_FEATURES].copy()

model_file = MODEL_FILES[(model_name, training_cohort)]
selected_model = load_saved_model(model_file)

probability = float(
    selected_model.predict_proba(patient_features)[0, 1]
)

predicted_label = int(probability >= 0.50)
observed_label = int(patient_row["heart_disease"])

result_name = RESULT_NAMES[
    (observed_label, predicted_label)
]

explorer_tab, results_tab, methods_tab = st.tabs(
    [
        "Explore a prediction",
        "Project results",
        "Methods and limitations",
    ]
)

with explorer_tab:
    st.subheader("Cross-cohort patient example")

    st.markdown(
        f"""
        <div class="context-box">
            This <strong>{evaluation_cohort}</strong> example is
            being evaluated by a <strong>{model_name}</strong>
            model trained on the
            <strong>{training_cohort}</strong> cohort.
        </div>
        """,
        unsafe_allow_html=True,
    )

    patient_column, output_column = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with patient_column:
        st.markdown(f"#### Patient example `{example_id}`")

        patient_display = pd.DataFrame(
            {
                "Feature": [
                    FEATURE_NAMES[feature]
                    for feature in MODEL_FEATURES
                ],
                "Recorded value": [
                    format_feature_value(
                        feature,
                        patient_row[feature],
                    )
                    for feature in MODEL_FEATURES
                ],
            }
        )

        st.dataframe(
            patient_display,
            hide_index=True,
            use_container_width=True,
        )

    with output_column:
        st.markdown("#### Model output")

        probability_column, class_column = st.columns(2)

        with probability_column:
            st.metric(
                "Predicted probability",
                f"{probability:.1%}",
            )

        with class_column:
            st.metric(
                "Predicted class",
                (
                    "Disease recorded"
                    if predicted_label == 1
                    else "No disease recorded"
                ),
            )

        st.progress(
            probability,
            text="Probability of the recorded disease label",
        )

        st.caption(
            "The displayed class uses a fixed threshold of 0.50."
        )

        observed_text = (
            "Disease recorded"
            if observed_label == 1
            else "No disease recorded"
        )

        st.metric(
            "Observed dataset label",
            observed_text,
        )

        if predicted_label == observed_label:
            st.success(
                f"{result_name}: the classification matches "
                "the recorded dataset label."
            )
        else:
            st.warning(
                f"{result_name}: the classification does not "
                "match the recorded dataset label."
            )

    st.divider()
    st.markdown("#### Compare both model families")

    comparison_rows = []

    for comparison_name in [
        "Logistic Regression",
        "Random Forest",
    ]:
        comparison_file = MODEL_FILES[
            (comparison_name, training_cohort)
        ]

        comparison_model = load_saved_model(
            comparison_file
        )

        comparison_probability = float(
            comparison_model.predict_proba(
                patient_features
            )[0, 1]
        )

        comparison_rows.append(
            {
                "Model": comparison_name,
                "Predicted probability":
                    comparison_probability,
            }
        )

    comparison_data = pd.DataFrame(comparison_rows)

    chart_column, table_column = st.columns(
        [1.2, 0.8],
        gap="large",
    )

    with chart_column:
        st.bar_chart(
            comparison_data.set_index("Model")
        )

    with table_column:
        comparison_display = comparison_data.copy()

        comparison_display["Predicted probability"] = (
            comparison_display["Predicted probability"]
            .map(lambda value: f"{value:.1%}")
        )

        st.dataframe(
            comparison_display,
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "Different probabilities do not automatically make one "
        "model correct. The project also evaluates calibration "
        "and classification errors."
    )

with results_tab:
    st.subheader("What the audit found")

    metric_one, metric_two, metric_three, metric_four = (
        st.columns(4)
    )

    with metric_one:
        st.metric("Patient records", "597")

    with metric_two:
        st.metric("Patient cohorts", "2")

    with metric_three:
        st.metric("Model families", "2")

    with metric_four:
        st.metric("Repeated splits", "30")

    st.markdown(
        """
        - Cleveland-trained models transferred relatively well
          to Hungary.
        - Hungary-trained models showed a more consistent decline
          when evaluated in Cleveland.
        - Random Forest generally achieved stronger external
          ROC-AUC, but model choice depended on sensitivity and
          specificity tradeoffs.
        - Accuracy alone concealed changes in discrimination
          and probability calibration.
        """
    )

    st.image(
        FIGURE_DIR / "model_performance_heatmaps.png",
        caption=(
            "Model performance across training and "
            "evaluation cohorts."
        ),
        use_container_width=True,
    )

    st.image(
        FIGURE_DIR / "repeated_split_roc_auc_change.png",
        caption=(
            "External minus within-cohort ROC-AUC across "
            "30 repeated splits."
        ),
        use_container_width=True,
    )

    with st.expander("View calibration analysis"):
        st.image(
            FIGURE_DIR / "cross_cohort_calibration.png",
            use_container_width=True,
        )

    with st.expander("View external feature importance"):
        st.image(
            FIGURE_DIR
            / "external_permutation_importance.png",
            use_container_width=True,
        )

with methods_tab:
    st.subheader("How the project works")

    st.markdown(
        """
        **Research question**

        Does a heart-disease classification model remain reliable
        when evaluated on a patient cohort different from the one
        used for training?

        **Workflow**

        1. Audit missingness, outcome coding and population shift.
        2. Harmonise ten features available in both cohorts.
        3. Split the source cohort into 75% training and 25% testing.
        4. Fit all preprocessing using training patients only.
        5. Train Logistic Regression and Random Forest.
        6. Evaluate the same model on an external cohort.
        7. Repeat the experiment across 30 stratified splits.
        8. Assess calibration, errors and permutation importance.

        **Metrics**

        ROC-AUC, PR-AUC, accuracy, sensitivity, specificity
        and Brier score.
        """
    )

    st.markdown("#### Important limitations")

    st.markdown(
        """
        - The dataset is small and historical.
        - The outcome represents recorded disease presence,
          not future cardiovascular risk.
        - Cohorts differ in missingness and outcome detail.
        - The threshold was not selected for clinical use.
        - Feature importance does not imply causation.
        - The app must not be used for medical decisions.
        """
    )

    st.markdown("#### Sources and complete analysis")

    source_column, data_column = st.columns(2)

    with source_column:
        st.link_button(
            "View the GitHub repository",
            GITHUB_URL,
            use_container_width=True,
        )

    with data_column:
        st.link_button(
            "View the UCI dataset",
            (
                "https://archive.ics.uci.edu/"
                "dataset/45/heart+disease"
            ),
            use_container_width=True,
        )

st.markdown(
    """
    <div class="footer">
        CardioModel Shift · Educational machine-learning audit ·
        Built with Python, scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)