from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="HelpDesk AI",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORIES = ["Technology", "Billing", "Account Access"]

DESCRIPTIONS = {
    "Technology": "Crashes, devices, software, and performance.",
    "Billing": "Charges, invoices, payments, and refunds.",
    "Account Access": "Passwords, sign-in, verification, and recovery.",
}

EXAMPLES = [
    "My screen freezes when I open the application.",
    "I was charged twice for my subscription.",
    "I forgot my password and cannot sign in.",
]


@st.cache_resource(show_spinner="Preparing the classroom model...")
def prepare_model(csv_contents):
    """Validate data, then train and evaluate a cached model."""
    # CSV bytes are the cache input. Changing the file rebuilds the model.
    try:
        data = pd.read_csv(
            BytesIO(csv_contents),
            encoding="utf-8-sig",
            dtype=str,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeDecodeError) as error:
        raise ValueError(
            "Cannot read tickets.csv. Save a valid CSV using UTF-8 "
            "encoding. Put quotation marks around requests containing commas."
        ) from error

    # Check the exact column names.
    if list(data.columns) != ["request", "category"]:
        raise ValueError(
            "The first CSV row must be exactly: request,category. "
            "The file must contain only these two columns."
        )

    if data.isna().any().any():
        raise ValueError(
            "The dataset contains missing values. "
            "Every row needs a request and a category."
        )

    # Remove accidental spaces before checking the values.
    data["request"] = data["request"].str.strip()
    data["category"] = data["category"].str.strip()

    if (data["request"] == "").any() or (data["category"] == "").any():
        raise ValueError(
            "The dataset contains blank text. "
            "Fill in each request and category."
        )

    invalid_categories = sorted(
        set(data["category"]) - set(CATEGORIES)
    )
    if invalid_categories:
        raise ValueError(
            "Unrecognized categories: "
            + ", ".join(invalid_categories)
            + ". Use Technology, Billing, or Account Access."
        )

    counts = (
        data["category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
    )

    if len(data) != 90 or not counts.eq(30).all():
        details = ", ".join(
            f"{category}: {int(counts[category])}"
            for category in CATEGORIES
        )
        raise ValueError(
            f"Expected 90 requests, with 30 per category. "
            f"Found {len(data)} requests. {details}."
        )

    normalized_requests = (
        data["request"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    if normalized_requests.duplicated().any():
        raise ValueError(
            "Duplicate requests were found. Replace repeated requests "
            "with different examples while keeping 30 per category."
        )

    # Keep test examples separate from model training.
    x_train, x_test, y_train, y_test = train_test_split(
        data["request"],
        data["category"],
        test_size=0.20,
        stratify=data["category"],
        random_state=42,
    )

    # The pipeline learns text features only from the training examples.
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    correct = int((test_predictions == y_test.to_numpy()).sum())

    return {
        "model": model,
        "counts": counts.to_dict(),
        "training_size": len(x_train),
        "test_size": len(x_test),
        "correct": correct,
        "accuracy": accuracy_score(y_test, test_predictions),
    }


# Find the CSV beside this Python file, locally or on Streamlit Cloud.
csv_path = Path(__file__).resolve().parent / "tickets.csv"

try:
    csv_contents = csv_path.read_bytes()
except FileNotFoundError:
    st.error(
        "tickets.csv was not found. Save it in the same folder as app.py. "
        "On GitHub, upload both files into the same repository folder."
    )
    st.stop()
except OSError:
    st.error(
        "The app cannot open tickets.csv. "
        "Check the file location and permission to read it."
    )
    st.stop()

try:
    results = prepare_model(csv_contents)
except ValueError as error:
    st.error(f"Dataset or model problem: {error}")
    st.info("Correct tickets.csv or the code, save it, and rerun the app.")
    st.stop()


# Sidebar describes the departments and the training dataset.
with st.sidebar:
    st.title("📨 HelpDesk AI")
    st.caption("A classroom machine learning project")
    st.divider()

    st.subheader("Departments")
    for category in CATEGORIES:
        st.markdown(f"**{category}**")
        st.caption(DESCRIPTIONS[category])

    st.divider()
    st.subheader("Dataset")
    st.caption("90 fictional requests • 30 per department")
    st.dataframe(
        pd.DataFrame({
            "Category": CATEGORIES,
            "Requests": [
                results["counts"][category]
                for category in CATEGORIES
            ],
        }),
        hide_index=True,
    )


# Main dashboard uses native components for light/dark theme support.
st.title("HelpDesk AI")
st.markdown("### Give every support request a starting point.")
st.write(
    "Enter a support request to predict which department should review it. "
    "This prototype suggests a category; it does not send or resolve tickets."
)

st.warning(
    "Classroom prototype trained on fictional examples. "
    "Predictions may be incorrect. "
    "Unrelated requests may still receive a category."
)

request_column, examples_column = st.columns([2, 1], gap="large")

with request_column:
    with st.form("request_form"):
        st.subheader("Classify a request")
        request_text = st.text_area(
            "What do you need help with?",
            placeholder="Describe one problem in a sentence or two...",
            height=160,
        )
        submitted = st.form_submit_button(
            "Classify Request",
            type="primary",
        )

    if submitted:
        cleaned_request = request_text.strip()

        if not cleaned_request:
            st.warning("Please enter a support request before classifying.")
        else:
            prediction = results["model"].predict(
                [cleaned_request]
            )[0]

            with st.container(border=True):
                st.caption("PREDICTED DEPARTMENT")
                st.subheader(prediction)
                st.write(DESCRIPTIONS[prediction])
                st.caption(
                    "Review this suggestion before routing the request. "
                    "No confidence score is shown."
                )

with examples_column:
    with st.container(border=True):
        st.subheader("Try an example")
        st.caption(
            "Use the copy button on a sample, "
            "then paste it into the request box."
        )

        for category, example in zip(CATEGORIES, EXAMPLES):
            st.markdown(f"**{category}**")
            st.code(example, language=None, wrap_lines=True)


st.divider()
st.subheader("Model Performance")
st.caption(
    "Results on held-out fictional classroom data. "
    "These figures are not confidence scores for individual requests."
)

accuracy_column, train_column, test_column = st.columns(3)

accuracy_column.metric(
    "Test accuracy",
    f"{results['accuracy']:.1%}",
)
train_column.metric(
    "Training examples",
    results["training_size"],
)
test_column.metric(
    "Test examples",
    results["test_size"],
)

st.write(
    f"The model correctly classified {results['correct']} "
    f"of {results['test_size']} held-out test requests."
)
st.caption(
    "A small synthetic test set does not establish real-world performance. "
    "Use additional new examples to investigate mistakes."
)

with st.expander("How It Works"):
    st.markdown(
        """
        1. **Data:** pandas reads the labeled requests from tickets.csv.
        2. **Training:** TF-IDF turns text into numerical features, and
           Logistic Regression learns patterns associated with departments.
        3. **Testing:** 20% of the examples are kept out of training and used
           to measure test accuracy.
        4. **Prediction:** the trained pipeline assigns a department to
           the text you enter.
        5. **Interface:** Streamlit provides the form, result, and dashboard.
        """
    )
    st.write(
        "The model must choose one of its three departments. "
        "It has no separate 'unrelated request' category."
    )

st.caption(
    "Entered requests are processed by this app. "
    "The code does not write them to a file or send them to an external AI service."
)