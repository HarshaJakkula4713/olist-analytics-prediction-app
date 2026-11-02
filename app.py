from pathlib import Path
import hmac

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"


# ---------- Login ----------
def require_login():
    if st.session_state.get("authenticated", False):
        return

    st.title("Olist Analytics — Login")
    st.write("Enter your user ID and password.")

    try:
        login_settings = st.secrets["app_login"]
        expected_user = str(login_settings["username"])
        expected_password = str(login_settings["password"])
    except (FileNotFoundError, KeyError):
        st.error("Add your login credentials to .streamlit/secrets.toml.")
        st.stop()

    with st.form("login", clear_on_submit=True):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        user_ok = hmac.compare_digest(
            user_id.encode("utf-8"),
            expected_user.encode("utf-8"),
        )
        password_ok = hmac.compare_digest(
            password.encode("utf-8"),
            expected_password.encode("utf-8"),
        )

        if user_ok and password_ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect user ID or password.")

    st.stop()


require_login()


def logout():
    for key in list(st.session_state):
        del st.session_state[key]


st.sidebar.title("Olist Analytics")
st.sidebar.button("Log out", on_click=logout)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Dashboards", "Predictions", "Model Performance"],
)


# ---------- Model utilities ----------
@st.cache_resource
def load_model():
    return joblib.load(
        ASSET_DIR / "olist_customer_satisfaction_model.joblib"
    )


def predict_orders(data, artifact):
    required = artifact["feature_columns"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    if data.empty:
        raise ValueError("Provide at least one order.")

    X = data[required].copy()

    for col in artifact["numeric_features"]:
        original = X[col]
        converted = pd.to_numeric(original, errors="coerce")

        invalid = (
            original.notna()
            & original.astype(str).str.strip().ne("")
            & converted.isna()
        )

        if invalid.any():
            raise ValueError(f"Invalid numeric values in '{col}'.")

        X[col] = converted.replace([np.inf, -np.inf], np.nan)

    for col in artifact["categorical_features"]:
        X[col] = X[col].astype("object").where(X[col].notna(), np.nan)

    model = artifact["model"]
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]

    result = data.copy()
    result["probability_low_satisfaction"] = probabilities
    result["predicted_low_satisfaction"] = (
        probabilities >= float(artifact["threshold"])
    ).astype(int)

    result["prediction_label"] = np.where(
        result["predicted_low_satisfaction"].eq(1),
        "Low satisfaction",
        "Satisfactory",
    )

    return result


# ---------- Overview ----------
if page == "Overview":
    st.title("Olist Analytics & Customer Satisfaction")

    st.write(
        "Explore business dashboards and predict which delivered "
        "orders may receive a low customer review."
    )

    st.info(
        "Prediction point: after delivery, before the review. "
        "Low satisfaction means an average valid review score below 3."
    )

    st.write(
        "Use the sidebar to open dashboards, generate predictions, "
        "or inspect the saved model performance."
    )
# ---------- Dashboards ----------


elif page == "Dashboards":
    from olist_dashboards import render_dashboards
    render_dashboards()

# ---------- Predictions ----------
elif page == "Predictions":
    st.title("Customer Satisfaction Predictions")

    artifact = load_model()

    st.write(f"Model: **{artifact['selected_model']}**")
    st.write(f"Decision threshold: **{artifact['threshold']:.3f}**")

    st.info(
        "Each input row must contain the same engineered order-level "
        "features used in training. Raw source CSVs cannot be scored directly."
    )

    examples = pd.read_csv(
        ASSET_DIR / "example_orders.csv",
        dtype={"customer_zip_region": str},
    )

    st.download_button(
        "Download example input CSV",
        examples.to_csv(index=False).encode("utf-8"),
        file_name="olist_example_inputs.csv",
        mime="text/csv",
    )

    mode = st.radio(
        "Input method",
        ["Use example orders", "Upload feature CSV"],
    )

    if mode == "Use example orders":
        input_data = examples
        st.dataframe(input_data, use_container_width=True)

    else:
        uploaded = st.file_uploader("Choose a CSV", type=["csv"])

        if uploaded is None:
            st.stop()

        try:
            input_data = pd.read_csv(
                uploaded,
                dtype={"customer_zip_region": str},
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not read CSV: {error}")
            st.stop()

        st.dataframe(input_data.head(), use_container_width=True)

    if st.button("Generate predictions", type="primary"):
        try:
            results = predict_orders(input_data, artifact)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        first, second = st.columns(2)
        first.metric("Orders scored", len(results))
        second.metric(
            "Flagged for low satisfaction",
            int(results["predicted_low_satisfaction"].sum()),
        )

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download predictions",
            results.to_csv(index=False).encode("utf-8"),
            file_name="olist_predictions.csv",
            mime="text/csv",
        )

        st.caption(
            "Scores are model estimates and are not guaranteed outcomes "
            "or necessarily calibrated probabilities."
        )


# ---------- Model performance ----------
elif page == "Model Performance":
    st.title("Final Holdout Performance")

    metrics = pd.read_csv(ASSET_DIR / "final_test_metrics.csv")
    st.dataframe(metrics, use_container_width=True)

    st.caption(
        "These metrics come from the notebook's final test evaluation."
    )

    if st.button("Check app predictions against notebook"):
        artifact = load_model()

        examples = pd.read_csv(
            ASSET_DIR / "example_orders.csv",
            dtype={"customer_zip_region": str},
        )
        expected = pd.read_csv(
            ASSET_DIR / "expected_example_predictions.csv"
        )
        actual = predict_orders(examples, artifact)

        scores_match = np.allclose(
            actual["probability_low_satisfaction"],
            expected["probability_low_satisfaction"],
            rtol=1e-6,
            atol=1e-8,
        )
        labels_match = np.array_equal(
            actual["predicted_low_satisfaction"],
            expected["predicted_low_satisfaction"],
        )

        if scores_match and labels_match:
            st.success("App predictions match the notebook.")
        else:
            st.error(
                "Predictions differ. Check exported files and package versions."
            )