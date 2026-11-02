# olist-analytics-prediction-app

**Author:** Harsha Vardhan  
**Application:** Streamlit business dashboards and machine-learning inference  
**Deployment:** Streamlit Community Cloud with GitHub-backed source  
**Live application:** Add the deployed `https://…streamlit.app` URL here. Access credentials are provided privately by the project owner.

This project brings historical e-commerce analysis and customer-satisfaction predictions into one application. Users sign in, explore nine dashboard pages, score delivered orders, download predictions, and inspect the model's final holdout metrics.

The project workflow includes cleaned SQL exports, Python feature engineering and modeling, a separately authored Power BI report, and deployment through Streamlit. The deployed dashboard pages are built directly in Python: Power BI embedding is not required.

> **Prediction scope:** Estimate low satisfaction after an order is delivered and before its review is received. The current model is not a purchase-time or pre-delivery predictor.

## Contents

- [Business problem](#business-problem)
- [Application features](#application-features)
- [Data and relationships](#data-and-relationships)
- [Dashboard pages and metric definitions](#dashboard-pages-and-metric-definitions)
- [Modeling methodology](#modeling-methodology)
- [Prediction input contract](#prediction-input-contract)
- [Repository structure](#repository-structure)
- [Local installation](#local-installation)
- [Authentication and sharing](#authentication-and-sharing)
- [Deployment](#deployment)
- [Validation](#validation)
- [Maintenance and troubleshooting](#maintenance-and-troubleshooting)
- [Limitations and future work](#limitations-and-future-work)
- [Attribution and licensing](#attribution-and-licensing)

## Business problem

A marketplace needs to understand order activity, customer geography, payment patterns, delivery outcomes, product demand, and customer reviews. It also needs a way to prioritize delivered orders that may result in poor customer feedback.

The dashboards support questions such as:

- How do order volumes and merchandise value change over time?
- Which customer states, sellers, products, and categories contribute the most activity?
- What proportion of eligible delivered orders arrive late?
- How do ratings differ between late and on-time orders?
- Which delivered orders should customer-service staff review first?

Predictions can support human follow-up and service recovery. The project does not claim that using a score causes improved retention, higher revenue, or fewer complaints. Those outcomes would require a separate intervention study.

## Application features

| Section | Function |
|---|---|
| Login | User ID and password gate before application content |
| Overview | Business purpose and prediction scope |
| Dashboards | Nine pages with KPIs, charts, filters, geographic visualization, and filtered-order export |
| Predictions | Score supplied example orders or an uploaded engineered-feature CSV |
| Model Performance | Display saved final test metrics and compare example predictions with notebook reference outputs |
| Logout | Clear session state and return to the login screen |

The application serves an exported model; it does not train models during user requests. It reads historical CSV snapshots rather than connecting directly to a live SQL database.

### Technology stack

| Area | Tools |
|---|---|
| Source preparation | SQL-based cleaning and CSV exports |
| Analysis and engineering | Python, pandas, NumPy, Jupyter |
| Model preprocessing and evaluation | scikit-learn |
| Gradient boosting candidate | XGBoost |
| Artifact serialization | joblib |
| Training visualizations | Matplotlib, Seaborn |
| Interactive application | Streamlit and its native chart/map elements |
| Separate BI deliverable | Power BI Desktop (`.pbix`) |
| Source management | Git, GitHub, GitHub Desktop |
| Hosting | Streamlit Community Cloud |

Use `requirements.txt` for deployed dependency versions and `assets/training_environment.json` for the recorded training environment. Notebook-only libraries may need separate installation when reproducing training.

## Data and relationships

The project uses cleaned tables based on the Brazilian Olist e-commerce dataset. Exact row counts depend on the cleaning/export version; inspect the supplied CSVs rather than assuming original dataset counts.

| Table | Grain / key | Role |
|---|---|---|
| Customers | One row per `customer_id` | Customer identifiers, city, state, ZIP prefix |
| Orders | One row per `order_id` | Status and purchase/delivery timestamps |
| Order items | One row per (`order_id`, `order_item_id`) | Product, seller, merchandise price, freight |
| Payments | One row per (`order_id`, `payment_sequential`) | Payment method, amount, installments |
| Reviews | Review records; multiple per order are allowed | Scores and supervised target construction |
| Products | One row per `product_id` | Category and physical/listing attributes |
| Sellers | One row per `seller_id` | Seller geography |
| Geolocation | Potentially multiple observations per ZIP prefix | Approximate coordinates |
| Category translations | One row per `product_category_name` | Portuguese-to-English category mapping |

`customer_id` joins a customer row to an order. `customer_unique_id` identifies repeat customers across orders. These identifiers are not interchangeable.

### Join discipline

Orders have one-to-many relationships with items, payments, and reviews. Joining those raw child tables together can multiply rows and inflate financial totals. The dashboard module first aggregates each child table to order level, then joins with explicit relationship validation.

Items join to products and category translations through many-to-one relationships. Dashboard ZIP coordinates are reduced to one median latitude/longitude pair per ZIP prefix before mapping.

### Dashboard CSV contract

Place one current cleaned export for each table in `data/`. Names can differ: the loader identifies tables from required column sets. Headers are stripped and lowercased. CSVs should be comma-delimited and UTF-8 encoded.

| Table | Required dashboard columns |
|---|---|
| Customers | `customer_id`, `customer_unique_id`, `customer_state`, `customer_city`, `customer_zip_code_prefix` |
| Orders | `order_id`, `customer_id`, `order_status`, `order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| Order items | `order_id`, `order_item_id`, `product_id`, `seller_id`, `price`, `freight_value` |
| Payments | `order_id`, `payment_sequential`, `payment_type`, `payment_value`, `payment_installments` |
| Reviews | `order_id`, `review_score` |
| Products | `product_id`, `product_category_name`, `product_weight_g`, `product_photos_qty` |
| Sellers | `seller_id`, `seller_state`, `seller_city` |
| Geolocation | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng` |
| Category translations | `product_category_name`, `product_category_name_english` |

Extra columns are accepted. The training notebook requires additional timestamps and product attributes beyond this minimum dashboard contract. Do not trim training inputs to this table's column list.

Missing tables, duplicate candidate exports, and missing/duplicate required keys cause readable validation errors. Invalid numeric/date values are coerced to missing values; this does not replace a full upstream data-quality audit. ZIP prefixes are loaded as strings to preserve leading zeros. Missing foreign-key matches are not silently invented or repaired.

## Dashboard pages and metric definitions

| Page | Main content |
|---|---|
| Customers | Unique customers, selected-period repeat share, item GMV, mean order rating, state/city activity, monthly orders |
| Orders | Order count, item GMV, mean delivery duration, late share, monthly activity, status distribution, state GMV |
| Order Items | Item rows, products sold, item GMV, freight, category/seller value, order-size distribution |
| Payments | Record count, payment total, mean record value, installments, method value, monthly payments by purchase month |
| Geolocation | Selected customer ZIP coverage, median-coordinate map, state order counts |
| Reviews | Valid review records, reviewed orders, average order rating, low-satisfaction share, star distribution, late/on-time comparison |
| Products | Products sold, full catalog count, product attributes, product/category merchandise value |
| Sellers | Active sellers, full seller count, seller geography, seller item GMV |
| Categories | Sold categories, products sold, translation coverage, untranslated products, category GMV and order counts |

### Filters

- Purchase-date range.
- Customer state.
- Order status.
- Payment method.
- Product category.

Empty multi-selects mean all available values. Different filter types are combined with AND; values selected within one multi-select use OR.

Payment and category filters select **whole orders**. If an order contains a selected category, all its items remain in the resulting analysis. Similarly, matching one payment method retains all payment records of that order. This is order selection, not item-only or payment-only filtering.

The date range uses purchase timestamps and excludes undated orders. Product and seller metrics generally describe entities active in selected orders; explicitly labelled full-catalog counts ignore those filters.

### Important definitions

| Metric | Definition |
|---|---|
| Item GMV | Sum of item `price`; excludes freight and is not net revenue |
| Payment total | Sum of payment values; separate from item GMV and may include freight |
| Repeat customer share | Selected customers with more than one selected order / selected unique customers |
| Delivery duration | Actual customer delivery timestamp minus purchase timestamp, in fractional days; negative durations excluded |
| Late delivery share | Late eligible delivered orders / delivered orders with both actual and estimated delivery dates |
| Calendar-day lateness | Actual delivery date is later than the estimated date; later time on the same day is not late |
| Mean order rating | Average of order-level means of valid review scores |
| Low-satisfaction share | Reviewed selected orders with mean valid review score below 3 / reviewed selected orders |
| Translation coverage | Sold products with a nonmissing English category translation / distinct sold products |
| Map coverage | Selected distinct customer ZIP prefixes with valid coordinates / selected distinct customer ZIP prefixes |

Amounts are shown in Brazilian reais (R$); no exchange-rate conversion is performed. Multiple-category order counts are not additive across categories. Map points represent ZIP prefixes, not individual customers.

The Streamlit pages recreate the main business analyses; they are not an exact conversion of every original Power BI visual, DAX measure, or layout.

## Modeling methodology

### Target and eligible population

Valid review scores range from 1 to 5. Where an order has multiple valid reviews, the notebook averages them before assigning the label:

```text
low_satisfaction = 1 if mean valid order review score < 3
low_satisfaction = 0 otherwise
```

For a single review, scores 1–2 are low satisfaction and scores 3–5 are satisfactory. With multiple reviews, the mean determines the label.

Training uses reviewed, delivered orders satisfying the notebook's timestamp eligibility checks. Unreviewed orders do not acquire a default satisfactory label. Reviews are outcomes, not model inputs.

### Feature engineering

The notebook produces 32 predictors: 28 numeric and four categorical.

| Family | Features |
|---|---|
| Purchase timing | `purchase_month`, `purchase_dayofweek`, `purchase_hour` |
| Delivery and processing | `approval_hours`, `delivery_days`, `estimated_delivery_days`, `delay_days`, `is_late`, `shipping_processing_days` |
| Order composition | `item_count`, `unique_products`, `seller_count`, `unique_categories` |
| Merchandise and freight | `total_price`, `mean_item_price`, `max_item_price`, `total_freight`, `mean_freight`, `freight_ratio` |
| Product attributes | `mean_weight`, `max_weight`, `mean_volume`, `mean_photos`, `mean_description_length` |
| Payments | `total_payment`, `payment_records`, `max_installments`, `payment_method_count` |
| Categorical | `customer_state`, `customer_zip_region`, `dominant_category`, `primary_payment_type` |

Key calculations:

- Durations in days use elapsed seconds divided by 86,400; approval duration uses hours.
- `delay_days` is actual delivery timestamp minus estimated timestamp and can be negative for early delivery.
- `is_late` compares normalized calendar dates; it is distinct from a fractional timestamp difference.
- `freight_ratio` is total freight divided by total merchandise price; zero denominators become missing.
- Product volume is length × height × width in cubic centimetres.
- `customer_zip_region` is the first two digits of a five-character ZIP prefix.
- Dominant category is the highest-merchandise-value category, with alphabetical tie-breaking.
- Primary payment method is selected from the largest-value individual payment record, with payment sequence used in ordering. It is not aggregated spend by method.

Raw order/customer identifiers, review scores, review text, and review-derived fields are excluded from predictors. No geolocation-coordinate join is required for this baseline model.

### Preprocessing

Preprocessing is part of the fitted scikit-learn pipeline:

- Numeric median imputation, missing indicators, and standard scaling.
- Categorical most-frequent imputation and one-hot encoding.
- Infrequent-category grouping with `min_frequency=50` and `handle_unknown='infrequent_if_exist'`.
- Infinite numeric values converted to missing before inference/training.

During model search, preprocessing is fitted within each training fold rather than across the complete dataset.

### Chronological evaluation

The notebook sorts orders by delivery timestamp and order ID, then uses approximately:

| Partition | Share | Purpose |
|---|---:|---|
| Training | 70% | Fit baseline models and perform training-only search |
| Validation | 15% | Compare candidates and select classification thresholds |
| Test | 15% | Final holdout evaluation |

Candidate thresholds range from 0.10 to 0.90 in steps of 0.01. Selection prioritizes validation macro F1. The chosen configuration is refitted on training + validation, retaining the validation-selected threshold, then evaluated on the final test period.

This is an order-level chronological assessment, not a customer-disjoint evaluation. Review-label availability delays are not fully simulated. Refitting can also change score distributions; threshold stability merits further validation.

### Candidate models

The notebook defines a most-frequent dummy baseline and these classifiers:

- Logistic Regression.
- Gaussian Naive Bayes.
- Decision Tree.
- Random Forest.
- Extra Trees.
- XGBoost.

KNN and Bagging are optional and disabled by default. The supplied notebook configures randomized hyperparameter search for Logistic Regression, Random Forest, Extra Trees, and XGBoost, using three-fold `TimeSeriesSplit` and eight search iterations per model when enabled. The saved artifact identifies the actual selected model; do not assume XGBoost or any other candidate won.

### Evaluation and actual results

The primary selection metric is macro F1, which gives both classes equal weight. Other outputs include accuracy, macro precision/recall, low-rating F1/recall, ROC AUC, and average precision. The notebook labels `average_precision_score` as “PR AUC”; technically it is average precision rather than trapezoidal PR-curve area.

**The exported metrics are authoritative. Populate the following from the deployed artifact and `assets/final_test_metrics.csv`; no numerical results are asserted here.**

| Result | Value |
|---|---|
| Selected model and version | Add from model artifact |
| Saved threshold | Add from model artifact |
| Final test macro F1 | Add from final test metrics |
| Final test low-rating recall | Add from final test metrics |
| Final test ROC AUC | Add from final test metrics |
| Final test average precision | Add from final test metrics |

Training-sample permutation importance is available from the notebook. It describes predictive associations, can be affected by correlated features, and is not causal evidence or an independent performance estimate.

## Prediction input contract

The app accepts a CSV containing all saved feature columns, with one row per delivered order. Use the downloadable example CSV as the schema reference.

1. Build features using exactly the notebook's calculations and category-selection rules.
2. Preserve ZIP-region strings, including leading zeros.
3. Provide numeric values in the original units; do not pre-scale or manually one-hot encode.
4. Retain all required columns. Missing numeric values can be imputed by the pipeline, but absent columns are rejected.
5. Do not include review-derived predictors or fill missing outcomes with invented labels.
6. Upload through Predictions and generate scores.
7. Download results for further review.

Raw source tables such as `Orders_Cleaned.csv` cannot be uploaded directly for prediction. The deployed dashboard joins are for analysis and are not a substitute for the model's full feature engineering.

For new unreviewed orders, feature assembly must not inner-join to review labels. The notebook's reviewed-only training-table assembly must be adapted for inference so it does not discard the orders to be scored.

Prediction output adds:

| Column | Meaning |
|---|---|
| `probability_low_satisfaction` | Model's class-1 score from `predict_proba` |
| `predicted_low_satisfaction` | 1 when score meets/exceeds the saved threshold; otherwise 0 |
| `prediction_label` | Human-readable class label |

Scores are not guaranteed outcomes and have not necessarily been calibrated as probabilities. Example input rows are historical demonstration features, not live incoming orders.

## Repository structure

```text
.
├── README.md
├── app.py
├── olist_dashboards.py
├── olist_logo.png
├── requirements.txt
├── .gitignore
├── assets/
│   ├── olist_customer_satisfaction_model.joblib
│   ├── example_orders.csv
│   ├── expected_example_predictions.csv
│   ├── final_test_metrics.csv
│   └── training_environment.json
├── data/
│   └── [nine cleaned CSV exports]
└── .streamlit/
    └── secrets.toml  # local only; excluded from Git
```

Training notebooks, SQL scripts, original PBIX files, and full modeling outputs can be maintained as separate deliverables; they are not required to execute the exported app.

### Main artifact responsibilities

- `app.py`: login, navigation, inference, output downloads, metrics, and consistency check.
- `olist_dashboards.py`: schema detection, validated joins, filtering, dashboard charts, order-summary download.
- Model `.joblib`: fitted preprocessing + classifier, threshold, feature schema, target/prediction metadata.
- `training_environment.json`: versions used for export/reproduction.
- `expected_example_predictions.csv`: reference for checking inference consistency.

## Local installation

### Requirements

Use a Python version compatible with `assets/training_environment.json` and the pinned dependencies. Obtain the cleaned data and trusted model artifact from the project owner. A source-only clone without those assets is not sufficient to run the complete application.

### Windows PowerShell

From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual environment's Python directly avoids changing PowerShell activation settings. If dependency installation fails because the Python version differs from training, fix the environment before changing model-library versions.

Create `.streamlit/secrets.toml` as described below, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal process running during local use. Stop with Ctrl+C.

For an existing working environment:

```powershell
cd "$env:USERPROFILE\Documents\Olist_Prediction_App"
python -m streamlit run app.py
```

A laptop restart does not require retraining when the exported files are present.

## Authentication and sharing

### Local secrets

```toml
[app_login]
username = "REPLACE_WITH_YOUR_USER_ID"
password = "REPLACE_WITH_A_STRONG_UNIQUE_PASSWORD"

[powerbi]
embed_url = ""
```

The Power BI section is unused by native dashboards and can remain empty. These values are placeholders, not working credentials.

The app checks credentials before displaying application content, records authentication in session state, and clears session state on logout. The current implementation is one shared account rather than a multi-user account service. It does not provide password recovery, MFA, audit trails, or durable rate limiting. Refreshes or session resets may require signing in again.

Never commit secrets. Recommended `.gitignore`:

```gitignore
.streamlit/secrets.toml
.venv/
__pycache__/
*.pyc
app_backup.py
```

Ignoring a file does not remove an already committed secret. If credentials were exposed, rotate them and address repository history as needed.

### Viewer access

Send the hosted app URL and user ID to intended viewers; communicate the password separately. Do not share GitHub, Microsoft, or hosting-admin account passwords.

Repository privacy, hosting visibility, and the custom login gate are separate controls. A private repository protects source-file visibility; it does not itself decide who can view a hosted app. For restricted audiences, hosting-level viewer invitations provide an additional access layer.

If the hosting app is publicly reachable, the custom login screen remains the first application view. Treat this as a basic demonstration login; for sensitive organizational workloads use stronger identity-based authentication and appropriate hosting restrictions.

## Deployment

1. Keep the working app files, `assets/`, and `data/` in the GitHub repository; use a private repository for restricted source data.
2. Do not upload `secrets.toml`, local environments, or backup scripts.
3. Push the repository changes. GitHub Desktop's Commit saves locally; Push origin uploads remotely.
4. In [Streamlit Community Cloud](https://share.streamlit.io), create an app from the repository.
5. Select the correct branch and `app.py` as the entry point.
6. In Advanced settings, choose a supported Python version matching training's major/minor version.
7. Paste the secrets configuration into the deployment Secrets field, using actual private credentials.
8. Deploy and inspect startup logs if installation or execution fails.
9. Configure viewer access under Sharing according to the intended audience.
10. Run the acceptance checks below against the hosted HTTPS URL.

The hosted app runs independently of the owner's laptop. Users must receive the deployed `https://…streamlit.app` URL, not `localhost:8501`.

The project's geolocation export was approximately 33.11 MiB during upload and was transferred through GitHub Desktop rather than the browser. GitHub documents separate browser and regular Git file limits; consult its current documentation for future larger artifacts. Avoid repeatedly committing large data revisions when external versioned storage is more appropriate.

### Deployment references

- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Streamlit sharing controls](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [GitHub file upload guidance](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## Validation

Before sharing any new release:

| Check | Expected outcome |
|---|---|
| Fresh/private browser session | Login shown before data or downloads |
| Incorrect credentials | Access denied |
| Correct credentials | Navigation appears |
| Nine dashboard pages | Each renders without schema/merge errors |
| Filter behavior | Counts and charts respond according to documented scope |
| Known totals | Item GMV, payments and order counts reconcile with source calculations |
| Example predictions | Scores and labels generated |
| Uploaded example CSV | Matches built-in example scoring |
| Consistency check | App matches notebook reference scores and labels |
| Downloads | Result CSV and filtered order summary download correctly |
| Logout | Returns to login; new session cannot view content without authenticating |
| Hosted mobile test | Works through the deployed URL without the local server |

The dashboard add-on's development checks used synthetic multi-item, multi-payment, and multi-review orders to test aggregation, review averaging, calendar-day lateness, leading-zero ZIP handling, and the nine render paths with a Streamlit API stub. This was not browser-based visual QA or full production-data verification. The project owner subsequently reported successful local and deployed operation; no independent load/security audit is claimed.

The prediction consistency check detects many environment/export mismatches. It is not evidence of generalization beyond the separately evaluated historical test set.

## Maintenance and troubleshooting

### Updates

- **Code/layout:** edit the cloned repository, test locally, commit, and push.
- **Historical data:** replace the current CSV exports as a consistent snapshot; check totals and keys before publishing.
- **Retraining:** run the notebook offline, then export model, feature lists, threshold, examples, expected predictions, metrics, and environment together.
- **Credentials:** update deployment Secrets, restart the application to end existing in-memory login sessions, and update local secrets separately.
- **Large workloads:** reduce unnecessary columns or use versioned analytical storage after measuring memory/performance; preserve metric definitions and ZIP semantics.

Do not replace only the model while keeping examples or metrics from another run. Do not retrain in response to every uploaded CSV. Do not change thresholds after inspecting test results without creating a new evaluation protocol.

### Common errors

| Symptom | What to check |
|---|---|
| Login not configured | Correct `[app_login]` keys in deployment Secrets or local secrets file |
| Login works locally but not online | Deployment secrets are configured separately |
| Missing CSV tables/columns | All nine exports, original column names, no duplicate old exports |
| Duplicate key validation | Investigate source exports; do not blindly drop meaningful rows |
| Missing model file | Exact path `assets/olist_customer_satisfaction_model.joblib` and matching filename case |
| Missing prediction columns | Upload engineered features, not raw source CSVs |
| Invalid numeric value | Correct text entered into numeric input columns |
| Model loading/version error | Match training Python and model-related package versions |
| Consistency check fails | Compare model, feature schema, threshold, examples and environment as one release |
| Browser file upload rejected | Use GitHub Desktop/Git/LFS as appropriate to current file limits |
| Hosted out-of-memory/restarts | Inspect logs and memory use; CSV size is smaller than in-memory data size |
| Map blank | Valid coordinates, selected ZIPs, internet connectivity, browser network restrictions |
| Viewer sees hosting permission screen | Invite viewer or adjust hosting visibility separately from app credentials |

The geolocation source can contain many rows; cached raw tables, derived joins, session activity, and a loaded model all contribute to hosting memory. Successful single-user testing does not establish high-concurrency capacity.

## Limitations and future work

### Current limitations

- Historical observations may not represent current marketplace behavior.
- Reviewed orders can differ systematically from unreviewed orders.
- Multiple-review averaging is an explicit modeling choice.
- Delivery-time predictors make the model unsuitable for pre-delivery use.
- The same customer may appear across chronological partitions.
- Delayed review-label availability is not fully simulated in evaluation.
- Classification scores are not necessarily calibrated probabilities.
- Dashboard associations and feature importance do not establish causation.
- Native charts do not provide automatic Power BI DAX/layout parity.
- Data refresh is a manual snapshot update, not a real-time database pipeline.
- The shared login has no per-user authorization or individual revocation.

### Potential improvements (not currently implemented)

- Reusable, tested feature-engineering module for raw-table inference.
- Separate pre-delivery model using only information available at that point.
- Label-availability-aware temporal backtesting and customer-group evaluation.
- Calibration and threshold selection based on contact capacity and business costs.
- Historical customer/seller features computed strictly from past events.
- Identity-provider authentication, per-user permissions, and audit logging.
- Monitoring for schema drift, input drift, score drift, and eventual outcome quality.
- Automated snapshot refresh, data reconciliation and release checks.
- Measured improvements to caching, chart layout, accessibility and mobile use.

## Attribution and licensing

Underlying data source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Consult the source's current terms and attribution requirements before redistributing data or derivative assets.

This is an analytical portfolio/project application, not an official Olist service or endorsement. The Olist name and logo belong to their respective owners. No software license is granted by this README; add an explicit LICENSE file if code reuse is intended, separately from dataset and brand-asset terms.
