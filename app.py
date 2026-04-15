# =============================================================================
# LOAN DEFAULT RISK PREDICTION - Flask Backend
# =============================================================================
# WHAT IS THIS FILE?
#   This is the "brain" of our web application. It handles:
#     1. Training the AI model with synthetic (fake-but-realistic) loan data
#     2. Serving the website to the browser
#     3. Receiving form data and returning a risk prediction
#
# KEY CONCEPT — WHAT IS LOAN DEFAULT?
#   A loan "default" means a borrower FAILS to repay their loan.
#   Banks lose money when this happens. Our AI predicts HOW LIKELY a person
#   is to default BEFORE the bank gives them money.
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os                        # File path operations
import json                      # Convert Python dicts to JSON (for API responses)
import pickle                    # Save/load the trained ML model to disk
import numpy as np               # Numerical operations, array handling
import pandas as pd              # Data manipulation (like Excel in Python)

# Flask = lightweight Python web framework
from flask import Flask, render_template, request, jsonify

# Database
from database import db_manager

# scikit-learn = the ML library we use
# RandomForestClassifier = our main prediction model (explained below)
# train_test_split = splits data into training vs. testing portions
# StandardScaler = normalizes features so no single feature dominates
# classification_report = shows model accuracy metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# scipy.special.expit = the sigmoid/logistic function: turns any number into 0-1 range
from scipy.special import expit  # We use this when generating synthetic training data

# ── FLASK APP SETUP ───────────────────────────────────────────────────────────
app = Flask(__name__)

# Path where we save the trained model so we don't retrain on every restart
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_model.pkl")


# =============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# =============================================================================
# WHY SYNTHETIC DATA?
#   Real bank data is confidential. For this learning project, we GENERATE
#   fake data that follows REAL statistical patterns found in credit research.
#
# WHAT FEATURES DO WE USE?
#   Features = the input variables our AI uses to make a prediction.
#
#   1. credit_score          : FICO score (300-850). Higher = more trustworthy.
#                              Below 580 = poor, 580-669 = fair, 670-739 = good,
#                              740-799 = very good, 800+ = exceptional.
#
#   2. annual_income         : Yearly salary in USD. Higher income = easier to repay.
#
#   3. loan_amount           : How much money they want to borrow.
#
#   4. dti_ratio             : Debt-To-Income ratio (in %).
#                              = (monthly debt payments / monthly gross income) × 100
#                              Example: Pay $500/mo in debt, earn $2000/mo → DTI = 25%
#                              Banks usually reject if DTI > 43%.
#
#   5. employment_length     : Years at current/recent job. Longer = more stable income.
#
#   6. num_delinquencies     : Times they've been 30+ days late on payments. More = riskier.
#
#   7. credit_history_years  : How long they've HAD credit. Longer = more track record.
#
#   8. num_open_accounts     : Current active credit cards, loans, etc.
#
#   9. home_ownership        : 0=Renting, 1=Has mortgage, 2=Owns outright.
#                              Owners tend to be more financially stable.
#
#  10. loan_purpose           : Why they need the loan (affects repayment motivation).
#                              0=personal, 1=education, 2=medical, 3=home, 4=business, 5=auto
#
#  11. num_inquiries          : Recent "hard pulls" on credit report (each inquiry = applied
#                               for new credit recently). More = desperation signal.
# =============================================================================

def generate_synthetic_data(n_samples=15000, random_state=42):
    """
    Generate realistic synthetic loan data with known risk patterns.

    We use a LOGISTIC (sigmoid) function to model default probability:
    P(default) = sigmoid(linear_combination_of_features)

    This mimics how real credit scoring models work — each factor adds or
    subtracts from a "risk score", then we convert to a probability 0-1.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    # ── GENERATE EACH FEATURE ──────────────────────────────────────────────
    # Each feature follows a realistic distribution (not just random noise)

    # Credit score: most people are in the 600-750 range (bell curve around 700)
    credit_score = np.clip(rng.normal(loc=680, scale=80, size=n), 300, 850)

    # Annual income: right-skewed (most earn $30k-$80k, few earn $200k+)
    # We use lognormal for right-skewed distributions
    annual_income = np.clip(np.exp(rng.normal(loc=10.9, scale=0.5, size=n)), 18000, 250000)

    # Loan amount: generally $5k-$40k range
    loan_amount = np.clip(rng.lognormal(mean=9.5, sigma=0.8, size=n), 1000, 50000)

    # DTI ratio: 0-50%, most people are around 20-35%
    dti_ratio = np.clip(rng.normal(loc=25, scale=12, size=n), 0, 65)

    # Employment length: 0-30 years, skewed toward shorter (job changes are common)
    employment_length = np.clip(rng.exponential(scale=5, size=n), 0, 35)

    # Delinquencies: most have 0-1, rare to have many (Poisson distribution for counts)
    num_delinquencies = np.clip(rng.poisson(lam=0.8, size=n), 0, 15)

    # Credit history: ranges from 0 (new) to 40 years, normally distributed
    credit_history_years = np.clip(rng.normal(loc=12, scale=8, size=n), 0, 45)

    # Open accounts: most have 4-10 active credit lines
    num_open_accounts = np.clip(rng.poisson(lam=6, size=n), 1, 25).astype(int)

    # Home ownership: categorical — 40% rent, 35% mortgage, 25% own
    home_ownership = rng.choice([0, 1, 2], size=n, p=[0.40, 0.35, 0.25])

    # Loan purpose: categorical — mix of reasons
    loan_purpose = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                               p=[0.30, 0.15, 0.10, 0.20, 0.15, 0.10])

    # Recent credit inquiries: 0-8, most have 0-2
    num_inquiries = np.clip(rng.poisson(lam=1.5, size=n), 0, 10).astype(int)

    # ── CALCULATE DEFAULT PROBABILITY (THE GROUND TRUTH) ──────────────────
    # This is the "logit" (log-odds). Negative values → low risk, positive → high risk.
    # Each term represents one feature's contribution to risk.
    #
    # READING THE FORMULA:
    #   - Negative coefficient (-) means the feature REDUCES risk
    #   - Positive coefficient (+) means the feature INCREASES risk
    #   - Larger absolute value = stronger effect

    logit = (
        -2.5                                                    # baseline (real default rate ~15-20%)
        + (-0.010) * (credit_score - 650)                      # ↑ score → ↓ risk (strong)
        + 0.055 * dti_ratio                                     # ↑ DTI → ↑ risk
        + (-0.000008) * annual_income                           # ↑ income → ↓ risk
        + 0.45 * num_delinquencies                              # ↑ late payments → ↑ risk (VERY strong)
        + (-0.12) * np.minimum(employment_length, 12)           # stable job → ↓ risk (caps at 12yr)
        + 0.35 * (loan_amount / (annual_income + 1))            # high loan-to-income → ↑ risk
        + (-0.025) * credit_history_years                       # longer history → ↓ risk
        + (-0.30) * home_ownership                              # owner → ↓ risk
        + 0.15 * num_inquiries                                  # many inquiries → ↑ risk
        + rng.normal(0, 0.4, n)                                 # random noise (real world is noisy!)
    )

    # Convert logit to probability using sigmoid function: P = 1 / (1 + e^(-logit))
    default_prob = expit(logit)

    # Convert probability to binary outcome (1=defaulted, 0=paid back)
    # We add a small random element so the boundary isn't perfectly sharp
    default = (default_prob > rng.uniform(0.3, 0.7, n)).astype(int)

    # ── ASSEMBLE INTO A DATAFRAME ──────────────────────────────────────────
    df = pd.DataFrame({
        'credit_score':         credit_score,
        'annual_income':        annual_income,
        'loan_amount':          loan_amount,
        'dti_ratio':            dti_ratio,
        'employment_length':    employment_length,
        'num_delinquencies':    num_delinquencies,
        'credit_history_years': credit_history_years,
        'num_open_accounts':    num_open_accounts,
        'home_ownership':       home_ownership,
        'loan_purpose':         loan_purpose,
        'num_inquiries':        num_inquiries,
        'default':              default          # TARGET: what we're predicting
    })

    print(f"[DATA] Generated {n_samples} samples | Default rate: {default.mean():.1%}")
    return df


# =============================================================================
# SECTION 2: MODEL TRAINING
# =============================================================================
# WHAT IS A RANDOM FOREST?
#   A Random Forest is an ENSEMBLE of many Decision Trees.
#
#   Decision Tree = a flowchart of if/else rules learned from data.
#     Example: "IF credit_score < 580 AND dti > 40 → THEN high risk"
#
#   Random Forest = build 200 such trees, each on a random subset of data
#   and features. Final prediction = MAJORITY VOTE of all trees.
#
# WHY RANDOM FOREST FOR CREDIT SCORING?
#   ✓ Handles missing values well (important for new customers)
#   ✓ Naturally produces probabilities (not just yes/no)
#   ✓ Feature importance = tells us WHICH factors matter most
#   ✓ Robust to outliers (rare extreme incomes, etc.)
#   ✓ No need to scale features (trees use thresholds, not distances)
#   ✓ Industry standard for credit risk (alongside XGBoost)
# =============================================================================

FEATURE_NAMES = [
    'credit_score', 'annual_income', 'loan_amount', 'dti_ratio',
    'employment_length', 'num_delinquencies', 'credit_history_years',
    'num_open_accounts', 'home_ownership', 'loan_purpose', 'num_inquiries'
]

# Human-readable labels for the frontend
FEATURE_LABELS = {
    'credit_score':          'Credit Score',
    'annual_income':         'Annual Income',
    'loan_amount':           'Loan Amount',
    'dti_ratio':             'Debt-to-Income Ratio',
    'employment_length':     'Employment Length',
    'num_delinquencies':     'Past Delinquencies',
    'credit_history_years':  'Credit History Length',
    'num_open_accounts':     'Open Accounts',
    'home_ownership':        'Home Ownership',
    'loan_purpose':          'Loan Purpose',
    'num_inquiries':         'Recent Inquiries'
}

def train_model():
    """
    Train the loan default prediction model and save it to disk.

    Returns: (model, scaler, feature_importances_dict)
    """
    print("[TRAINING] Generating synthetic loan data...")
    df = generate_synthetic_data(n_samples=15000)

    X = df[FEATURE_NAMES]  # Features (inputs)
    y = df['default']       # Target (what we predict: 1=default, 0=no default)

    # ── TRAIN/TEST SPLIT ───────────────────────────────────────────────────
    # CONCEPT: We can't test a model on data it was trained on — it would
    # memorize the answers. So we split: 80% for training, 20% for testing.
    # This gives an honest measure of how well it generalizes to new customers.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
        # stratify=y ensures both train/test have same proportion of defaults
    )
    print(f"[TRAINING] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── FEATURE SCALING ────────────────────────────────────────────────────
    # CONCEPT: StandardScaler transforms features to have mean=0, std=1.
    # Random Forests don't strictly need this, but it helps with model
    # calibration (converting raw scores to accurate probabilities).
    # CRITICAL: Fit ONLY on training data, then apply same transform to test.
    # If we fit on test data too, we'd "cheat" — using future information.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Only transform, don't refit!

    # ── RANDOM FOREST MODEL ────────────────────────────────────────────────
    base_model = RandomForestClassifier(
        n_estimators=200,         # Build 200 different decision trees
        max_depth=10,             # Max depth of each tree (prevents overfitting)
        min_samples_split=20,     # A node needs 20+ samples before splitting
        min_samples_leaf=10,      # Each leaf needs 10+ samples (smooths predictions)
        max_features='sqrt',      # Each tree sees sqrt(11) ≈ 3 random features
        class_weight='balanced',  # Adjust weights since defaults are rarer than non-defaults
        random_state=42,          # For reproducibility
        n_jobs=-1                 # Use all CPU cores for speed
    )

    # ── FIT BASE MODEL FIRST ───────────────────────────────────────────────
    # We fit the base RandomForest on the training data first.
    # This gives us (a) feature importances and (b) a model ready for calibration.
    print("[TRAINING] Fitting model (this takes ~30 seconds)...")
    base_model.fit(X_train_scaled, y_train)

    # ── FEATURE IMPORTANCES ────────────────────────────────────────────────
    # CONCEPT: Feature importance tells us HOW MUCH each feature contributed
    # to the model's decisions across all trees. Higher = more influential.
    # This is a key explainability tool for regulators and customers.
    # We read this from the BASE model before wrapping it in calibration.
    raw_importances = base_model.feature_importances_

    # ── PROBABILITY CALIBRATION ────────────────────────────────────────────
    # CONCEPT: Raw Random Forest probabilities can be over-confident (e.g.,
    # it might say 99% when the true probability is 70%). We use FrozenEstimator
    # to wrap the already-fitted base model so CalibratedClassifierCV knows
    # not to re-fit the trees — only fit the calibration mapping layer.
    # FrozenEstimator is the modern (sklearn 1.6+) way to do prefit calibration.
    try:
        from sklearn.frozen import FrozenEstimator
        model = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic')
    except ImportError:
        # Fallback for older sklearn versions
        model = CalibratedClassifierCV(base_model, cv='prefit', method='isotonic')
    model.fit(X_test_scaled, y_test)  # Fit calibration layer on held-out test set

    # ── EVALUATE THE MODEL ─────────────────────────────────────────────────
    # AUC-ROC Score: measures how well the model RANKS risks.
    # 0.5 = random guessing, 1.0 = perfect, anything above 0.75 is good for credit risk.
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"[TRAINING] AUC-ROC Score: {auc:.4f} (0.5=random, 1.0=perfect)")
    importance_dict = {
        FEATURE_LABELS[feat]: float(imp)
        for feat, imp in zip(FEATURE_NAMES, raw_importances)
    }
    # Sort by importance (highest first)
    importance_dict = dict(sorted(importance_dict.items(),
                                   key=lambda x: x[1], reverse=True))

    print("[TRAINING] Feature Importances:")
    for feat, imp in importance_dict.items():
        bar = "█" * int(imp * 100)
        print(f"  {feat:<30} {bar} ({imp:.3f})")

    # ── SAVE MODEL TO DISK ─────────────────────────────────────────────────
    # pickle = Python's way to serialize (save) a Python object to a file.
    # We save BOTH the model AND the scaler — we need both to make predictions.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler,
                     'importances': importance_dict}, f)
    print(f"[TRAINING] Model saved to {MODEL_PATH}")

    return model, scaler, importance_dict


def load_or_train_model():
    """
    Load an existing trained model, or train a new one if none exists.
    This is called ONCE when the Flask app starts.
    """
    if os.path.exists(MODEL_PATH):
        print("[STARTUP] Loading existing model...")
        with open(MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)
        return saved['model'], saved['scaler'], saved['importances']
    else:
        print("[STARTUP] No saved model found — training from scratch...")
        return train_model()


# ── LOAD MODEL ON STARTUP ──────────────────────────────────────────────────
# This runs ONCE when Flask starts. The model stays in memory for all requests.
model, scaler, feature_importances = load_or_train_model()


# =============================================================================
# SECTION 3: RISK PREDICTION LOGIC
# =============================================================================
# CONCEPT: THE THREE RISK TIERS
#
#   The model outputs a probability P(default) between 0.0 and 1.0.
#   We map this to three business decisions:
#
#   HIGH RISK   (P > 0.55) → REJECT loan.
#               The probability of losing money is too high.
#               Used threshold: 55% because defaults cost banks 3-5x more
#               than the interest income they'd gain.
#
#   MODERATE    (0.30 < P ≤ 0.55) → CONDITIONAL approval.
#   RISK        May approve with: higher interest rate, lower loan amount,
#               collateral, co-signer, or manual review.
#
#   LOW RISK    (P ≤ 0.30) → APPROVE loan.
#               Low enough probability that expected return > expected loss.
#
# FOR NEW CUSTOMERS (no credit history — the "cold start problem"):
#   We apply MORE CONSERVATIVE thresholds because we have LESS information.
#   HIGH RISK   : P > 0.40  (stricter — we know less about them)
#   MODERATE    : 0.22 < P ≤ 0.40
#   LOW RISK    : P ≤ 0.22
# =============================================================================

# Thresholds for EXISTING customers (have credit history)
THRESHOLDS_EXISTING = {'high': 0.55, 'moderate': 0.30}

# Thresholds for NEW customers (no credit history)
# More conservative because we have less information to be confident
THRESHOLDS_NEW      = {'high': 0.40, 'moderate': 0.22}


def classify_risk(prob, is_new_customer):
    """
    Map a default probability to a risk category and decision.

    Args:
        prob (float): Probability of default (0.0 to 1.0)
        is_new_customer (bool): Whether this is a customer with no credit history

    Returns:
        dict with risk_level, decision, color, and description
    """
    thresholds = THRESHOLDS_NEW if is_new_customer else THRESHOLDS_EXISTING

    if prob > thresholds['high']:
        return {
            'level': 'HIGH',
            'decision': 'LOAN DENIED',
            'color': '#ef233c',
            'icon': '✗',
            'description': 'The applicant poses too high a risk of default. Loan should not be granted.',
            'recommendation': 'Consider improving credit score, reducing existing debt, or applying again after 6-12 months.'
        }
    elif prob > thresholds['moderate']:
        return {
            'level': 'MODERATE',
            'decision': 'MANUAL REVIEW',
            'color': '#fb8500',
            'icon': '⚠',
            'description': 'The applicant presents borderline risk. Human review is recommended.',
            'recommendation': 'May approve with higher interest rate, reduced loan amount, collateral requirement, or co-signer.'
        }
    else:
        return {
            'level': 'LOW',
            'decision': 'LOAN APPROVED',
            'color': '#06d6a0',
            'icon': '✓',
            'description': 'The applicant demonstrates low risk of default. Loan can be safely granted.',
            'recommendation': 'Standard loan terms applicable. Monitor repayment as per normal procedures.'
        }


def generate_factor_analysis(features_dict, prob):
    """
    Generate human-readable explanations of the KEY factors driving the prediction.

    This is the "explainability" layer — regulators (and customers!) need to know
    WHY a loan was rejected. This is required by law in many countries (ECOA, GDPR).

    Returns a list of factor dicts with: factor, impact (positive/negative), explanation
    """
    factors = []

    # Credit Score Analysis
    cs = features_dict.get('credit_score', 650)
    if cs >= 750:
        factors.append({'factor': 'Credit Score', 'impact': 'positive',
                        'value': f'{cs:.0f}', 'detail': 'Exceptional credit score — demonstrates excellent repayment history'})
    elif cs >= 670:
        factors.append({'factor': 'Credit Score', 'impact': 'positive',
                        'value': f'{cs:.0f}', 'detail': 'Good credit score — indicates reliable payment behavior'})
    elif cs >= 580:
        factors.append({'factor': 'Credit Score', 'impact': 'neutral',
                        'value': f'{cs:.0f}', 'detail': 'Fair credit score — some concerns about past payment history'})
    else:
        factors.append({'factor': 'Credit Score', 'impact': 'negative',
                        'value': f'{cs:.0f}', 'detail': 'Poor credit score — significant history of late/missed payments'})

    # DTI Analysis
    dti = features_dict.get('dti_ratio', 25)
    if dti <= 20:
        factors.append({'factor': 'Debt-to-Income', 'impact': 'positive',
                        'value': f'{dti:.1f}%', 'detail': f'Low DTI of {dti:.1f}% — well within healthy range (≤20%)'})
    elif dti <= 36:
        factors.append({'factor': 'Debt-to-Income', 'impact': 'neutral',
                        'value': f'{dti:.1f}%', 'detail': f'Manageable DTI of {dti:.1f}% — standard range, monitor closely'})
    elif dti <= 43:
        factors.append({'factor': 'Debt-to-Income', 'impact': 'negative',
                        'value': f'{dti:.1f}%', 'detail': f'High DTI of {dti:.1f}% — most banks cap at 43%'})
    else:
        factors.append({'factor': 'Debt-to-Income', 'impact': 'negative',
                        'value': f'{dti:.1f}%', 'detail': f'Very high DTI of {dti:.1f}% — exceeds standard lending limits'})

    # Delinquencies Analysis
    delinq = features_dict.get('num_delinquencies', 0)
    if delinq == 0:
        factors.append({'factor': 'Payment History', 'impact': 'positive',
                        'value': '0 late payments', 'detail': 'Clean payment record — no history of missed/late payments'})
    elif delinq <= 1:
        factors.append({'factor': 'Payment History', 'impact': 'neutral',
                        'value': f'{delinq} late payment(s)', 'detail': 'Minor delinquency history — isolated incident'})
    else:
        factors.append({'factor': 'Payment History', 'impact': 'negative',
                        'value': f'{delinq} late payments', 'detail': f'Multiple delinquencies — pattern of payment difficulties detected'})

    # Employment Stability
    emp = features_dict.get('employment_length', 0)
    if emp >= 5:
        factors.append({'factor': 'Employment', 'impact': 'positive',
                        'value': f'{emp:.1f} years', 'detail': 'Stable long-term employment — reliable income stream'})
    elif emp >= 2:
        factors.append({'factor': 'Employment', 'impact': 'neutral',
                        'value': f'{emp:.1f} years', 'detail': 'Moderate employment tenure — some income stability'})
    else:
        factors.append({'factor': 'Employment', 'impact': 'negative',
                        'value': f'{emp:.1f} years', 'detail': 'Short employment history — income stability uncertain'})

    # Loan-to-Income Ratio
    lti = features_dict.get('loan_amount', 10000) / max(features_dict.get('annual_income', 50000), 1)
    if lti <= 0.15:
        factors.append({'factor': 'Loan-to-Income', 'impact': 'positive',
                        'value': f'{lti:.1%}', 'detail': 'Loan is small relative to income — easily manageable repayments'})
    elif lti <= 0.35:
        factors.append({'factor': 'Loan-to-Income', 'impact': 'neutral',
                        'value': f'{lti:.1%}', 'detail': 'Moderate loan relative to income — review repayment schedule'})
    else:
        factors.append({'factor': 'Loan-to-Income', 'impact': 'negative',
                        'value': f'{lti:.1%}', 'detail': 'Large loan relative to income — repayment burden is high'})

    # Credit History
    ch = features_dict.get('credit_history_years', 0)
    if ch >= 10:
        factors.append({'factor': 'Credit History', 'impact': 'positive',
                        'value': f'{ch:.0f} years', 'detail': 'Long credit history provides strong evidence of repayment behavior'})
    elif ch >= 3:
        factors.append({'factor': 'Credit History', 'impact': 'neutral',
                        'value': f'{ch:.0f} years', 'detail': 'Moderate credit history — some track record established'})
    else:
        factors.append({'factor': 'Credit History', 'impact': 'negative',
                        'value': f'{ch:.0f} years', 'detail': 'Limited credit history — insufficient data to assess reliably'})

    return factors


def build_radar_scores(features_dict):
    """
    Compute normalized 0-100 scores for the radar chart categories shown in the UI.

    The radar chart gives a visual snapshot of the applicant's financial profile
    across 6 key dimensions. Each dimension is scored 0-100 (100 = safest).
    """
    scores = {}

    # 1. Credit Health (based on credit score)
    cs = features_dict.get('credit_score', 650)
    scores['Credit Health'] = int(np.interp(cs, [300, 850], [0, 100]))

    # 2. Income Stability (based on employment length + income)
    emp = features_dict.get('employment_length', 0)
    income = features_dict.get('annual_income', 50000)
    emp_score = np.interp(min(emp, 15), [0, 15], [0, 60])
    income_score = np.interp(min(income, 150000), [18000, 150000], [0, 40])
    scores['Income Stability'] = int(emp_score + income_score)

    # 3. Debt Management (DTI-based)
    dti = features_dict.get('dti_ratio', 25)
    scores['Debt Management'] = int(np.interp(dti, [0, 60], [100, 0]))

    # 4. Repayment History (delinquencies-based)
    delinq = features_dict.get('num_delinquencies', 0)
    scores['Repayment History'] = int(np.interp(min(delinq, 8), [0, 8], [100, 0]))

    # 5. Credit Experience (credit history years)
    ch = features_dict.get('credit_history_years', 0)
    scores['Credit Experience'] = int(np.interp(min(ch, 25), [0, 25], [0, 100]))

    # 6. Loan Safety (loan-to-income ratio)
    lti = features_dict.get('loan_amount', 10000) / max(features_dict.get('annual_income', 50000), 1)
    scores['Loan Safety'] = int(np.interp(lti, [0, 1.5], [100, 0]))

    return scores


# =============================================================================
# SECTION 4: NEW CUSTOMER HANDLING (The "Cold Start Problem")
# =============================================================================
# WHAT IS THE COLD START PROBLEM?
#   When a bank sees a BRAND NEW customer (no credit history, no past records),
#   they face the "cold start" problem: how to assess risk with minimal data?
#
# OUR THREE-LAYER APPROACH:
#
#   LAYER 1 — Imputation with Conservative Defaults
#     We fill missing features with CONSERVATIVE estimates (not the average):
#     - Credit score: we use 580 (fair, not average 680)
#     - Delinquencies: 0 (no history = clean slate, but also no positive record)
#     - Credit history: 0 years (genuinely unknown)
#     - Open accounts: 1 (minimal)
#     - Inquiries: 0
#
#   LAYER 2 — Tighter Decision Thresholds
#     We shift the approval threshold lower (40% vs 55% for existing customers).
#     Meaning: we need MORE confidence of safety before approving.
#     This is standard "precautionary principle" in risk management.
#
#   LAYER 3 — UI Warning + Alternative Evidence Requests
#     We flag it as "limited data" and show what additional documents
#     the bank should request (utility bills, rent history, bank statements)
#     to improve the assessment.
# =============================================================================

NEW_CUSTOMER_IMPUTE = {
    'credit_score':         580,  # Conservative: "fair" instead of average "good"
    'num_delinquencies':    0,    # No history = clean slate
    'credit_history_years': 0,    # Genuinely none
    'num_open_accounts':    1,    # Minimal
    'num_inquiries':        0     # No prior applications
}


def prepare_features(form_data, is_new_customer):
    """
    Convert raw form input into the feature vector expected by the model.

    For new customers, fills missing credit-history features with conservative defaults.
    Returns a numpy array ready for model.predict_proba()
    """
    features = {}

    # These features are always available (every customer can provide them)
    features['annual_income']      = float(form_data.get('annual_income', 50000))
    features['loan_amount']        = float(form_data.get('loan_amount', 10000))
    features['dti_ratio']          = float(form_data.get('dti_ratio', 25))
    features['employment_length']  = float(form_data.get('employment_length', 3))
    features['home_ownership']     = int(form_data.get('home_ownership', 0))
    features['loan_purpose']       = int(form_data.get('loan_purpose', 0))

    if is_new_customer:
        # ── NEW CUSTOMER: impute missing credit features ───────────────────
        # We have no credit bureau data, so we fill in conservative estimates.
        for feat, default_val in NEW_CUSTOMER_IMPUTE.items():
            features[feat] = default_val
    else:
        # ── EXISTING CUSTOMER: use the actual values they provided ─────────
        features['credit_score']          = float(form_data.get('credit_score', 650))
        features['num_delinquencies']     = int(form_data.get('num_delinquencies', 0))
        features['credit_history_years']  = float(form_data.get('credit_history_years', 5))
        features['num_open_accounts']     = int(form_data.get('num_open_accounts', 4))
        features['num_inquiries']         = int(form_data.get('num_inquiries', 1))

    # Build the feature array as a DataFrame (preserves column names, silences sklearn warning)
    feature_vector = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    feature_vector_scaled = scaler.transform(feature_vector)

    return feature_vector_scaled, features


# =============================================================================
# SECTION 5: FLASK ROUTES (URL Endpoints)
# =============================================================================
# Flask "routes" map URLs to Python functions.
# @app.route('/') means: when someone visits the homepage, run this function.
# =============================================================================

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Receives form data as JSON, runs the ML model, returns a detailed
    risk assessment as JSON.

    HTTP POST because we're sending data (not just fetching a page).
    """
    try:
        # Parse the JSON body sent by the frontend JavaScript
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        is_new_customer = data.get('is_new_customer', False)

        # Prepare features (handle missing values for new customers)
        feature_vector_scaled, features_dict = prepare_features(data, is_new_customer)

        # ── GET MODEL PREDICTION ───────────────────────────────────────────
        # predict_proba returns [[P(no default), P(default)]]
        # We take the second column: P(default)
        proba = model.predict_proba(feature_vector_scaled)[0]
        default_prob = float(proba[1])   # Probability of default (0.0 to 1.0)
        safe_prob    = float(proba[0])   # Probability of safe repayment

        # Classify into risk tier
        risk_result = classify_risk(default_prob, is_new_customer)

        # Build factor analysis (human-readable explanation)
        factors = generate_factor_analysis(features_dict, default_prob)

        # Build radar chart scores
        radar_scores = build_radar_scores(features_dict)

        # ── CONFIDENCE INDICATOR ───────────────────────────────────────────
        # How far from the boundary are we?
        # If prob is 0.1 or 0.9 → high confidence
        # If prob is 0.45 → low confidence (borderline)
        thresholds = THRESHOLDS_NEW if is_new_customer else THRESHOLDS_EXISTING
        dist_from_high = abs(default_prob - thresholds['high'])
        dist_from_mod  = abs(default_prob - thresholds['moderate'])
        min_dist = min(dist_from_high, dist_from_mod)
        confidence = min(100, int((min_dist / 0.30) * 100))  # 0-100%

        # Build the complete response object
        response = {
            'default_probability':    round(default_prob * 100, 1),   # e.g., 34.5 (as %)
            'safe_probability':       round(safe_prob * 100, 1),
            'risk_level':             risk_result['level'],
            'decision':               risk_result['decision'],
            'color':                  risk_result['color'],
            'icon':                   risk_result['icon'],
            'description':            risk_result['description'],
            'recommendation':         risk_result['recommendation'],
            'confidence':             confidence,
            'is_new_customer':        is_new_customer,
            'factors':                factors,
            'radar_scores':           radar_scores,
            'feature_importances':    feature_importances,
        }

        # ── SAVE ASSESSMENT TO DATABASE ────────────────────────────────────
        # Store the applicant profile and assessment result for future reference
        applicant_email = data.get('applicant_email', 'unknown@example.com')
        if db_manager.is_connected():
            db_manager.save_assessment(applicant_email, data, response)
            response['saved_to_db'] = True
        else:
            response['saved_to_db'] = False
            print("[WARNING] Database not connected — assessment not saved")

        return jsonify(response)

    except Exception as e:
        # Log the error server-side and return a generic error to the client
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/assessment-history/<applicant_email>')
def get_assessment_history(applicant_email):
    """
    Retrieve assessment history for a specific applicant.

    Args:
        applicant_email (str): The applicant's email

    Returns:
        JSON array of past assessments, sorted by most recent first
    """
    if not db_manager.is_connected():
        return jsonify({'error': 'Database not connected'}), 503

    history = db_manager.get_assessment_history(applicant_email, limit=20)
    return jsonify({
        'applicant_email': applicant_email,
        'assessment_count': len(history),
        'assessments': history
    })


@app.route('/database-stats')
def get_database_stats():
    """
    Return overall database statistics for analytics/dashboard.

    Returns:
        JSON with total applicants, assessments, risk distribution, etc.
    """
    if not db_manager.is_connected():
        return jsonify({'error': 'Database not connected', 'connected': False}), 503

    stats = db_manager.get_statistics()
    stats['connected'] = True
    return jsonify(stats)


@app.route('/feature-importances')
def get_feature_importances():
    """
    Return the model's feature importances (for the info panel in the UI).
    These show which factors the model considers most important overall.
    """
    return jsonify(feature_importances)


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model with fresh synthetic data.
    Useful for demonstrating that results can vary with different training sets.
    """
    global model, scaler, feature_importances
    try:
        # Delete old model file to force retraining
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        model, scaler, feature_importances = train_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# SECTION 6: APP ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  LOAN DEFAULT RISK PREDICTOR")
    print("  Starting Flask server...")
    print("  Open: http://127.0.0.1:5001")
    print("=" * 60)
    # debug=True enables auto-reload on code changes (development mode only!)
    app.run(debug=True, host='0.0.0.0', port=5001)
