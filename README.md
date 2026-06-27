# 🏠 House Price Prediction

A machine learning web application that predicts residential property prices in Bengaluru, India. Users can input property details through an interactive web interface and receive an instant price estimate powered by a trained Linear Regression model.

---

## 📌 Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
- [Outlier Removal](#outlier-removal)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Reference](#api-reference)

---

## ✨ Features

- Predict house prices across **255 locations** in Bengaluru
- Input fields: Location, BHK, Number of Bathrooms, and Total Square Footage
- Asynchronous price prediction — no page reload required
- Clean, responsive Bootstrap 5 UI
- Pre-trained model served via a Flask REST endpoint

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML / Data | scikit-learn, pandas, numpy |
| Model Persistence | pickle |
| Backend | Flask |
| Frontend | HTML5, Bootstrap 5.3, Vanilla JS (XHR) |

---

## 📁 Project Structure

```
House-price-prediction-main/
│
├── House_price_pred.ipynb           # Full data cleaning, EDA, and model training notebook
├── Cleaned_data.csv                 # Processed dataset used by the Flask app at runtime
├── LinearRegressionModel_MINI_PRJCT.pkl  # Serialized trained pipeline (OHE + Scaler + LR)
├── main.py                          # Flask web application entry point
└── templates/
    └── index.html                   # Frontend UI (Jinja2 template)
```

---

## 📊 Dataset

**Source:** Bengaluru House Data (`Bengaluru_House_Data.csv`)

**Raw columns:** `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`

**After cleaning (`Cleaned_data.csv`):**

| Column | Type | Description |
|---|---|---|
| `location` | string | Property location in Bengaluru (255 unique areas) |
| `total_sqft` | float | Total area in square feet (300 – 30,400) |
| `bath` | float | Number of bathrooms (1 – 16) |
| `bhk` | int | Number of bedrooms / BHK (1 – 16) |
| `price` | float | Price in **lakhs (₹)** (10 – 2,200) |

**Final cleaned dataset size: 7,397 records × 5 features**

---

## 🤖 ML Pipeline

The trained model is a **scikit-learn Pipeline** with three sequential steps:

```
Input Features: [location, total_sqft, bath, bhk]
        │
        ▼
┌─────────────────────────────────────┐
│  ColumnTransformer                  │
│  OneHotEncoder → location column    │
│  passthrough   → numerical columns  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  StandardScaler                     │
│  Normalizes all features            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  LinearRegression                   │
│  Predicts price in lakhs            │
└─────────────────────────────────────┘
        │
        ▼
Output: Predicted Price × 1e5 → ₹ value
```

---

## 🔧 Data Preprocessing & Feature Engineering

The full workflow is documented in `House_price_pred.ipynb`. Here is a summary of each step:

**Step 1 — Drop irrelevant columns**  
`area_type`, `availability`, `society`, and `balcony` were removed as they are either too sparse or not useful for price prediction.

**Step 2 — Handle missing values**  
- `location` nulls → filled with mode (`'Sarjapur Road'`)  
- `size` nulls → filled with mode (`'2 BHK'`)  
- `bath` nulls → filled with median

**Step 3 — Feature engineering**  
- **`bhk`**: Extracted from the `size` string (e.g., `"3 BHK"` → `3`)  
- **`total_sqft`**: Range values like `"1200 - 1470"` were converted to a single float by averaging the bounds; non-numeric entries were dropped  
- **`price_per_sqft`**: Derived as `price × 100000 / total_sqft` — used solely for outlier detection, not as a model input

---

## 🚫 Outlier Removal

Four separate outlier removal strategies were applied:

**1. Rare location grouping**  
Locations with fewer than 10 data points were consolidated into a single `'others'` category to prevent one-hot encoding sparsity.

**2. Sqft-per-BHK minimum threshold**  
Properties where `total_sqft / bhk < 300` were removed as physically implausible.

**3. Price-per-sqft standard deviation filter** (`remove_outlier_sqft`)  
Within each location group, properties with `price_per_sqft` outside ±1 standard deviation of the group mean were removed.

**4. BHK price logic filter** (`bhk_outlier_remover`)  
Within the same location, if a higher-BHK property had a lower `price_per_sqft` than the mean for `bhk - 1` (and that group had >5 samples), it was removed as an illogical pricing anomaly.

After outlier removal, `size` and `price_per_sqft` were dropped before saving `Cleaned_data.csv`.

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| Train/Test Split | 80% / 20% (random_state=0) |
| **R² Score (test set)** | **0.844** |

The model explains ~84.4% of the variance in house prices on the held-out test set.

---

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.8+

**1. Clone the repository**
```bash
git clone https://github.com/yshmsra/House-price-prediction.git
cd House-price-prediction
```

**2. Install dependencies**
```bash
pip install flask pandas numpy scikit-learn==1.6.1
```

> ⚠️ **Important:** The `.pkl` model file was serialized with **scikit-learn 1.6.1**. Using a different version may cause a version mismatch warning or error when loading the model. Pin to `1.6.1` for guaranteed compatibility.

**3. Verify required files are present**
```
✅ Cleaned_data.csv
✅ LinearRegressionModel_MINI_PRJCT.pkl
✅ main.py
✅ templates/index.html
```

---

## 🚀 Usage

**Start the Flask development server:**
```bash
python main.py
```

The app will be available at: `http://127.0.0.1:5000`

**Using the web interface:**
1. Select a **location** from the dropdown (255 Bengaluru areas)
2. Enter the number of **BHK** (bedrooms)
3. Enter the number of **bathrooms**
4. Enter the property size in **square feet**
5. Click **Predict Price** — the estimated price appears instantly below the form (no page reload)

---

## 🔌 API Reference

### `GET /`
Returns the main prediction UI with all available locations pre-loaded in the dropdown.

---

### `POST /predict`

Accepts form data and returns the predicted price as a plain string.

**Request (form-encoded):**

| Field | Type | Description | Example |
|---|---|---|---|
| `location` | string | Property location | `"Whitefield"` |
| `bhk` | number | Number of bedrooms | `3` |
| `bath` | number | Number of bathrooms | `2` |
| `total_sqft` | number | Total area in sq. ft. | `1500` |

**Response:**

```
₹8542318.75
```

Plain text — the predicted price in Indian Rupees (₹), rounded to 2 decimal places.

**Example using `curl`:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "location=Whitefield&bhk=3&bath=2&total_sqft=1500"
```

---

## 📝 Notes

- Predictions are in **Indian Rupees (₹)**. The model internally predicts in lakhs; the Flask app multiplies by `1e5` before returning.
- The frontend uses **XHR (XMLHttpRequest)** for asynchronous prediction, so the result renders inline without a page reload.
- To retrain the model, run all cells in `House_price_pred.ipynb`. Ensure the raw `Bengaluru_House_Data.csv` is accessible at the path specified in Cell 8.
