# 🏠 Sherlock Homes

**London Real Estate Development | Predictions & Recommendations**

Sherlock Homes is a data-driven Real Estate project focused on the London market. By combining digitalization with construction insights, we developed **Watson V1.0**, a machine learning model that predicts dwelling prices and acts as a strategic advisor to identify high-ROI renovation opportunities.

### 🔗 Project Links
*   **Web App:** [Sherlock Homes App](https://sherlockhomes.streamlit.app/)
*   **Tableau Dashboard:** [London Real Estate Analysis](https://public.tableau.com/app/profile/antonio.galv.o/viz/SH_17691167823370/SH-LondonPrices?publish=yes)
*   **Project Presentation:** [Canva Slides](https://sherlockhomes.my.canva.site/)

---

## 🏙️ Market Context
*   **Market Value:** The London residential market is valued at approximately £2.64 trillion.
*   **Stability:** After a correction in 2023 due to mortgage rate surges, the market has stabilized.
*   **Current State:** As of late 2025 data, the average property price is **£652,000**, with median prices around **£505,000**.

---

## 📊 Data & Methodology

### Data Sources
The dataset aggregates two primary sources (filtered for data > 2023):
1.  **London House Price Data (Kaggle):** Transactional data, rooms, tenure, energy rates.
2.  **London Building Stock Model (London Datastore):** Conservation area status, building age bands.

**Processing:**
*   Cleaned ~4.3M raw rows down to **~80,000 high-quality rows** for training.
*   **Feature Engineering:** Added neighbourhood median values (£/m²) and size buckets.
*   **Handling Missing Data:** Null room counts replaced by the median of similar properties; outliers (e.g., short leaseholds <18 years) removed.

---

## 🤖 Machine Learning: Watson V1.0

We experimented with several algorithms to optimize prediction accuracy. **LightGBM** was selected as the final model due to its superior performance and speed.

| Model | Algorithm | $R^2$ Score | MAPE | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ML-RF** | Random Forest | 79.5% | 25.20% | Tested |
| **ML-XG** | XGBoost | 88.7% | 16.96% | Tested |
| **ML-LGBM** | **LightGBM** | **89.2%** | **16.72%** | **Deployed (Watson)** |

**Model Specs:**
*   **Target:** Logarithmic transformation of Selling Price.
*   **Encoders:** One-Hot (Property Type, Tenure), Label (Energy), Target Encoding (Location).

---

## 📱 The Application

The Streamlit app (`watsonV1_0.py`) allows users to input property details and receive:
1.  **Valuation:** Estimated market value with a confidence interval (25%-75% variance).
2.  **Sherlock's Strategic Advisor:**
    *   **Feasibility Check:** Ensures recommended layouts (e.g., adding a bedroom) fit within the floor area.
    *   **Market Reality Check:** Verifies that the proposed layout actually exists among the nearest 50 neighbors.
    *   **ROI Logic:** Only recommends changes if the predicted value uplift is **>10%**.

---

## 📂 Repository Structure

*   **`1_Data/`**: Contains the processed dataset (`df_cleaned.csv`).
*   **`2_Notebooks/`**:
    *   `Sherlock_Homes-Analysis.ipynb`: Exploratory Data Analysis (EDA).
    *   `LBSM_datacleaning.ipynb`: Cleaning external building stock data.
    *   `Sherlock_Homes - ML-RF...`: Random Forest training.
    *   `Sherlock_Homes - ML-XG...`: XGBoost training.
    *   `Sherlock_Homes - ML-LGBM`: Final LightGBM model training.
    *   `Watson_LGBM.pkl`: The pickled model used by the app.
*   **`3_App/`**:
    *   `watsonV1_0.py`: The main application script.
    *   `requirements.txt`: List of dependencies.
    *   `SH2.png`: Assets/Images.

---

## 🛠️ Installation & Usage

To run the application locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/SherlockHomes.git
    cd SherlockHomes
    ```

2.  **Install dependencies:**
    *(Note: The requirements file is located inside the `3_App` folder)*
    ```bash
    pip install -r 3_App/requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run 3_App/watsonV1_0.py
    ```

---

### 👤 Author
**António Galvão**
