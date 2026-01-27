import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pgeocode
import os
full_dataset= pd.read_csv(r"..\1_Data\df_cleaned.csv")

# --- 1. SETUP ---
st.set_page_config(page_title="Sherlock Homes", page_icon="🏠")

# Initialize the UK Postcode database (Offline, fast)
nomi = pgeocode.Nominatim('gb') 

@st.cache_resource
def load_brain():
    # 1. Get the directory where THIS file (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Construct the path dynamically (works on Windows & Mac/Linux)
    # Go up one level ('..'), then down into '2_Notebooks', then find 'Watson_LGBM.pkl'
    watson_path = os.path.join(current_dir, '..', '2_Notebooks', 'Watson_LGBM.pkl')

    # 3. Load Watson
    with open(watson_path, 'rb') as f:
        data = pickle.load(f)
    return data

brain = load_brain()
model = brain['model']
neigh_map = brain['neighborhood_map']
options = brain['options']
feature_order = brain['features']

# --- 2. SIDEBAR ---
st.sidebar.header("Property Details")

# A. LOCATION 
postcode_input = st.sidebar.text_input("Postcode", "NW1 6XE")
outcode = postcode_input.split(' ')[0].upper().strip()

# B. SIZE & ROOMS (Same as before)
sqm = st.sidebar.number_input("Floor Area (sqm)", 30, 1000, 65)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 1)
living_rooms = st.sidebar.slider("Living Rooms", 0, 5, 1)

# C. DETAILS (Same as before)
prop_type = st.sidebar.selectbox("Property Type", options['propertyType'])
age_band = st.sidebar.selectbox("Construction Age", options['construction_age_band'])
tenure = st.sidebar.selectbox("Tenure", options['tenure'])
is_conservation = st.sidebar.checkbox("In Conservation Area?", value=False)
valid_ratings = [x for x in options['currentEnergyRating'] if str(x) != 'nan']
valid_ratings.sort() 
energy = st.sidebar.selectbox("Energy Rating", valid_ratings, index=len(valid_ratings)-3)

def get_sqm_bounds(user_sqm):
    bin_rules = {
        0:   [0.98, 1.02],   # < 37
        37:  [0.98, 1.035],  # 37-50
        50:  [0.97, 1.055],  # 50-60
        60:  [0.955, 1.08],  # 60-72
        72:  [0.935, 1.11],  # 72-90
        90:  [0.91, 1.145],  # 90-120
        120: [0.90, 1.185],  # 120-160
        160: [0.90, 1.20],   # 160-200
        200: [0.85, 1.25]    # > 200
    }

    # Find the correct bin
    sorted_keys = sorted(bin_rules.keys(), reverse=True)
    for key in sorted_keys:
        if user_sqm >= key:
            multipliers = bin_rules[key]
            return user_sqm * multipliers[0], user_sqm * multipliers[1]
    return user_sqm * 0.9, user_sqm * 1.1 # Fallback

# --- 3. LOGIC FUNCTION ---
def get_prediction_data(p_outcode, p_sqm, p_beds, p_baths, p_living, p_type, p_age, p_tenure, p_energy, p_cons):

    # 1. Get Neighborhood Value (From your training data)
    n_val = neigh_map.get(p_outcode, options['global_median_price'])

    # 2. Get Latitude & Longitude (EXTERNAL LIBRARY)
    # This works for ANY UK postcode, even ones you didn't train on
    location = nomi.query_postal_code(p_outcode)

    # Check if pgeocode found it. If not (NaN), default to London Center
    if pd.isna(location.latitude):
        lat = 51.5074
        lon = -0.1278
    else:
        lat = location.latitude
        lon = location.longitude

    # 3. Create DataFrame
    data = pd.DataFrame({
        'neighborhood_value': [n_val],
        'bathrooms': [p_baths],
        'bedrooms': [p_beds],
        'floorAreaSqM': [p_sqm],
        'livingRooms': [p_living],
        'in_conservation_area': [1 if p_cons else 0],
        'latitude': [lat],
        'longitude': [lon],
        'propertyType': [p_type],
        'tenure': [p_tenure],
        'construction_age_band': [p_age],
        'currentEnergyRating': [p_energy]
    })

    cat_cols = ['propertyType', 'tenure', 'construction_age_band', 'currentEnergyRating']
    for c in cat_cols:
        data[c] = data[c].astype('category')

    return data[feature_order]

def sherlock_strategic_advisor(user_row, full_dataset, watson_model):
    user_lat = user_row['latitude'].iloc[0]
    user_lon = user_row['longitude'].iloc[0]
    user_sqm = user_row['floorAreaSqM'].iloc[0]

    # Current Value (Baseline)
    current_price = np.expm1(watson_model.predict(user_row)[0])
    current_layout = (user_row['bedrooms'].iloc[0], 
                      user_row['bathrooms'].iloc[0], 
                      user_row['livingRooms'].iloc[0])

    # 1st & 2nd: Calculate Bounds and find 50 Nearest Peers
    lower, upper = get_sqm_bounds(user_sqm)
    area_filtered = full_dataset[(full_dataset['floorAreaSqM'] >= lower) & 
                                 (full_dataset['floorAreaSqM'] <= upper)].copy()

    # Vectorized Haversine Distance
    area_filtered['dist'] = np.sqrt((area_filtered['latitude'] - user_lat)**2 + 
                                    (area_filtered['longitude'] - user_lon)**2)
    nearest_50 = area_filtered.sort_values('dist').head(50)

    # 3rd: Physical Feasibility & Prediction
    possible_layouts = []
    for b in range(1, 6):
        for ba in range(1, 4):
            for lr in range(1, 3):
                # Your physical feasibility formula
                if (b * 9) + (ba * 4) + (lr * 12) + 10 > user_sqm:
                    continue

                # Predict Price for this layout
                sim_row = user_row.copy()
                sim_row['bedrooms'], sim_row['bathrooms'], sim_row['livingRooms'] = b, ba, lr
                pred_price = np.expm1(watson_model.predict(sim_row)[0])

                # 4th: Comparison with Peers (Frequency)
                freq = len(nearest_50[(nearest_50['bedrooms'] == b) & 
                                      (nearest_50['bathrooms'] == ba) & 
                                      (nearest_50['livingRooms'] == lr)])

                # Only keep if price is > 5% higher than current property
                if pred_price >= current_price * 1.05:
                    possible_layouts.append({
                        'layout': (b, ba, lr),
                        'price': pred_price,
                        'freq': freq,
                        'is_high_freq': freq >= 6
                    })

    if not possible_layouts:
        return []

    res_df = pd.DataFrame(possible_layouts).sort_values('price', ascending=False)

    # Logic: If Best Price matches High Frequency -> 1 Recommendation
    # Else -> 2 Recommendations
    best_layout = res_df.iloc[0]
    high_freq_layouts = res_df[res_df['is_high_freq'] == True]

    if best_layout['is_high_freq'] or high_freq_layouts.empty:
        return [best_layout]
    else:
        # Return both: The high-value outlier and the top "Safe/Frequent" choice
        return [best_layout, high_freq_layouts.iloc[0]]



# --- 4. MAIN INTERFACE ---
st.title("Sherlock Homes")

if st.button("Sherlock it!", type="primary"):
    # 1. Generate the input data
    input_df = get_prediction_data(outcode, sqm, bedrooms, bathrooms, living_rooms, 
                                   prop_type, age_band, tenure, energy, is_conservation)

    # 2. Get the base prediction
    log_price = model.predict(input_df)[0]
    real_price = np.expm1(log_price) 

    st.metric(label="Estimated Market Value", value=f"£{real_price:,.0f}")

    # 3. Strategic Layout Advisor Section (MUST BE INDENTED HERE)
    st.divider()
    st.subheader("🧠 Strategic Layout Advisor")

    # Run your strategic advisor logic
    # Note: Use full_dataset (the variable you loaded at the top) instead of brain['full_dataset']
    advice = sherlock_strategic_advisor(input_df, full_dataset, model)

    if not advice:
        st.info("No layout changes are likely to increase value by more than 5%.")
    else:
        for i, rec in enumerate(advice, 1):
            b, ba, lr = rec['layout']
            price = rec['price']
            freq = rec['freq']
            delta = price - real_price

            st.success(
                f"**Option {i}:** {b} Bed · {ba} Bath · {lr} Living\n\n"
                f"• Estimated Value: **£{price:,.0f}**\n"
                f"• Uplift vs Current: **+£{delta:,.0f}**\n"
                f"• Market Frequency in Area: {freq} / 50 peers"
            )
