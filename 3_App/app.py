import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pgeocode # <--- NEW LIBRARY

# --- 1. SETUP ---
st.set_page_config(page_title="London Value Predictor", page_icon="🏠")

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

# --- 4. MAIN INTERFACE ---
st.title("🇬🇧 London House Value Estimator")

if st.button("Calculate Value", type="primary"):
    # ... (Rest of logic is identical to previous version) ...
    # Copy lines 88-124 from the previous code block here
    # Just ensure you use the new get_prediction_data function defined above
    
    input_df = get_prediction_data(outcode, sqm, bedrooms, bathrooms, living_rooms, 
                                   prop_type, age_band, tenure, energy, is_conservation)
    
    log_price = model.predict(input_df)[0]
    real_price = np.expm1(log_price) 
    
    st.metric(label="Estimated Market Value", value=f"£{real_price:,.0f}")
    
    st.divider()
    st.subheader("💡 Renovation ROI Analysis")
    
    recommendations = []
    
    if sqm > 65 and bathrooms < 3:
        ghost_df = get_prediction_data(outcode, sqm, bedrooms, bathrooms + 1, living_rooms, 
                                       prop_type, age_band, tenure, energy, is_conservation)
        ghost_price = np.expm1(model.predict(ghost_df)[0])
        diff = ghost_price - real_price
        if diff > 5000:
            recommendations.append(f"**Add Bathroom:** Estimated Value Increase: **£{diff:,.0f}**")
            
    best_rating = valid_ratings[0]
    if energy != best_rating:
        ghost_df_e = get_prediction_data(outcode, sqm, bedrooms, bathrooms, living_rooms, 
                                         prop_type, age_band, tenure, best_rating, is_conservation)
        ghost_price_e = np.expm1(model.predict(ghost_df_e)[0])
        diff_e = ghost_price_e - real_price
        if diff_e > 2000:
            recommendations.append(f"**Improve Efficiency to {best_rating}:** Estimated Value Increase: **£{diff_e:,.0f}**")

    if recommendations:
        for rec in recommendations:
            st.success(rec)
    else:
        st.info("No clear high-ROI renovations found.")