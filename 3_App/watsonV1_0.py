import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pgeocode
import os
# full_dataset= pd.read_csv(r"..\1_Data\df_cleaned.csv")
# df_cArea_yBuilt= pd.read_csv(r'..\1_data\App\df_lbsm_streamlit.csv')
# Get the folder where THIS file (watsonV1_0.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to this file
# Note: Ensure folder names like '1_Data' match your repo EXACTLY (Case Sensitive!)
path_data = os.path.join(current_dir, '..', '1_Data', 'df_cleaned.csv')
path_conservation = os.path.join(current_dir, '..', '1_Data', 'App', 'df_lbsm_streamlit.csv')
path_image = os.path.join(current_dir, 'SH2.png')
full_dataset = pd.read_csv(path_data)
df_cArea_yBuilt = pd.read_csv(path_conservation)
# --- 1. SETUP ---
st.set_page_config(page_title="Sherlock Homes", page_icon=path_image)

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
# Conservation area data
df_isconservation = df_cArea_yBuilt[df_cArea_yBuilt['in_conservation_area'] == 1]
postcodes_in_conservation_area = df_isconservation['postcode']
conservation_area_options = ('automatic check', 'yes', 'no')

def get_CArea(postcode_input, postcodes_in_conservation_area, is_conservation_choice):
    if is_conservation_choice == 'automatic check':
        return postcode_input in postcodes_in_conservation_area
    elif is_conservation_choice == 'yes':
        return True
    else:
        return False

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
is_conservation_choice = st.sidebar.selectbox("In Conservation Area?", conservation_area_options)
is_conservation = get_CArea(outcode, postcodes_in_conservation_area, is_conservation_choice)
valid_ratings = [x for x in options['currentEnergyRating'] if str(x) != 'nan']
valid_ratings.sort() 
energy = st.sidebar.selectbox("Energy Rating", valid_ratings, index=len(valid_ratings)-3)
epc_options = ("Sherlock's recommendation", "Same", "A", "B", "C", "D", "E", "F")
epc_choice = st.sidebar.selectbox("Desired EPC rating?", epc_options, index=0)

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

def get_confidence_interval(prediction, brain_data):
    """
    Calculates the 50% confidence interval based on price bins.
    """
    conf_data = brain_data['confidence_bins']
    conf_map = conf_data['map_user'] # Using the 50% user map

    # 1. Determine the Price Label
    # (Matches the bins used in training: 0, 250k, 550k, 1M, 1.6M, inf)
    if prediction < 250000:
        label = 'low'
    elif prediction < 550000:
        label = 'medium'
    elif prediction < 1000000:
        label = 'high'
    elif prediction < 1600000:
        label = 'very-high'
    else:
        label = 'luxury'

    # 2. Get Multipliers & Calculate
    # keys are 0.25 and 0.75 because that's how we saved the quantiles
    multipliers = conf_map[label]
    low = prediction * multipliers[0.25]
    high = prediction * multipliers[0.75]

    return low, high

def sherlock_strategic_advisor(user_row, full_dataset, watson_model, epc_choice):
    user_lat = user_row['latitude'].iloc[0]
    user_lon = user_row['longitude'].iloc[0]
    user_sqm = user_row['floorAreaSqM'].iloc[0]
    user_epc = user_row['currentEnergyRating'].iloc[0]

    # Current Value (Baseline)
    current_price = np.expm1(watson_model.predict(user_row)[0])

    # 1st & 2nd: Calculate Bounds and find 50 Nearest Peers
    lower, upper = get_sqm_bounds(user_sqm)
    area_filtered = full_dataset[(full_dataset['floorAreaSqM'] >= lower) &
                                 (full_dataset['floorAreaSqM'] <= upper)].copy()

    area_filtered['dist'] = np.sqrt((area_filtered['latitude'] - user_lat)**2 +
                                    (area_filtered['longitude'] - user_lon)**2)
    nearest_50 = area_filtered.sort_values('dist').head(50)

    # ---- EPC helpers (kept inside to avoid extra global names) ----
    order = ["A", "B", "C", "D", "E", "F", "G"]
    rank = {k: i for i, k in enumerate(order)}

    def _norm_epc(x):
        x = str(x).strip().upper()
        return x if x in rank else None

    def _median_epc(peers):
        s = peers["currentEnergyRating"].map(_norm_epc).dropna()
        if s.empty:
            return None
        r = s.map(rank).astype(int)
        med = int(np.median(r))
        med = max(0, min(med, len(order) - 1))
        return order[med]

    def _better(a, b):
        # True if a is better (higher EPC) than b
        a = _norm_epc(a); b = _norm_epc(b)
        if a is None: return False
        if b is None: return True
        return rank[a] < rank[b]

    def _next_better(x):
        x = _norm_epc(x)
        if x is None:
            return "C"
        if x == "A":
            return "A"
        return order[max(rank[x] - 1, 0)]

    def _epc_for_option_A():
        # Option A: median of nearest_50 if it's better than user's; else user's
        if epc_choice == "Same":
            return _norm_epc(user_epc) or "C"
        if epc_choice in order:
            return epc_choice
        # Sherlock's recommendation
        med = _median_epc(nearest_50)
        if med is None:
            return _norm_epc(user_epc) or "C"
        return med if _better(med, user_epc) else (_norm_epc(user_epc) or "C")

    def _epc_for_option_B():
        # Option B: if worse than C -> C, else next better (C->B, B->A, A->A)
        if epc_choice == "Same":
            return _norm_epc(user_epc) or "C"
        if epc_choice in order:
            return epc_choice
        # Sherlock's recommendation
        u = _norm_epc(user_epc)
        if u is None:
            return "C"
        if rank[u] > rank["C"]:
            return "C"
        return _next_better(u)

    epc_A = _epc_for_option_A()
    epc_B = _epc_for_option_B()

    # 3rd: Physical Feasibility & Prediction
    best_A = None   # best price among freq layouts (Option A)
    best_B = None   # best price among all feasible layouts (Option B)

    for b in range(1, 6):
        for ba in range(1, 4):
            for lr in range(1, 3):
                # Physical feasibility
                if (b * 9) + (ba * 4) + (lr * 12) + 10 > user_sqm:
                    continue

                # Frequency based on real peers (layout existence)
                freq = len(nearest_50[(nearest_50['bedrooms'] == b) &
                                      (nearest_50['bathrooms'] == ba) &
                                      (nearest_50['livingRooms'] == lr)])

                # Build a sim row once
                sim_row = user_row.copy()
                sim_row['bedrooms'], sim_row['bathrooms'], sim_row['livingRooms'] = b, ba, lr

                # --- Option A prediction uses epc_A ---
                sim_row_A = sim_row.copy()
                sim_row_A['currentEnergyRating'] = epc_A
                sim_row_A['currentEnergyRating'] = sim_row_A['currentEnergyRating'].astype('category')
                pred_A = np.expm1(watson_model.predict(sim_row_A)[0])

                # --- Option B prediction uses epc_B ---
                sim_row_B = sim_row.copy()
                sim_row_B['currentEnergyRating'] = epc_B
                sim_row_B['currentEnergyRating'] = sim_row_B['currentEnergyRating'].astype('category')
                pred_B = np.expm1(watson_model.predict(sim_row_B)[0])

                # Only keep if > 5% higher than current (same rule you already had)
                if pred_A >= current_price * 1.1 and freq >= 3:
                    if (best_A is None) or (pred_A > best_A['price']):
                        best_A = {'layout': (b, ba, lr), 'price': pred_A, 'freq': freq, 'epc': epc_A}

                if pred_B >= current_price * 1.1:
                    if (best_B is None) or (pred_B > best_B['price']):
                        best_B = {'layout': (b, ba, lr), 'price': pred_B, 'freq': freq, 'epc': epc_B}

    # Final return logic — ALWAYS put Option A first if both exist
    if best_A and best_B:
        if best_A['price'] >= best_B['price']:
            return [best_A]
        else:
            return [best_A, best_B]

    if best_A:
        return [best_A]

    if best_B:
        return [best_B]

    return []



# --- 4. MAIN INTERFACE ---
# Create columns to put Title on left and Image on right
col_title, col_img = st.columns([4, 1]) 

with col_title:
    # Custom HTML to make the title BIGGER than standard st.title
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>Sherlock Homes</h1>", unsafe_allow_html=True)

with col_img:
    # Make sure SH2.png is in the same folder!
    st.image(path_image, width=120) 

if st.button("Sherlock it!", type="primary"):
    # 1. Generate the input data
    input_df = get_prediction_data(outcode, sqm, bedrooms, bathrooms, living_rooms, 
                                   prop_type, age_band, tenure, energy, is_conservation)

    # 2. Get the base prediction
    log_price = model.predict(input_df)[0]
    real_price = np.expm1(log_price)
    # Confidence interval 25% to 75%
    low_bound, high_bound = get_confidence_interval(real_price, brain)

    st.metric(label="Estimated Market Value", value=f"£{real_price:,.0f}")
    st.caption(f"📉 50% of similar properties sell between: £{low_bound:,.0f} — £{high_bound:,.0f}")

    # 3. Strategic Layout Advisor Section
    st.divider()

    # Header with Icon
    col1, col2 = st.columns([1, 10]) 
    with col1:
        st.markdown("## 👣") 
    with col2:
        st.subheader("We sneaked through London to uncover this:")

    # Run logic
    advice = sherlock_strategic_advisor(input_df, full_dataset, model, epc_choice)

    if not advice:
        st.info("Although we couldn't find any recommendations to increase value by more than 10%, our team would be glad to support you further on!")
    else:
        # Define helper function
        def freq_text(freq):
            if freq >= 6: return "✅ This layout is **common in the area**."
            elif freq >= 3: return "ℹ️ This layout is **used in the area**, although not very common."

        # --- 1. PREPARE CURRENT VALUES ---
        curr_b = input_df['bedrooms'].iloc[0]
        curr_ba = input_df['bathrooms'].iloc[0]
        curr_lr = input_df['livingRooms'].iloc[0]
        curr_e = input_df['currentEnergyRating'].iloc[0]

        # --- 2. DISPLAY RECOMMENDATIONS ---
        for i, rec in enumerate(advice, 1):
            b, ba, lr = rec['layout']
            price = rec['price']
            freq = rec['freq']
            rec_epc = rec['epc']
            delta = price - real_price

            # --- CALCULATE CHANGES ---
            all_changes = []

            # Check Rooms (Bed, Bath, Living)
            room_data = [
                (b, curr_b, "bedroom"),
                (ba, curr_ba, "bathroom"),
                (lr, curr_lr, "living room")
            ]

            for new_val, old_val, label in room_data:
                if new_val != old_val:
                    diff = new_val - old_val
                    sign = "+" if diff > 0 else "-"
                    # Pluralize if absolute difference > 1
                    name = f"{label}s" if abs(diff) != 1 else label
                    all_changes.append(f"{sign}{abs(diff)} {name}")

            # Check EPC
            if rec_epc != curr_e:
                all_changes.append(f"{rec_epc} rating")

            # --- CONSTRUCT THE SENTENCE ---
            if not all_changes:
                uplift_sentence = f"The estimated Uplift for this layout is **£{delta:,.0f}**"
            else:
                # Join with commas and an 'and' for the last item
                if len(all_changes) > 1:
                    change_str = "**, **".join(all_changes[:-1]) + " and " + all_changes[-1]
                else:
                    change_str = all_changes[0]

                uplift_sentence = f"With **{change_str}**, the estimated **Uplift is +£{delta:,.0f}**"

            # --- FREQUENCY LOGIC ---
            freq_html = ""
            if freq > 0:
                f_msg = "✅ This layout is **common in the area**." if freq >= 6 else "ℹ️ This layout is **used in the area**, although not very common."
                freq_html = f"\n{f_msg}\n_Market frequency: {freq} / 50 nearest peers_"

            # --- RENDER CARD ---
            title = "Sherlock's recommendation" if i == 1 else "Alternative recommendation"

            msg = (
                f"### {title}\n"
                f"Estimated Value\n"
                f"### £{price:,.0f}\n" 
                f"{uplift_sentence}\n\n" 
                f"---\n" 
                f"**Proposed:** {b} Bed · {ba} Bath · {lr} Living | **EPC:** {rec_epc}\n"
                f"{freq_html}"
            )

            st.success(msg)


        # --- 2. SHOW NEXT STEP / CTA LAST (Markdown) ---
        st.divider()
        st.markdown(
            "### Next step\n"
            "Watson AI and our team are ready to assess the feasibility and optimization potential of this proposal.\n\n"
            "**Upload the floor plans for a tailored architectural proposal and detailed review.**"
        )
