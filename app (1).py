from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Bangalore House Price Predictor', page_icon='🏡', layout='wide')
BASE = Path(__file__).resolve().parent

st.markdown('''<style>
.stApp {background: #f5f8fc;}
.block-container {max-width: 1050px; padding-top: 2rem;}
.hero {background: linear-gradient(110deg,#15345c,#2876ad); color:white; padding:30px; border-radius:18px; margin-bottom:22px;}
.hero h1 {color:white; margin:0; font-size:2.1rem;}
.hero p {color:#e2edf9; margin:8px 0 0;}
[data-testid="stForm"] {background:white; padding:22px; border-radius:15px; border:1px solid #e2e8f0;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    required = ['model1.pkl', 'locations.pkl', 'area_type.pkl']
    missing = [name for name in required if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing files: ' + ', '.join(missing))
    with (BASE / 'model1.pkl').open('rb') as f:
        model = pickle.load(f)
    with (BASE / 'locations.pkl').open('rb') as f:
        locations = pickle.load(f)
    with (BASE / 'area_type.pkl').open('rb') as f:
        area_types = pickle.load(f)
    return model, sorted({str(x) for x in locations if pd.notna(x)}), sorted({str(x) for x in area_types if pd.notna(x)})

st.markdown('<div class="hero"><h1>🏡 Bangalore House Price Predictor</h1><p>Apne dream home ki estimated price seconds mein check karein.</p></div>', unsafe_allow_html=True)

try:
    model, locations, area_types = load_assets()
except Exception as exc:
    st.error(f'Model load nahi ho paaya: {exc}')
    st.info('app.py ke same folder mein model1.pkl, locations.pkl aur area_type.pkl rakhein. Sirf trusted pickle files use karein.')
    st.stop()

if not locations or not area_types:
    st.error('Location ya area type ki list empty hai. PKL files check karein.')
    st.stop()

st.subheader('📋 Property Details')
with st.form('prediction_form'):
    left, right = st.columns(2, gap='large')
    with left:
        location = st.selectbox('📍 Location', locations)
        area_type = st.selectbox('🏢 Area Type', area_types)
        bhk = st.number_input('🛏️ Bedrooms (BHK)', min_value=1, max_value=20, value=2, step=1)
    with right:
        total_sqft = st.number_input('📐 Total Square Feet', min_value=100.0, max_value=50000.0, value=1200.0, step=50.0)
        bath = st.number_input('🛁 Bathrooms', min_value=1, max_value=20, value=2, step=1)
        balcony = st.number_input('🌇 Balconies', min_value=0, max_value=20, value=1, step=1)
    submitted = st.form_submit_button('🔍 Predict House Price', type='primary', use_container_width=True)

if submitted:
    user_input = pd.DataFrame([{
        'area_type': area_type,
        'location': location,
        'size': int(bhk),
        'total_sqft': float(total_sqft),
        'bath': float(bath),
        'balcony': int(balcony),
    }])
    try:
        prediction = float(model.predict(user_input)[0])
        if not (-1e12 < prediction < 1e12):
            raise ValueError('Model returned an invalid number.')
        st.divider()
        st.subheader('🎯 Estimated House Price')
        if prediction <= 0:
            st.warning('Model ne non-positive price predict ki hai. Yeh estimate reliable nahi hai; inputs aur model check karein.')
        else:
            st.success(f'₹ {prediction:,.2f} Lakhs')
            st.metric('Approximate price in crores', f'₹ {prediction/100:,.2f} Cr')
        st.caption(f'{bhk} BHK • {total_sqft:,.0f} sq ft • {location} • {area_type}')
        with st.expander('View model input'):
            st.dataframe(user_input, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')
        st.info('Model ko notebook wale exact 6 features chahiye. Check karein ki model1.pkl sahi notebook se export hua hai.')

st.divider()
st.caption('Educational ML project • Predictions are estimates, not official property valuations.')
