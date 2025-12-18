import streamlit as st
import pickle
import random
import numpy as np
import re

# Sample product descriptions for testing (not from training data)
SAMPLE_DESCRIPTIONS = [
    # Electronics
    "USB-C charging cable, 6 feet long, supports fast charging up to 60W",
    "Wireless Bluetooth headphones with active noise cancellation, 40-hour battery life, and premium sound quality. Includes carrying case and audio cable for wired connection.",
    "4K webcam with auto-focus and built-in microphone, perfect for video conferencing and streaming",
    "Portable power bank 20000mAh with dual USB ports and LED display showing remaining battery percentage",
    "RGB mechanical gaming keyboard with customizable backlighting and programmable macro keys",
    
    # Books
    "The Art of Computer Programming by Donald Knuth - Classic algorithms textbook",
    "A comprehensive guide to machine learning algorithms, covering supervised and unsupervised learning, neural networks, and practical implementations in Python with scikit-learn.",
    "Mystery thriller novel set in Victorian London, featuring a detective solving supernatural crimes",
    "Complete cookbook with over 500 vegetarian recipes from around the world, includes nutritional information",
    "Self-help book about building productive habits and achieving personal goals through small daily improvements",
    
    # Household
    "Stainless steel kitchen utensil set, 5 pieces including spatula, ladle, and tongs",
    "Modern LED floor lamp with adjustable brightness and color temperature control. Features remote control, timer function, and energy-efficient bulbs. Perfect for living rooms and bedrooms.",
    "Ceramic dinner plate set, service for 6, microwave and dishwasher safe",
    "Decorative wall clock with silent quartz movement, 12-inch diameter, wooden frame",
    "Vacuum insulated stainless steel water bottle, keeps drinks cold for 24 hours or hot for 12 hours, leak-proof lid",
    "Cotton bedsheet set with pillowcases, queen size, 400 thread count, wrinkle resistant",
    
    # Clothing & Accessories
    "Men's cotton polo shirt, navy blue, size large, classic fit",
    "Premium leather wallet with RFID blocking technology, multiple card slots, and zippered coin compartment. Slim design fits comfortably in front pocket. Handcrafted from genuine cowhide leather.",
    "Women's running shoes with breathable mesh upper and cushioned sole, available in multiple colors",
    "Sunglasses with polarized lenses and UV400 protection, lightweight metal frame",
    "Wool winter scarf, extra soft and warm, measures 70 inches long",
    "Stainless steel analog wristwatch with leather strap, water resistant up to 50 meters",
    "Canvas backpack with laptop compartment, multiple pockets, and padded shoulder straps",
    "Cotton socks pack of 6 pairs, crew length, moisture wicking fabric"
]

# 1. Load the bundled model and vectorizer
# We use st.cache_resource to load it only once, making the app faster
@st.cache_resource
def load_model():
    with open('linearsvc.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Preprocessing function for user input
def preprocess_text(text):
    """
    Preprocesses user input text before classification.
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove special characters (keep alphanumeric and spaces)
    5. Remove extra whitespace
    6. Strip leading/trailing whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Load data
try:
    data = load_model()
    model = data['model']
    vectorizer = data['vectorizer']
    # If your pickle saved them as a tuple or individual vars, adjust above
except FileNotFoundError:
    st.error("Error: The file 'linearsvc.pkl' was not found. Please upload it to the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# 2. Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .category-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .electronics { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .household { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .books { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .clothing { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .char-counter {
        text-align: right;
        font-size: 0.85rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown('<p class="main-header">🛍️ E-commerce Product Categorizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Machine Learning | Classify products into Electronics, Household, Books, or Clothing & Accessories</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. User Input Section
st.subheader("📝 Product Description")
user_input = st.text_area(
    "Enter or generate a product description:",
    value=st.session_state.user_input,
    height=150,
    placeholder="Example: Wireless Bluetooth Headphones with noise cancellation, 40-hour battery life..."
)

# Update session state with current input
st.session_state.user_input = user_input

# Character counter
char_count = len(user_input)
st.markdown(f'<div class="char-counter">{char_count} characters</div>', unsafe_allow_html=True)

# 4. Action Buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🎲 Generate Sample", use_container_width=True, type="secondary"):
        st.session_state.user_input = random.choice(SAMPLE_DESCRIPTIONS)
        st.session_state.show_result = False
        st.rerun()

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.user_input = ""
        st.session_state.show_result = False
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.rerun()

with col3:
    classify_btn = st.button("🔍 Classify Product", use_container_width=True, type="primary")

# 5. Classification Logic
if classify_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a product description first.")
    else:
        with st.spinner("🤔 Analyzing product description..."):
            # Preprocess user input
            preprocessed_input = preprocess_text(user_input)
            
            # Vectorize preprocessed text
            input_vector = vectorizer.transform([preprocessed_input])
            
            # Predict
            prediction = model.predict(input_vector)[0]
            
            # Get confidence scores using decision_function
            decision_scores = model.decision_function(input_vector)[0]
            
            # Convert to confidence percentages (approximate probabilities)
            # Using softmax-like transformation for better interpretation
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            confidences = exp_scores / exp_scores.sum()
            
            # Get the confidence for predicted category
            predicted_idx = list(model.classes_).index(prediction)
            confidence = confidences[predicted_idx] * 100
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.all_confidences = dict(zip(model.classes_, confidences * 100))
            st.session_state.show_result = True
            
            # Add to history (keep last 10)
            from datetime import datetime
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                'prediction': prediction,
                'confidence': confidence
            }
            st.session_state.history.insert(0, history_entry)  # Add to front
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]

# Category icons and colors (used for results and history)
category_info = {
    "Electronics": {"icon": "📱", "class": "electronics", "emoji": "⚡"},
    "Household": {"icon": "🏠", "class": "household", "emoji": "🏡"},
    "Books": {"icon": "📚", "class": "books", "emoji": "📖"},
    "Clothing & Accessories": {"icon": "👕", "class": "clothing", "emoji": "👗"}
}

# 6. Display Results
if st.session_state.show_result and st.session_state.prediction:
    st.markdown("---")
    st.subheader("📊 Classification Results")
    
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence
    
    info = category_info.get(prediction, {"icon": "📦", "class": "electronics", "emoji": "✨"})
    
    # Main result card
    st.markdown(f"""
    <div class="category-card {info['class']}">
        <h2 style="margin: 0; font-size: 2rem;">{info['icon']} {prediction}</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# 7. Classification History Section
if len(st.session_state.history) > 0:
    st.markdown("---")
    col_header, col_clear = st.columns([3, 1])
    with col_header:
        st.subheader("📜 Classification History")
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    st.caption(f"Showing {len(st.session_state.history)} most recent classification(s)")
    
    # Display history items
    for idx, entry in enumerate(st.session_state.history):
        cat_info = category_info.get(entry['prediction'], {"icon": "📦", "class": "electronics"})
        
        with st.expander(f"{cat_info['icon']} {entry['prediction']} ({entry['confidence']:.1f}%) - {entry['timestamp']}", expanded=(idx==0)):
            st.write(f"**Description:** {entry['description']}")
            st.caption(f"🕒 Classified on {entry['timestamp']}")

# 8. Enhanced Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("ℹ️ About")
    
    st.markdown("### 🎓 Project Information")
    st.info("""
    **Section:** IT4R12  
    **Developed by:** VILLARTE & VIESCA  
    **Project:** E-commerce Text Classification
    """)
    
    st.markdown("---")
    
    st.markdown("### 🤖 Model Details")
    st.write("""
    This application uses a **Linear Support Vector Classifier (SVC)** 
    trained on e-commerce product descriptions.
    """)
    
    with st.expander("📊 Model Specifications"):
        st.write("""
        - **Algorithm:** LinearSVC
        - **Vectorizer:** TfidfVectorizer
        - **Training Data:** 50,000+ products
        - **Categories:** 4 classes
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Supported Categories")
    st.success("📱 **Electronics**\nGadgets, devices, adapters")
    st.error("🏠 **Household**\nHome decor, kitchen items")
    st.info("📚 **Books**\nAll genres and formats")
    st.warning("👕 **Clothing & Accessories**\nApparel, fashion items")
    
    st.markdown("---")
    
    st.markdown("### 💡 How It Works")
    with st.expander("Click to learn more"):
        st.write("""
        1. **Enter** a product description
        2. **Click** 'Classify Product'
        3. **View** predicted category and confidence
        4. The model analyzes text patterns to determine the category
        """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sample Descriptions", len(SAMPLE_DESCRIPTIONS))
    with col2:
        st.metric("Categories", "4")
    
    st.markdown("---")
    st.caption("Built with Streamlit & scikit-learn")