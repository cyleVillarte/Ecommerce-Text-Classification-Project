# E-commerce Text Classification Project

## Project Overview

Academic ML project that classifies e-commerce product descriptions into 4 categories: **Electronics**, **Household**, **Books**, and **Clothing & Accessories**. Uses a pre-trained Linear SVM model deployed via Streamlit web interface.

## Architecture & Data Flow

1. **Model Loading** (`@st.cache_resource`): Pre-trained LinearSVC + TfidfVectorizer loaded from `linearsvc.pkl` at app startup
2. **User Input** → Text vectorization (`.transform()` only, never `.fit_transform()`) → Prediction → Category display
3. **Training data**: `IT4R12_VILLARTE_VIESCA_ECOMMERCE TEXT CLASSIFICATION.csv` contains labeled product descriptions (50k+ samples)

## Critical Technical Details

### Model Files

- **Primary model**: `linearsvc.pkl` (dict with 'model' and 'vectorizer' keys)
- **Legacy file**: `linear_svm_model_and_vectorizer.pkl` (referenced in error handling but not actively used)
- **Never retrain or fit** the vectorizer in production - it must use the vocabulary from training

### Streamlit Best Practices (This Project)

- Use `@st.cache_resource` for model loading (not `@st.cache_data`) - loads once per app lifecycle
- Vectorizer usage: **Always** use `.transform()`, never `.fit_transform()` (critical for production inference)
- Input validation: Check for empty strings with `.strip()` before processing
- Sidebar for static info; main area for interactive elements

## Running the Application

```powershell
# Install dependencies
pip install streamlit scikit-learn

# Launch app
streamlit run app.py
```

App opens at `http://localhost:8501` with live reload on file changes.

## Common Development Tasks

### Testing predictions locally

```python
# Quick test in Python REPL
import pickle
with open('linearsvc.pkl', 'rb') as f:
    data = pickle.load(f)
test_text = ["Wireless Bluetooth Headphones"]
prediction = data['model'].predict(data['vectorizer'].transform(test_text))
```

### Modifying categories

If categories change, you must:

1. Retrain the model with new labels
2. Replace `linearsvc.pkl`
3. Update sidebar info in `app.py` (line 48-50)

### UI Customization

- **Title/emoji**: Line 24 (`st.title()`)
- **Input placeholder**: Line 29 (`placeholder=` parameter)
- **Sidebar content**: Lines 47-50 (within `st.sidebar` context)

## Data Format Expectations

CSV structure: `Category,Description` (no header row)

- First column: Category label (exact match to training categories)
- Second column: Product description text (can contain commas/quotes)

## Debugging Tips

- **Model load errors**: Check pickle file exists and dict keys ('model', 'vectorizer')
- **Prediction errors**: Ensure input is list format for vectorizer: `[user_input]` not `user_input`
- **Empty predictions**: Verify vectorizer vocabulary hasn't changed (compare with training artifacts)

## Academic Context

IT4R12 course project by VILLARTE & VIESCA. Model training likely done in separate notebook (not included in this repo).
