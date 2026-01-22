# 🛍️ E-commerce Product Categorizer

An intelligent machine learning application that automatically classifies e-commerce product descriptions into four distinct categories: **Electronics**, **Household**, **Books**, and **Clothing & Accessories**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Categories](#categories)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Development](#development)
- [Academic Context](#academic-context)
- [License](#license)

## 🎯 Overview

This Streamlit-based web application leverages a pre-trained Linear Support Vector Classifier (LinearSVC) to categorize product descriptions in real-time. The system uses TF-IDF vectorization for text feature extraction and provides confidence scores for predictions.

### Key Capabilities

- **Real-time Classification**: Instant categorization of product descriptions
- **Confidence Scoring**: Shows prediction confidence using decision function scores
- **Sample Generator**: Built-in sample descriptions for quick testing
- **Classification History**: Tracks the last 10 classifications with timestamps
- **Text Preprocessing**: Automated cleaning and normalization of input text
- **Interactive UI**: Modern, gradient-styled interface with responsive design

## ✨ Features

### User Features

- 📝 **Text Input Area**: Enter custom product descriptions (up to unlimited characters)
- 🎲 **Sample Generator**: Randomly select from 28 pre-loaded sample descriptions
- 🔍 **One-Click Classification**: Simple button to trigger prediction
- 📊 **Confidence Metrics**: View prediction confidence as percentage
- 📜 **History Tracking**: Review past classifications with expandable details
- 🗑️ **Clear Functions**: Reset input or history with one click
- 📈 **Character Counter**: Real-time character count display

### Technical Features

- ⚡ **Cached Model Loading**: Uses `@st.cache_resource` for optimal performance
- 🧹 **Advanced Text Preprocessing**:
  - Lowercase conversion
  - URL removal
  - Email address removal
  - Special character handling
  - Whitespace normalization
- 🎨 **Custom CSS Styling**: Gradient-based category cards and polished UI
- 💾 **Session State Management**: Maintains state across interactions
- 📱 **Responsive Design**: Works on various screen sizes

## 🎯 Categories

The model classifies products into four main categories:

| Category                   | Icon | Description                          | Examples                                |
| -------------------------- | ---- | ------------------------------------ | --------------------------------------- |
| **Electronics**            | 📱   | Tech gadgets, devices, adapters      | Headphones, cables, keyboards, webcams  |
| **Household**              | 🏠   | Home decor, kitchen items, utilities | Lamps, utensils, bottles, bedsheets     |
| **Books**                  | 📚   | All genres and formats               | Textbooks, novels, cookbooks, self-help |
| **Clothing & Accessories** | 👕   | Apparel, fashion items               | Shirts, shoes, wallets, watches, bags   |

## 🏗️ Technical Architecture

### Data Flow

```
User Input → Text Preprocessing → TF-IDF Vectorization → LinearSVC Model → Prediction + Confidence
```

### Component Breakdown

1. **Frontend**: Streamlit web interface with custom CSS
2. **Preprocessing**: Regex-based text cleaning pipeline
3. **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
4. **Classification**: Linear Support Vector Classifier
5. **Post-processing**: Softmax-like transformation for confidence scores

### Model Pipeline

```python
# Preprocessing
text → lowercase → remove URLs/emails → remove special chars → normalize whitespace

# Vectorization (TF-IDF)
preprocessed_text → vectorizer.transform() → feature vector

# Classification
feature_vector → model.predict() → category label
feature_vector → model.decision_function() → confidence scores
```

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone or Download the Repository**

   ```bash
   cd "c:\Users\63963\Dropbox\4TH YEAR 1ST SEM\Elective 3\Ecommerce Text Classification FINAL PIT"
   ```

2. **Install Required Dependencies**

   ```bash
   pip install streamlit scikit-learn numpy
   ```

3. **Verify Model File Exists**
   Ensure `linearsvc.pkl` is in the project directory. This file contains both the trained model and vectorizer.

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

5. **Access the Web Interface**
   The app will automatically open in your default browser at `http://localhost:8501`

### Dependencies

```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

## 🚀 Usage

### Basic Workflow

1. **Launch the Application**

   ```bash
   streamlit run app.py
   ```

2. **Enter Product Description**

   - Type or paste a product description in the text area
   - Or click "🎲 Generate Sample" for a random example

3. **Classify the Product**

   - Click the "🔍 Classify Product" button
   - View the predicted category and confidence score

4. **Review History**
   - Scroll down to see past classifications
   - Click on history items to expand details

### Example Descriptions to Try

**Electronics:**

```
Wireless Bluetooth headphones with active noise cancellation, 40-hour battery life,
and premium sound quality. Includes carrying case and audio cable.
```

**Household:**

```
Modern LED floor lamp with adjustable brightness and color temperature control.
Features remote control, timer function, and energy-efficient bulbs.
```

**Books:**

```
A comprehensive guide to machine learning algorithms, covering supervised and
unsupervised learning, neural networks, and practical Python implementations.
```

**Clothing & Accessories:**

```
Premium leather wallet with RFID blocking technology, multiple card slots, and
zippered coin compartment. Handcrafted from genuine cowhide leather.
```

## 🤖 Model Details

### Training Data

- **Source**: E-commerce product descriptions dataset
- **Size**: 50,000+ labeled samples
- **Format**: CSV with Category and Description columns
- **Distribution**: Balanced across 4 categories

### Algorithm Specifications

| Component             | Details                                            |
| --------------------- | -------------------------------------------------- |
| **Model Type**        | Linear Support Vector Classifier (LinearSVC)       |
| **Vectorization**     | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Input Features**    | Text-based product descriptions                    |
| **Output Classes**    | 4 categories (multi-class classification)          |
| **Training Method**   | Supervised learning                                |
| **Confidence Metric** | Decision function with softmax transformation      |

### Model Architecture

```python
# Model components saved in linearsvc.pkl
{
    'model': LinearSVC(class_weight=None, dual=True, ...),
    'vectorizer': TfidfVectorizer(max_features=None, ...)
}
```

### Performance Characteristics

- **Inference Speed**: < 100ms per prediction
- **Memory Footprint**: Minimal (cached after first load)
- **Scalability**: Handles descriptions up to several paragraphs
- **Accuracy**: High confidence on clear product descriptions

### Text Preprocessing Pipeline

```python
def preprocess_text(text):
    # 1. Lowercase conversion
    # 2. URL removal (http/https patterns)
    # 3. Email address removal
    # 4. Special character filtering (keeps alphanumeric + spaces)
    # 5. Whitespace normalization
    # 6. Strip leading/trailing spaces
    return cleaned_text
```

## 📁 Project Structure

```
Ecommerce Text Classification FINAL PIT/
│
├── app.py                          # Main Streamlit application
├── linearsvc.pkl                   # Pre-trained model and vectorizer
├── README.md                       # This file
├── .github/
│   └── copilot-instructions.md     # AI assistant context
└── .git/                           # Git version control
```

### File Descriptions

- **app.py**: Complete Streamlit web application with UI, preprocessing, and prediction logic
- **linearsvc.pkl**: Pickled dictionary containing trained LinearSVC model and TfidfVectorizer
- **README.md**: Comprehensive project documentation
- **.github/copilot-instructions.md**: Development guidelines and technical context

## 🛠️ Development

### Running in Development Mode

```bash
# Run with live reload
streamlit run app.py

# Run on custom port
streamlit run app.py --server.port 8502
```

### Testing Predictions Locally

```python
import pickle

# Load model
with open('linearsvc.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
vectorizer = data['vectorizer']

# Test prediction
test_text = ["Wireless Bluetooth Headphones"]
prediction = model.predict(vectorizer.transform(test_text))
print(f"Category: {prediction[0]}")
```

### Modifying the Application

#### Adding New Sample Descriptions

Edit the `SAMPLE_DESCRIPTIONS` list in [app.py](app.py#L7):

```python
SAMPLE_DESCRIPTIONS = [
    "Your new sample description here",
    # ... existing samples
]
```

#### Customizing UI Colors

Modify the CSS in [app.py](app.py#L93):

```python
st.markdown("""
<style>
    .electronics { background: linear-gradient(...); }
    .household { background: linear-gradient(...); }
    # Add your custom gradients
</style>
""", unsafe_allow_html=True)
```

#### Updating Categories

**⚠️ Important**: If you change categories, you must:

1. Retrain the model with new labels
2. Replace `linearsvc.pkl`
3. Update category_info dictionary in [app.py](app.py#L230)
4. Update sidebar information in [app.py](app.py#L308)

### Critical Development Notes

- **Never use `.fit_transform()`** on the vectorizer in production - always use `.transform()`
- The vectorizer vocabulary is frozen from training time
- Model loading uses `@st.cache_resource` (not `@st.cache_data`) for proper lifecycle management
- Input must be a list for vectorizer: `[user_input]` not `user_input`

## 🎓 Academic Context

**Course**: IT4R12 - Elective 3  
**Institution**: [Your University Name]  
**Semester**: 4th Year, 1st Semester  
**Developers**: VILLARTE & VIESCA  
**Project Type**: Machine Learning Classification System  
**Academic Year**: 2025-2026

### Learning Objectives Demonstrated

- ✅ Text preprocessing and feature engineering
- ✅ Machine learning model training and deployment
- ✅ Web application development with Streamlit
- ✅ Model serialization and loading
- ✅ User interface design for ML applications
- ✅ Session state management in web apps

## 🔧 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: linearsvc.pkl not found`

- **Solution**: Ensure `linearsvc.pkl` is in the same directory as `app.py`

**Issue**: Empty predictions or errors during classification

- **Solution**: Verify input is non-empty and contains text (not just special characters)

**Issue**: Model loads slowly on first run

- **Solution**: This is normal; subsequent runs will be cached and fast

**Issue**: Confidence scores seem unusual

- **Solution**: Decision function scores are transformed using softmax approximation - this is expected behavior

### Getting Help

For issues specific to this project:

1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.7+)
3. Test model loading independently (see [Testing Predictions Locally](#testing-predictions-locally))

## 📊 Future Enhancements

Potential improvements for future versions:

- [ ] Multi-language support for product descriptions
- [ ] CSV bulk upload for batch classification
- [ ] Export classification history to CSV
- [ ] Fine-tuning interface for custom categories
- [ ] API endpoint for programmatic access
- [ ] Model performance metrics dashboard
- [ ] Integration with e-commerce platforms
- [ ] Advanced confidence visualization (bar charts)

## 📄 License

This project is developed for academic purposes as part of the IT4R12 course curriculum.

---

**Built with** ❤️ **using Streamlit and scikit-learn**

_For questions or collaboration, contact the developers: VILLARTE & VIESCA_
