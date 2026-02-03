# 🎯 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.2+-61dafb.svg)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-79.06%25-brightgreen.svg)](https://github.com)

A comprehensive machine learning solution for predicting customer churn in telecommunications companies. This project combines advanced data science techniques with a beautiful, user-friendly interface to help businesses identify at-risk customers and take proactive retention measures.

## 🌟 Features

- **🤖 Advanced ML Pipeline**: Comprehensive data preprocessing, feature selection, and model comparison
- **🎨 Modern UI/UX**: Beautiful React frontend with glassmorphism design and responsive layout
- **📊 Smart Form**: Reduced from 19 to 6 key inputs without sacrificing prediction accuracy
- **⚡ Real-time Predictions**: Instant churn risk assessment with actionable recommendations
- **📈 High Accuracy**: 79.06% accuracy with 83.20% ROC-AUC score
- **🔧 Multiple Algorithms**: Comparison of Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting
- **📋 Detailed Analytics**: Comprehensive model evaluation with all key metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sarveshh77/CUSTOMER_CHUNK_PREDICTION.git
   cd CUSTOMER_CHUNK_PREDICTION
   ```

2. **Set up the Backend (Flask)**
   ```bash
   cd churn-prediction-backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

3. **Set up the Frontend (React)**
   ```bash
   cd ../churn-prediction-frontend
   npm install
   ```

4. **Get Dataset and Train the Model**
   
   ⚠️ **Note**: The dataset and trained model files are not included in this repository due to size constraints.
   
   **Train the model:**
   ```bash
   cd ../churn-prediction-backend
   python train_model.py
   ```
   
   This will create the following files in the `model/` directory:
   - `churn_model.pkl` (trained Random Forest model)
   - `encoders.pkl` (label encoders)
   - `feature_importance.pkl` (feature rankings)
   - `model_comparison.csv` (algorithm comparison)

5. **Start the Servers**
   
   **Backend (Terminal 1):**
   ```bash
   cd churn-prediction-backend
   venv\Scripts\activate
   python app.py
   ```
   
   **Frontend (Terminal 2):**
   ```bash
   cd churn-prediction-frontend
   npm start
   ```

6. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## 📁 Project Structure

```
CUSTOMER_CHUNK_PREDICTION/
├── 📊 churn-prediction-backend/
│   ├── 🗂️ model/                   # ⚠️ Generated after training (not in repo)
│   │   ├── churn_model.pkl          # Trained Random Forest model
│   │   ├── encoders.pkl             # Label encoders for categorical features
│   │   ├── feature_importance.pkl   # Feature importance rankings
│   │   └── model_comparison.csv     # Algorithm comparison results
│   ├── 📄 app.py                    # Flask API server
│   ├── 🤖 train_model.py            # Comprehensive ML pipeline
│   ├── 📊 dataset.csv              # ⚠️ Download separately (not in repo)
│   └── 📋 requirements.txt         # Python dependencies
├── 🎨 churn-prediction-frontend/
│   ├── 📁 public/
│   ├── 📁 src/
│   │   ├── 🧩 components/
│   │   │   ├── ChurnForm.js         # Smart prediction form
│   │   │   └── ResultCard.js        # Results display with recommendations
│   │   ├── 🔧 services/
│   │   │   └── api.js               # API service layer
│   │   ├── 🎨 App.css               # Modern styling with glassmorphism
│   │   └── 📄 App.js                # Main React component
│   └── 📦 package.json             # Node.js dependencies
└── 📖 README.md                    # Project documentation
```

## 🔬 Model Performance

### Algorithm Comparison Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | **79.06%** | **63.21%** | **50.53%** | **56.17%** | **83.20%** |
| Gradient Boosting | 80.77% | 67.82% | 52.41% | 59.13% | 84.42% |
| Logistic Regression | 79.91% | 64.09% | 55.35% | 59.40% | 84.10% |
| Decision Tree | 77.43% | 57.91% | 54.81% | 56.32% | 76.38% |

### Key Features Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | TotalCharges | 18.73% | Total amount charged to customer |
| 2 | MonthlyCharges | 17.94% | Monthly subscription fee |
| 3 | tenure | 15.50% | Length of customer relationship |
| 4 | Contract | 7.93% | Contract type (month-to-month, yearly) |
| 5 | PaymentMethod | 5.00% | How customer pays bills |
| 6 | OnlineSecurity | 4.95% | Online security service subscription |

## 📡 API Endpoints

### POST `/predict`

Predict customer churn risk based on customer data.

**Request Body:**
```json
{
  "MonthlyCharges": 45.00,
  "tenure": 36,
  "TotalCharges": 1620.00,
  "Contract": "Two year",
  "PaymentMethod": "Credit card (automatic)",
  "OnlineSecurity": "Yes",
  "gender": "Female",
  "SeniorCitizen": "0",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "PaperlessBilling": "No"
}
```

**Response:**
```json
{
  "churn": "NO"
}
```

## 🎯 Usage Examples

### Low Risk Customer Profile
```javascript
const lowRiskCustomer = {
  MonthlyCharges: "45.00",
  tenure: "36",
  TotalCharges: "1620.00",
  Contract: "Two year",
  PaymentMethod: "Credit card (automatic)",
  OnlineSecurity: "Yes"
};
// Expected Result: "NO" (Low Risk)
```

### High Risk Customer Profile
```javascript
const highRiskCustomer = {
  MonthlyCharges: "120.00",
  tenure: "1",
  TotalCharges: "120.00",
  Contract: "Month-to-month",
  PaymentMethod: "Electronic check",
  OnlineSecurity: "No"
};
// Expected Result: "YES" (High Risk)
```

## 📸 Screenshots

### Main Dashboard
![O/P]("./Screenshot 2026-02-03 231950.png")
## ⚠️ **Repository Notes**

**Files Not Included in Repository:**
- 📊 `dataset.csv`
- 🤖 `model/` directory - Generated after running `python train_model.py`
- 📁 `venv/` - Create virtual environment locally
- 📁 `node_modules/` - Install with `npm install`

**Reason for Exclusion:**
- **Dataset**: Large file size and potential licensing restrictions
- **Model files**: Generated during training, can be recreated
- **Dependencies**: Platform-specific, should be installed locally

## 🛠️ Technology Stack

### Backend
- **Python 3.9+** - Core programming language
- **Flask** - Web framework for API
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **joblib** - Model serialization

### Frontend
- **React 18.2** - User interface library
- **JavaScript ES6+** - Programming language
- **CSS3** - Modern styling with gradients and animations
- **Axios** - HTTP client for API calls

### Machine Learning
- **Random Forest** - Primary prediction algorithm
- **Label Encoding** - Categorical variable handling
- **Feature Selection** - Dimensionality reduction
- **Cross-validation** - Model evaluation

## 📊 Data Science Pipeline

1. **📥 Data Loading**: 7,043 customer records with 21 features
2. **🧹 Data Preprocessing**: 
   - Handle missing values (11 TotalCharges entries)
   - Encode 16 categorical variables
   - Feature scaling and normalization
3. **🔍 Feature Selection**: 
   - Identify top 15 most important features
   - Reduce dimensionality while maintaining accuracy
4. **🤖 Model Training**: 
   - Compare 4 different algorithms
   - Hyperparameter optimization
   - Cross-validation
5. **📈 Model Evaluation**: 
   - Comprehensive metrics analysis
   - Confusion matrix interpretation
   - ROC curve analysis

## 🎨 UI/UX Features

- **🌈 Modern Design**: Gradient backgrounds with glassmorphism effects
- **📱 Responsive Layout**: Works on desktop, tablet, and mobile
- **⚡ Smart Form**: Auto-calculation of derived fields
- **🎯 Visual Feedback**: Loading states and animated transitions
- **💡 Helpful Tips**: Built-in guidance and explanations
- **🔔 Actionable Results**: Specific recommendations based on predictions

### CORS Settings
The Flask app is configured to accept requests from:
- http://localhost:3000
- http://localhost:3001
- http://localhost:3002

## 🧪 Testing

### Test the API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MonthlyCharges": "45.00",
    "tenure": "36",
    "TotalCharges": "1620.00",
    "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)",
    "OnlineSecurity": "Yes"
  }'
```

### Run Frontend Tests
```bash
cd churn-prediction-frontend
npm test
```
## 📈 Performance Optimization

- **Model Caching**: Pre-loaded encoders and model for faster predictions
- **Input Validation**: Client and server-side validation
- **Lazy Loading**: Efficient component loading
- **Code Splitting**: Optimized bundle sizes
- **API Optimization**: Minimal data transfer

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Write unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sarvesh** - *Initial work* - [sarveshh77](https://github.com/sarveshh77)

## 📞 Support

For support, email your-email@example.com or create an issue in this repository.

---

**⭐ If you found this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/sarveshh77/CUSTOMER_CHUNK_PREDICTION.svg?style=social&label=Star)](https://github.com/sarveshh77/CUSTOMER_CHUNK_PREDICTION)
