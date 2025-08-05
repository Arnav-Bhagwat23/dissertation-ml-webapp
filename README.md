# 🚴‍♂️ Critical Power & W' Prediction - Dissertation Web App

A comprehensive machine learning web application showcasing research on predicting Critical Power (CP) and W' (Work Capacity) for endurance athletes.

## 📋 Project Overview

This project presents a machine learning analysis focused on predicting two fundamental physiological parameters in endurance sports:

- **Critical Power (CP)**: The highest power output that can be sustained indefinitely without fatigue
- **W' (Work Capacity)**: The finite amount of work that can be performed above CP before exhaustion

## 🎯 Features

### 📊 Interactive Web Application
- **Overview Section**: Project introduction and key insights
- **Data Analysis**: Exploratory data analysis with interactive visualizations
- **Model Performance**: Comparison of multiple ML algorithms
- **Interactive Predictions**: Real-time prediction tool for CP and W'
- **Methodology**: Detailed research methodology and technical implementation

### 🤖 Machine Learning Models
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Regression
- Neural Networks
- LightGBM

### 📈 Key Metrics
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Root Mean Square Error (RMSE)
- Cross-validation scores

## 🚀 Deployment on Vercel

### Prerequisites
- Python 3.8+
- Vercel account
- Git repository

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Dissertation_Hosted_Website
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser and go to `http://localhost:8501`

### Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy to Vercel**
   ```bash
   vercel
   ```

4. **Follow the prompts**
   - Set project name
   - Choose your team/account
   - Confirm deployment settings

5. **Access your deployed app**
   - Vercel will provide a URL (e.g., `https://your-app.vercel.app`)

### Alternative: GitHub Integration

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the Python/Streamlit setup

3. **Deploy**
   - Vercel will automatically build and deploy your app
   - Each push to main will trigger a new deployment

## 📁 Project Structure

```
Dissertation_Hosted_Website/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel configuration
├── README.md             # Project documentation
└── Dissertation_Baseline.ipynb  # Original Jupyter notebook
```

## 🔧 Configuration Files

### `vercel.json`
```json
{
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### `requirements.txt`
Contains all necessary Python packages for the application.

## 📊 Data Sources

The application currently uses synthetic data generated to demonstrate the functionality. In a production environment, you would:

1. **Replace the data generation function** in `app.py` with your actual dataset
2. **Load your trained models** using `joblib.load()`
3. **Update the feature names** to match your actual data

## 🎨 Customization

### Styling
- Modify the CSS in the `st.markdown()` section of `app.py`
- Update colors, fonts, and layout as needed

### Content
- Update the project description and methodology sections
- Add your specific research findings
- Include your actual model performance metrics

### Features
- Add more interactive visualizations
- Include additional ML models
- Add data upload functionality
- Implement user authentication

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all packages in `requirements.txt` are installed
   - Check Python version compatibility

2. **Vercel Deployment Issues**
   - Verify `vercel.json` configuration
   - Check build logs in Vercel dashboard
   - Ensure all files are committed to Git

3. **Streamlit Issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Check for syntax errors in `app.py`

### Performance Optimization

1. **Caching**
   - Use `@st.cache_data` for expensive computations
   - Cache model training and predictions

2. **Data Loading**
   - Optimize data loading for large datasets
   - Consider using data sampling for demonstrations

## 📚 Research Context

This application demonstrates the application of machine learning in sports science, specifically for:

- **Performance Prediction**: Estimating athlete capabilities
- **Training Optimization**: Informing training prescription
- **Talent Identification**: Identifying key performance indicators
- **Research Validation**: Supporting scientific hypotheses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is for academic/research purposes. Please ensure compliance with your institution's policies regarding data sharing and publication.

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review Vercel documentation
- Consult Streamlit documentation

---

**Built with ❤️ using Streamlit and deployed on Vercel** 