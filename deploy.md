# 🚀 Deployment Guide - Critical Power & W' Prediction App

## Quick Deploy to Vercel

### Option 1: GitHub Integration (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Dissertation ML web app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/Login with GitHub
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the Python/Streamlit setup
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.vercel.app`

### Option 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

4. **Follow the prompts**
   - Set project name
   - Choose your team/account
   - Confirm deployment settings

## 📁 Files Overview

Your project now contains:

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `vercel.json` - Vercel configuration
- `README.md` - Project documentation
- `Dissertation_Baseline.ipynb` - Your original notebook
- `.gitignore` - Git ignore rules

## 🔧 Configuration

The `vercel.json` file is already configured to:
- Use Python runtime
- Route all requests to `app.py`
- Handle Streamlit deployment

## 🎯 What Your App Includes

### 📊 Interactive Sections
1. **Overview** - Project introduction and key insights
2. **Data Analysis** - EDA with interactive visualizations
3. **Model Performance** - ML model comparisons
4. **Interactive Predictions** - Real-time prediction tool
5. **Methodology** - Research methodology details

### 🤖 Features
- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, etc.
- **Interactive Visualizations**: Plotly charts, correlation heatmaps
- **Real-time Predictions**: Input athlete data and get CP/W' predictions
- **Performance Metrics**: MAPE, R², RMSE comparisons
- **Responsive Design**: Works on desktop and mobile

## 🎨 Customization

### Update Content
- Modify the project description in `app.py`
- Add your specific research findings
- Update methodology section with your approach

### Replace Sample Data
- Replace the `generate_sample_data()` function with your actual dataset
- Load your trained models using `joblib.load()`
- Update feature names to match your data

### Styling
- Modify the CSS in the `st.markdown()` section
- Update colors, fonts, and layout
- Add your institution's branding

## 🔍 Troubleshooting

### Common Issues

1. **Build Fails on Vercel**
   - Check that all packages are in `requirements.txt`
   - Verify Python version compatibility
   - Check build logs in Vercel dashboard

2. **Import Errors**
   - Ensure all dependencies are listed in `requirements.txt`
   - Test locally first: `streamlit run app.py`

3. **App Not Loading**
   - Check the Vercel deployment URL
   - Verify `vercel.json` configuration
   - Check browser console for errors

### Performance Tips

1. **Caching**
   - The app uses `@st.cache_data` for expensive computations
   - Models are cached to avoid retraining

2. **Data Size**
   - For large datasets, consider sampling for the demo
   - Use efficient data structures

## 📈 Next Steps

1. **Deploy your app** using one of the methods above
2. **Customize the content** with your specific research
3. **Add your actual data** and models
4. **Share the URL** with your supervisors/examiners
5. **Consider adding**:
   - User authentication
   - Data upload functionality
   - More advanced visualizations
   - Export capabilities

## 🎓 Academic Use

This web application is perfect for:
- **Dissertation Defense**: Interactive presentation of your research
- **Research Publication**: Shareable link for your findings
- **Teaching**: Educational tool for sports science
- **Collaboration**: Easy sharing with research partners

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Vercel documentation: [vercel.com/docs](https://vercel.com/docs)
3. Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
4. Test locally first before deploying

---

**Your dissertation ML analysis is now ready for the web! 🚀** 