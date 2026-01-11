from setuptools import setup, find_packages

setup(
    name='real_estate_price_prediction',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'Flask>=3.0.0',
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'joblib>=1.3.2',
        'xgboost>=2.0.3',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Real Estate Price Prediction System using Machine Learning',
    keywords='machine-learning real-estate prediction flask',
    python_requires='>=3.8',
)
