FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --timeout=600 \
    scikit-learn numpy scipy matplotlib

RUN pip install --no-cache-dir --timeout=600 \
    plotly-express plotly optuna streamlit

RUN pip install --no-cache-dir --timeout=600 \
    catboost xgboost shap

COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "models/streamlit_dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
