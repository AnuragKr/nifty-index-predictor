import pandas as pd
import joblib
import pickle
import ta  
import uvicorn
import pandas_datareader.data as web
import os
import requests
from datetime import datetime, timedelta

from nsepython import index_history
from mangum import Mangum
from fastapi import FastAPI
from fastapi.responses import JSONResponse


# Configure joblib for Lambda environment
joblib.parallel.DEFAULT_BACKEND = 'threading'
joblib.parallel.DEFAULT_N_JOBS = 1

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'NH08C1X0FR5Z950H')


# Load the trained model
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "nifty_ridge_model.pkl")
    
    # Load model with explicit configuration for Lambda
    with joblib.parallel_backend('threading', n_jobs=1):
        model = joblib.load(model_path)
    
    print(f"[MODEL] Loaded successfully from: {model_path}")
except Exception as e:
    print(f"[MODEL] ERROR loading model: {e}")
    model = None

# Initialize FastAPI
app = FastAPI(title="Nifty Index Predictor", version="1.0.0")
handler = Mangum(app)

@app.get("/")
def health_check():
    """Health check endpoint with network connectivity test"""
    try:
        # Test network connectivity
        print("[HEALTH_CHECK] Testing network connectivity...")
        response = requests.get("https://www.google.com", timeout=10)
        network_status = f"Network OK - Status: {response.status_code}"
        print(f"[HEALTH_CHECK] {network_status}")
        
        return {
            "status": "healthy", 
            "message": "Nifty Index Predictor API is running",
            "network_test": network_status,
        }
    except Exception as e:
        error_type = type(e).__name__
        print(f"[HEALTH_CHECK] ERROR - Network test failed: {error_type}: {str(e)}")
        return {
            "status": "degraded", 
            "message": "API running but network connectivity issues detected",
            "network_test": f"FAILED - {error_type}: {str(e)}",
        }

# Define prediction endpoint
@app.get("/predict")
def predict_next_close():
    """
    Predict next day's NIFTY Close.

    Returns:
        predicted next day Close (float)
    """
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    
    # Fetch last 20 trading days with Alpha Vantage API
    df = None
    ticker_used = None
    
    # Calculate dynamic date range
    #end_date = datetime.now()
    #start_date = end_date - timedelta(days=30)

    end_date = datetime.now().strftime("%d-%b-%Y")
    end_date = str(end_date)

    start_date = (datetime.now()- timedelta(days=10)).strftime("%d-%b-%Y")
    start_date = str(start_date)
    
    print(f"[DATA_FETCH] Date range: {start_date} to {end_date}")
    
    try:
        # Primary ticker with retry mechanism
        ticker = "NIFTY 50"
        max_retries = 2
        #df = web.DataReader(ticker, 'av-daily', start=start_date, end=end_date, api_key=ALPHA_VANTAGE_API_KEY)
        df = index_history(ticker, start_date, end_date)
        for attempt in range(max_retries):
            try:
                print(f"[DATA_FETCH] Attempt {attempt + 1}/{max_retries} for ticker: {ticker}")
                df = web.DataReader(ticker, 'av-daily', start=start_date, end=end_date, api_key=ALPHA_VANTAGE_API_KEY)
                print(f"[DATA_FETCH] Data: {df.tail(5)}")
                
                if not df.empty and len(df) > 0:
                    ticker_used = ticker
                    print(f"[DATA_FETCH] SUCCESS - {ticker} data shape: {df.shape}")
                    break
                else:
                    print(f"[DATA_FETCH] WARNING - Empty response for {ticker} (attempt {attempt + 1})")
                    
            except Exception as e:
                error_type = type(e).__name__
                print(f"[DATA_FETCH] ERROR - Attempt {attempt + 1} failed for {ticker}: {error_type}: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
        
        # Final validation
        if df is None or df.empty:
            print("[DATA_FETCH] CRITICAL - All ticker sources failed")
            return JSONResponse({
                "error": "Market data unavailable", 
                "details": "Alpha Vantage API access restricted or network timeout"
            }, status_code=503)
            
    except Exception as e:
        error_type = type(e).__name__
        print(f"[DATA_FETCH] CRITICAL - Unexpected error: {error_type}: {str(e)}")
        return JSONResponse({
            "error": "Data fetch system error", 
            "details": f"{error_type}: {str(e)}"
        }, status_code=500)
    
    # If columns are multiindex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = [col[0] if col[1]=='' else f"{col[0]}" for col in df.columns.values]

    # Standardize column names to match expected format
    df.columns = df.columns.str.title()  # Convert to Title Case
    print(f"[DATA_PROCESSING] Columns after standardization: {df.columns.tolist()}")

    # Keep only the last 20 rows
    df = df.tail(20).copy()
    print(f"[DATA_PROCESSING] Data after tail(20): {df.shape}")

    # ---- 1. Lag & Returns ----
    df['Return_1d'] = df['Close'].pct_change()
    df['Lag_1'] = df['Close'].shift(1)

    # ---- 2. Moving Averages ----
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # ---- 3. Volatility ----
    df['Rolling_STD_10'] = df['Return_1d'].rolling(10).std()

    # ---- 4. Momentum Indicators ----
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # ---- 5. SMA Differences ----
    df['Close_SMA5_diff'] = df['Close'] - df['SMA_5']
    df['Close_SMA20_diff'] = df['Close'] - df['SMA_20']

    # ---- 6. Select last row (most recent day) ----
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Data shape before feature selection: {df.shape}")
    
    # Check if required columns exist
    required_cols = ['Return_1d','Rolling_STD_10','RSI_14','Close_SMA5_diff','Close_SMA20_diff']
    
    features = df[required_cols].iloc[-1:]
    
    # Debug: Check if features are valid
    print(f"Features shape: {features.shape}")
    print(f"Features columns: {features.columns.tolist()}")
    print(f"Features values: {features.values}")
    
    if features.empty or features.shape[0] == 0:
        return JSONResponse({"error": "No valid features extracted from data"}, status_code=500)

    # ---- 7. Use the already loaded model ----
    # Model is already loaded at the top of the file

    # ---- 8. Predict next day Close ----
    try:
        print(f"[PREDICTION] Starting prediction with ticker: {ticker_used}")
        next_close_pred = model.predict(features)
        prediction_value = float(next_close_pred[0])
        rounded_prediction = round(prediction_value, 4)
        
        print(f"[PREDICTION] SUCCESS - Predicted value: {rounded_prediction}")
        return JSONResponse({
            "prediction": rounded_prediction,
            "ticker_used": ticker_used,
            "data_points": len(df)
        })
        
    except Exception as e:
        error_type = type(e).__name__
        print(f"[PREDICTION] ERROR - {error_type}: {str(e)}")
        print(f"[PREDICTION] DEBUG - Features shape: {features.shape}, Model type: {type(model)}")
        
        return JSONResponse({
            "error": "Prediction failed", 
            "details": f"{error_type}: {str(e)}"
        }, status_code=500)


if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)