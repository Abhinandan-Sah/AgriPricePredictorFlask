from flask import Flask, jsonify, render_template, request, app, Response
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import io 
import base64
from joblib import load,dump
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import json
import joblib
import gzip


applications = Flask(__name__)
app=applications
CORS(applications, resources={r"/*": {"origins": "https://agripricepredictor-1.onrender.com"}})


df = pd.read_csv('Model/crop_price.csv')

# Preprocess Data
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month_name()
df["Day"] = df["Date"].dt.day
df.drop("Date", axis=1, inplace=True)
continuous_numerical_features = [
    'Temperature (°C)', 'Rainfall (mm)', 'Supply Volume (tons)',
    'Demand Volume (tons)', 'Transportation Cost (₹/ton)',
    'Fertilizer Usage (kg/hectare)', 'Pest Infestation (0-1)',
    'Market Competition (0-1)', 'Price (₹/ton)'
]
categorical_features = ['State', 'City', 'Crop Type', 'Season', 'Year', 'Month', 'Day']

# Helper function to convert matplotlib plots to base64 images for HTML rendering
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/api/index', methods=['GET', 'POST'])
def test():
    # Retrieve JSON data from the request
    data = request.get_json()
    print(f"Data received: {data}")
    # Process the data if needed
    # For now, just return a success message
    return jsonify({"message": "Success", "data_received": data})

# Home page with all visualizations
@app.route('/')
def home():
    return render_template('index.html')

# Route to display EDA summary statistics
@app.route('/stats')
def stats():
    stats_data = df.describe().to_html()
    return jsonify(stats_data)

# Route to show histogram plots
@app.route('/histograms')
def histograms():
    plot_urls = []
    for col in continuous_numerical_features:
        plt.figure()
        sns.histplot(df[col], bins=10)
        plt.title(f'Histogram of {col}')
        plot_url = plot_to_base64()
        plot_urls.append(plot_url)
        plt.clf()
    return render_template('histograms.html', plot_urls=plot_urls)

# Route to show box plots
@app.route('/boxplots')
def boxplots():
    plot_urls = []
    for col in continuous_numerical_features:
        plt.figure()
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
        plot_url = plot_to_base64()
        plot_urls.append(plot_url)
        plt.clf()
    return render_template('boxplots.html', plot_urls=plot_urls)

@app.route('/categorical_analysis')
def categorical_analysis():
    plot_urls = []
    for col in categorical_features:
        plt.figure(figsize=(12,6))
        sns.countplot(data=df, x=col, palette="Set2")
        plt.title(f'Countplot of {col}')
        plot_url = plot_to_base64()
        plot_urls.append(f"data:image/png;base64,{plot_url}")
        plt.clf()
    return jsonify(plot_urls=plot_urls)

# Bivariate and multivariate analysis
@app.route('/bivariate_analysis')
def bivariate_analysis():
    # Generate scatter plot for Supply Volume vs Price
    plt.figure()
    sns.scatterplot(data=df, x='Supply Volume (tons)', y='Price (₹/ton)')
    supply_vs_price = plot_to_base64()
    plt.clf()
    
    # Generate scatter plot for Demand Volume vs Price
    plt.figure()
    sns.scatterplot(data=df, x='Demand Volume (tons)', y='Price (₹/ton)')
    demand_vs_price = plot_to_base64()
    plt.clf()

    return jsonify({
        'supply_vs_price': f"data:image/png;base64,{supply_vs_price}",
        'demand_vs_price': f"data:image/png;base64,{demand_vs_price}"
    })

# Heatmaps and Pivot Tables
@app.route('/heatmaps')
def heatmaps():
    x = df.pivot_table(index="Crop Type", columns="Month", values="Price (₹/ton)", aggfunc="mean")
    plt.figure(figsize=(20,10))
    sns.heatmap(x, annot=True, linewidths=0.5, cmap="coolwarm")
    heatmap_price = plot_to_base64()
    plt.clf()

    y = df.pivot_table(index="Crop Type", columns="Month", values="Transportation Cost (₹/ton)", aggfunc="mean")
    plt.figure(figsize=(20,10))
    sns.heatmap(y, annot=True, linewidths=0.5, cmap="Blues")
    heatmap_transport_cost = plot_to_base64()
    plt.clf()

    return jsonify({
        'heatmap_price': f"data:image/png;base64,{heatmap_price}",
        'heatmap_transport_cost': f"data:image/png;base64,{heatmap_transport_cost}"
    })









# ----------------------------------------------------------------

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# -----------------------------------------------------------------------------------
# # Load the models
# with open("Model/RandomForest.pkl", "rb") as f:
#     rf_pipe = pickle.load(f)

# prophet_model = load("Model/Meta_prophet.joblib")

# # prophet_model = load("Model/prophet_transformer.pkl")

# with open("Model/final_model.pkl", "rb") as f:
#     final_model = pickle.load(f)

try:
    with gzip.open('Model/random.joblib.gz', 'rb') as f:
        rf_pipe= joblib.load(f)
except (FileNotFoundError, EOFError):
    rf_pipe = None  # Handle the case where the model file is not found or is corrupted

try:
    with gzip.open('Model/meta.joblib.gz', 'rb') as f:
        prophet_model= joblib.load(f)
except (FileNotFoundError, EOFError):
    prophet_model = Prophet()  # Handle the case where the model file is not found or is corrupted

try:
    with gzip.open('Model/finale.joblib.gz', 'rb') as f:
        final_model= joblib.load(f)
except (FileNotFoundError, EOFError):
    final_model = None 



# @app.route('/predict',methods=['GET','POST'])
# def predict():
#     result = ""

#     if request.method == 'POST':
#         # Retrieve JSON data from the request
#         data = request.get_json()
#         print("Success inside predict ")
#         # Extract data from JSON
#         date = data.get('date')
#         state = data.get('state')
#         city = data.get('city')
#         crop_type = data.get('crop_type')
#         season = data.get('season')
#         temp = float(data.get('temperature'))
#         rainfall = float(data.get('rainfall'))
#         supply_volume = float(data.get('supply_volume'))
#         demand_volume = float(data.get('demand_volume'))
#         transport_cost = float(data.get('transport_cost'))
#         fertilizer_usage = float(data.get('fertilizer_usage'))
#         pest_infestation = float(data.get('pest_infestation'))
#         market_competition = float(data.get('market_competition'))


#         # year = int(data.get('year'))
#         # month = data.get('month')


#         # class ProphetTransformer(BaseEstimator, TransformerMixin):  # BaseEstimator and TransformerMixin added
#         #     def _init_(self, loaded_model=None):
#         #         self.prophet_model = loaded_model if loaded_model is not None else Prophet()

#         #     def fit(self, X, y=None):
#         #         # Fit the Prophet model (though here you may not need to since it's pre-trained)
#         #         df_prophet = pd.DataFrame({'ds': X['Date'], 'y': y})
#         #         self.prophet_model.fit(df_prophet)
#         #         return self

#         #     def transform(self, X, y=None):
#         #         # Transform (predict future) using the Prophet model
#         #         future_dates = pd.DataFrame({'ds': X['Date']})
#         #         forecast = self.prophet_model.predict(future_dates)
#         #         return forecast[['yhat']].values








#         # Prepare input features for prediction
#         input_features = np.array([
#             date,state, city, crop_type, season, temp, rainfall, supply_volume,
#             demand_volume, transport_cost, fertilizer_usage, pest_infestation,
#             market_competition
#         ], dtype=object).reshape(1, 13)

#         col_names=["date","State","City","Crop Type","Season","Temperature (°C)","Rainfall (mm)","Supply Volume (tons)","Demand Volume (tons)","Transportation Cost (₹/ton)","Fertilizer Usage (kg/hectare)","Pest Infestation (0-1)","Market Competition (0-1)"]
#         # Convert the input data to a DataFrame (similar to training data)
#         input_df = pd.DataFrame(input_features, columns=col_names)
        
#         input_df["date"] = pd.to_datetime(input_df["date"], format="%Y-%m-%d")

#         # Prepare the date column for Prophet
#         input_df["ds"] = input_df["date"]


#         # Make predictions using the loaded models
#         rf_pred = rf_pipe.predict(input_df.drop(columns=["ds", "date"]))
#         prophet_pred= prophet_model.predict(input_df[['ds']]) 
#         combined_features = np.column_stack((rf_pred,prophet_pred['yhat'].values))
#         final_pred = final_model.predict(combined_features)  # working
#         # print(input_df['ds'].dtype)
#         # return jsonify({"predicted_price": 20.2})



#         # # Make prediction
#         # prediction = model.predict(input_features)
#         print(f"This is Final predict - {final_pred[0]}")
#         predicted_price = round(final_pred[0], 2)

#         # Return the prediction result
#         # result = {"predicted_price": final_pred[0]}
#         result = {"predicted_price": predicted_price}

#     return jsonify(result)

@app.route('/predict',methods=['GET','POST'])
def predict():
    result = ""

    if request.method == 'POST':
        # Retrieve JSON data from the request
        data = request.get_json()
        print("Success inside predict ")
        # Extract data from JSON
        date = data.get('date')
        state = data.get('state')
        city = data.get('city')
        crop_type = data.get('crop_type')
        season = data.get('season')
        temp = float(data.get('temperature'))
        rainfall = float(data.get('rainfall'))
        supply_volume = float(data.get('supply_volume'))
        demand_volume = float(data.get('demand_volume'))
        transport_cost = float(data.get('transport_cost'))
        fertilizer_usage = float(data.get('fertilizer_usage'))
        pest_infestation = float(data.get('pest_infestation'))
        market_competition = float(data.get('market_competition'))



        # Prepare input features for prediction
        input_features = np.array([
            date,state, city, crop_type, season, temp, rainfall, supply_volume,
            demand_volume, transport_cost, fertilizer_usage, pest_infestation,
            market_competition
        ], dtype=object).reshape(1, 13)

        col_names=["date","State","City","Crop Type","Season","Temperature (°C)","Rainfall (mm)","Supply Volume (tons)","Demand Volume (tons)","Transportation Cost (₹/ton)","Fertilizer Usage (kg/hectare)","Pest Infestation (0-1)","Market Competition (0-1)"]
        # Convert the input data to a DataFrame (similar to training data)
        input_df = pd.DataFrame(input_features, columns=col_names)
        
        input_df["date"] = pd.to_datetime(input_df["date"], format="%Y-%m-%d")

        # Prepare the date column for Prophet
        input_df["ds"] = input_df["date"]


        # Make predictions using the loaded models
        rf_pred = rf_pipe.predict(input_df.drop(columns=["ds", "date"]))
        prophet_pred= prophet_model.predict(input_df[['ds']]) 
        combined_features = np.column_stack((rf_pred,prophet_pred['yhat'].values))
        final_pred = final_model.predict(combined_features)  # working
        # print(input_df['ds'].dtype)
        # return jsonify({"predicted_price": 20.2})



        # # Make prediction
        # prediction = model.predict(input_features)
        print(f"This is Final predict - {final_pred[0]}")
        predicted_price = round(final_pred[0], 2)

        # Return the prediction result
        # result = {"predicted_price": final_pred[0]}
        # result = {"predicted_price": predicted_price}

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(prophet_pred['ds'], prophet_pred['yhat'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Predicted Price Over Time')

        # Convert the graph to a base64-encoded image string
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png')
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Add the graph to the result JSON
        result = {"predicted_price": predicted_price, "graph": img_str}

        # forecast=prophet_model.plot(prophet_pred)
        # fig=forecast.get_figure()
        # result = {"predicted_price": predicted_price, "fig": fig}

    return jsonify(result)

    # Configure logging
# logging.basicConfig(level=logging.DEBUG)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        data_array = request.get_json()  # Parse the JSON array from the request body
        if not isinstance(data_array, list):
            return jsonify({"error": "Expected a JSON array"}), 400
        print(f"Received data array: {data_array}")  # Debug print

        # Convert new data to DataFrame
        new_data_df = pd.DataFrame(data_array)
        new_data_df["date"] = pd.to_datetime(new_data_df["date"], format="%Y-%m-%d")
    
        new_data_df["ds"] =new_data_df["date"]
  
        new_data_df["y"] = new_data_df["Price (₹/ton)"]

   
        # Split features and target for RandomForest training
        X_rf = new_data_df.drop(columns=["Price (₹/ton)", "date", "ds",'y'])
        y_rf = new_data_df["Price (₹/ton)"]
   
        # Retrain RandomForest model
        rf_pipe.fit(X_rf, y_rf)

        # Retrain Prophet model
        prophet_model = Prophet()
        prophet_model.fit(new_data_df[['ds', 'y']])

        # Generate predictions using the retrained models
        rf_preds = rf_pipe.predict(X_rf)
        prophet_preds = prophet_model.predict(new_data_df[['ds']])

        # Combine the predictions for final model training
        combined_features = np.column_stack((rf_preds,prophet_preds['yhat'].values))

        # Retrain the final model (e.g., Linear Regression or other)
        final_model.fit(combined_features, y_rf)

        # Dump (save) the models after retraining
        # with open("Model/RandomForest.pkl", "wb") as f:
        #     pickle.dump(rf_pipe, f)
        with gzip.open('random.joblib.gz', 'wb') as f:
            joblib.dump(rf_pipe,f)

        with gzip.open('meta.joblib.gz', 'wb') as f:
            joblib.dump(prophet_model,f)

        with gzip.open('finale.joblib.gz', 'wb') as f:
            joblib.dump(final_model,f)

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")  # Capture decoding errors
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({'message': 'Data received successfully'})

@app.route('/recommendations', methods=['GET'])
def recommendations():
    state="Maharashtra"
    city="Pune"
    crop_type="Rice"

    # data = request.get_json()
    # state = data.get('state', 'Maharashtra')
    # city = data.get('city', 'Pune')
    # crop_type = data.get('crop_type', 'Rice')
    # date = data.get('date', '2024-09-18')  # Example date
    # season = data.get('season', 'Kharif')
    # temperature = data.get('temperature', 30.0)
    # rainfall = data.get('rainfall', 100.0)
    # supply_volume = data.get('supply_volume', 1000.0)
    # demand_volume = data.get('demand_volume', 800.0)
    # transport_cost = data.get('transport_cost', 500.0)
    # fertilizer_usage = data.get('fertilizer_usage', 50.0)
    # pest_infestation = data.get('pest_infestation', 0.1)
    # market_competition = data.get('market_competition', 0.5)
    



    if not all([crop_type, state, city]):
        return jsonify({'error': 'Missing input parameters'}), 400

    # Load data
    df = pd.read_csv('Model/crop_price.csv')

    # Filter data
    filtered_df = df[(df['Crop Type'] == crop_type) & (df['State'] == state) & (df['City'] == city)]

    if filtered_df.empty:
        return jsonify({'error': 'No data found for the given location and crop'}), 404

    # Prepare input data for prediction
    input_data = {
        'date':'2024-09-18',
        'state':'Maharashtra',
        'city':'Pune',
        'crop_type':'Rice',
        'season': 'Kharif',
        'temperature':30.0,
        'rainfall': 100.0,
        'supply_volume':1089.67,
        'demand_volume':807.0,
        'transport_cost':578.0,
        'fertilizer_usage': 70,
        'pest_infestation': 0.34,
        'market_competition':0.6}

    # Use the existing predict function to get the predicted price
    # prediction_response = predict()
    # predicted_price = prediction_response.get_json()['predicted_price']

    prediction_response = request.post('http://localhost:5000/predict', json=input_data)
    if prediction_response.status_code != 200:
        return jsonify({'error': 'Failed to get prediction'}), prediction_response.status_code

    predicted_price = prediction_response.json().get('predicted_price')

    # Generate recommendations
    historical_avg = filtered_df['Price (₹/ton)'].mean()
    if predicted_price > historical_avg:
        recommendation = f'Sell your {crop_type} at the predicted price of ₹{predicted_price:.2f} (higher than historical average of ₹{historical_avg:.2f}).'
    else:
        recommendation = f'Hold onto your {crop_type} for better prices (predicted price of ₹{predicted_price:.2f} is lower than historical average of ₹{historical_avg:.2f}).'

    # return jsonify({
    #     'recommendation': recommendation,
    #     'predicted_price': predicted_price,
    #     'historical_average': historical_avg
    # })
    print(recommendation)

if __name__=="__main__":
    app.run(host="localhost", port=5000, debug=True)