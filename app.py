# Use inference-env

import hopsworks
project = hopsworks.login()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import gradio as gr
import matplotlib.pyplot as plt

def determine_valuation(row):
    tolerance = 0.05  # 5% tolerance
    if row['Actual'] < row['Predicted'] * (1 - tolerance):
        return 'UNDERVALUED'
    elif row['Actual'] > row['Predicted'] * (1 + tolerance):
        return 'OVERVALUED'
    else:
        return 'FAIR-VALUED'

def valuate(file):
    df = pd.read_csv(file)
    q1 = df['list_price'].quantile(0.25)
    q3 = df['list_price'].quantile(0.75)
    iqr = q3 - q1
    filtered_df = df.query('(@q1 - 1.5 * @iqr) <= list_price <= (@q3 + 1.5 * @iqr)')
    q1 = filtered_df['sqft'].quantile(0.25)
    q3 = filtered_df['sqft'].quantile(0.75)
    iqr = q3 - q1
    filtered_df = filtered_df.query('(@q1 - 1.5 * @iqr) <= sqft <= (@q3 + 1.5 * @iqr)')
    q1 = filtered_df['price_per_sqft'].quantile(0.25)
    q3 = filtered_df['price_per_sqft'].quantile(0.75)
    iqr = q3 - q1
    filtered_df = filtered_df.query('(@q1 - 1.5 * @iqr) <= price_per_sqft <= (@q3 + 1.5 * @iqr)')
    q1 = filtered_df['stories'].quantile(0.25)
    q3 = filtered_df['stories'].quantile(0.75)
    iqr = q3 - q1
    filtered_df = filtered_df.query('(@q1 - 1.5 * @iqr) <= stories <= (@q3 + 1.5 * @iqr)')
    filtered_df = filtered_df.dropna(subset=['sqft', 'list_price', 'price_per_sqft', 'stories', 'year_built'])
    filtered_df = filtered_df.dropna(subset=['sqft', 'list_price', 'price_per_sqft', 'stories', 'year_built'])
    feature_df = filtered_df.drop(columns=['property_url', 'mls', 'mls_id', 'status', 'street', 'unit', 'last_sold_date', 'sold_price', 'days_on_mls', 'primary_photo', 'alt_photos'])
    feature_df.fillna(0, inplace=True)
    le = LabelEncoder()
    le.fit(feature_df['style'])
    encoded_labels = le.transform(feature_df['style'])
    feature_df['style'] = encoded_labels
    le.fit(feature_df['city'])
    encoded_labels = le.transform(feature_df['city'])
    feature_df['city'] = encoded_labels
    le.fit(feature_df['state'])
    encoded_labels = le.transform(feature_df['state'])
    feature_df['state'] = encoded_labels
    le.fit(feature_df['list_date'])
    encoded_labels = le.transform(feature_df['list_date'])
    feature_df['list_date'] = encoded_labels    
    true_values = feature_df['list_price']
    feature_df = feature_df.drop('list_price', axis=1)

    mr = project.get_model_registry()
    model = mr.get_model("xgboost_model", version=3)
    model_dir = model.download()
    model = joblib.load(model_dir + "/xgboost_model.pkl")

    predicted_val = model.predict(feature_df)

    result = pd.DataFrame({'Property URL': filtered_df['property_url'], 'Predicted': predicted_val, 'Actual': true_values})
    result['valuation'] = result.apply(determine_valuation, axis=1)

    plot = plt.figure(figsize=(10,10))
    plt.scatter(result['Actual'], result['Predicted'], c='crimson')

    p1 = max(max(result['Predicted']), max(result['Actual']))
    p2 = min(min(result['Predicted']), min(result['Actual']))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')

    return result, plot

iface1 = gr.Interface(
    fn=valuate,
    inputs=gr.File(type='filepath'),
    outputs=["dataframe", "plot"],
    title="New York Residential Property Valuation",
    description="Identify Undervalued Properties in New York!",
)

gr.TabbedInterface(
    [iface1]
).launch(share=True)