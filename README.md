# ID2223 Project - New York Residential Property Valuation
Group DLLM - Pang Jun Rong & Lachun Li

## Public UI URL
https://huggingface.co/spaces/jaydxn1/NY-Residential-Property-Valuation

You may use the provided inference_test.csv file to test; alternatively you may use the scraper.py to generate a new set of data points (past_days=1) for valuation.

## Overview
In many real estate investment firms, deal sourcing is a laborious and time-consuming task requiring extensive analyses of the local market. The due diligence is conducted based on past financial indicators retrieved from third-party market data vendors, and analysts are expected to perpetually update their figures with respect to market volatility for the most informed decision-making to take place. By building a serverless ML system, we can automatically identify potential deals based on streams providing the latest financial data – increasing the efficiency of deal sourcing and providing new business opportunities previously overlooked or undiscovered by investment professionals.

The data used for this project is extracted using HomeHarvest, an open-source scraping library which fetches property data from "realtor.com" in the Multiple Listing Service (MLS) format, which is used by real estate agents in the US.

In this project, the magnitude of “undervaluedness” can serve as a clear indication to investment analysts that certain properties are worth more to look into within a locale. The primary algorithm we employed is XGBoost Regression, and the past 180 days of residential property data in New York was used to train the model. The testing data was extracted from a more recent day not within the 180 days of training data, and preliminary results indicate a 99.6% model performance.

## Results
When using the provided test data called inference_test.csv, we observe that only 1 property is undervalued, 3 properties are overvalued and the rest are fair-valued. The graph provided in the UI also shows that the model makes reasonable predictions of housing prices and can detect anomalies based on the publically available data.

## Running The Code
You can run the entire code from scratch by following the instructions below:
1. Modify the past_days argument in scraper.py to extract your intended dataset. It is recommended to do past_days=1 for a reasonably-sized dataset which is guaranteed to not overlap the training dataset used for this project.
2. Run scraper.py and save the resulting .csv file in the same directory.
3. Run feature_engineering_pipeline.ipynb to perform exploratory analysis of the training dataset, and to wrangle the data into a desired format. The data will be uploaded to Hugging Face for use in the next step.
3. Run training_pipeline.ipynb; the training dataset ny_realtor_listing_sales.csv will be used but if you wish to generate your own training data, you may do Steps 1-2 again with past_days=180 or similar. The important thing to note is that your testing data should not be a subset of the generated training data, hence using the original training dataset is strongly recommended.
4. Run inference_pipeline.ipynb if you wish to run the UI locally, otherwise you can perform gradio deploy and work with the code found in app.py.
5. Upload your test data file into the UI, and the results are shown on the right - the rightmost column in the dataframe indicates whether a property is undervalued, fair-valued or overvalued. The threshold of 5% between the prediced and actual value is used to determine if a property is fairly valued.
