# ID2223 Project - New York Residential Property Valuation
Group DLLM - Pang Jun Rong & Lachun Li

## Overview
In many real estate investment firms, deal sourcing is a laborious and time-consuming task requiring extensive analyses of the local market. The due diligence is conducted based on past financial indicators retrieved from third-party market data vendors, and analysts are expected to perpetually update their figures with respect to market volatility for the most informed decision-making to take place. By building a serverless ML system, we can automatically identify potential deals based on streams providing the latest financial data – increasing the efficiency of deal sourcing and providing new business opportunities previously overlooked or undiscovered by investment professionals.

The data used for this project is extracted using HomeHarvest, an open-source scraping library which fetches property data from "realtor.com" in the Multiple Listing Service (MLS) format, which is used by real estate agents in the US.

In this project, the magnitude of “undervaluedness” can serve as a clear indication to investment analysts that certain properties are worth more to look into within a locale. The primary algorithm we employed is XGBoost Regression, and the past 180 days of residential property data in New York was used to train the model. The testing data was extracted from a more recent day not within the 180 days of training data, and preliminary results indicate a 99.6% model performance.

## Running The Code 