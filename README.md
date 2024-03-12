# Customer Churn Prediction Project

## Overview

This project aims to predict customer churn using machine learning techniques. The goal is to develop a use Logistic Regression model that can identify customers at risk of churning, enabling businesses to take proactive measures to retain them. 

## Credits

This project was developed with the support and collaboration of [DataTalksClub]([https://www.datatalks.club]), a community dedicated to fostering knowledge sharing and collaboration in the field of data science.

Special thanks to the DataTalksClub community for their valuable insights and contributions to this project.


## Problem Statement

Customer churn, the loss of customers over time, poses a significant challenge for businesses. Early identification of potential churners allows companies to implement targeted strategies for customer retention, ultimately improving overall customer satisfaction and profitability.

## Project Structure

- `src/`: Contains the source code for the machine learning model and prediction API.
- `data/`: Placeholder for the dataset used in the project.
- `pipeline/`: Saved models generated during training.
- `Dockerfile`: Configuration file for building the Docker image.
- `Pipfile`: List of Python dependencies.
- `json`: an inference example to test it in platforms like Postman.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction

2. **Run the docker image:**

 `docker run -p 8000:8000 churn`

3. **Open Postman and choose New Request -> Select POST -> URL:  http://0.0.0.0:8000/predict_churn -> Body - raw - JSON
   The result obtained: (Example)

{
    "Probability": 0.5341817046575446,
    "Churn": 1
}



    
