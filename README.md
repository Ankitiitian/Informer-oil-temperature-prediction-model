# Informer-oil-temperature-prediction-model

# Overview:

This program uses the Informer model (Transformer-based architecture) to predict oil temperature ('OT') from a time series dataset. The dataset contains hourly data, and the model uses 4 days (96 hours) of historical data to predict the next 24 hours of oil temperature. The script performs the following tasks:

1. **Data Preprocessing**: 
   - Loads the dataset.
   - Converts the date column to datetime and sets it as the index.
   - Extracts the target variable ('OT') and scales the data.
   - Creates sequences for time series prediction.

2. **Model Definition**:
   - Implements an Informer model with an encoder-decoder architecture.
   - The model uses positional encoding, Transformer layers for encoding and decoding time series data.

3. **Training and Evaluation**:
   - The model is trained using mean squared error loss and Adam optimizer.
   - It is evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics.

4. **Plotting**:
   - The results are visualized by plotting actual vs predicted oil temperature over time using Matplotlib.

### How to Run the Program:

#### Requirements:
Make sure you have the following dependencies installed:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

#### Steps to Run the Program:

1. **Prepare the Dataset**:
   - The dataset file `ett.csv` should be placed in the same directory as the script.
   - The dataset should have a date column and an 'OT' (oil temperature) column.

2. **Run the Script**:
   - Execute the Python script `script.py` by running the following command:
   
     ```bash
     python script.py
     ```

3. **Model Training**:
   - The model will be trained for 10 epochs by default, and you'll see the training and validation loss after each epoch.

4. **Evaluation**:
   - After training, the script evaluates the model on the test set and prints performance metrics (MSE, MAE, and R²).

5. **Visualization**:
   - The script will plot a graph showing the actual vs predicted oil temperature over time.

### Key Functions:

1. **`create_sequences`**: 
   - This function generates input-output sequences for the time series model based on the given sequence length and prediction length.

2. **`TimeSeriesDataset`**:
   - A custom dataset class to load the time series data for the model.

3. **`Informer`**:
   - The main model class implementing the Informer architecture using PyTorch's Transformer layers.

4. **`train_model`**:
   - Trains the Informer model using training data and validates it on the validation set.

5. **`evaluate_model`**:
   - Evaluates the trained model on the test data and computes performance metrics.

6. **`plot_results`**:
   - Plots actual vs predicted oil temperatures using the test dataset to visualize model performance.

