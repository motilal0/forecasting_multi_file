import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression


# Initialize Streamlit App
st.title("Time Series Analysis & Forecasting")

# Step 1: Upload multiple files
uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True, type=["csv", "xlsx", "txt"])
dataframes = []

if uploaded_files:
    all_columns = set()

    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, parse_dates=True, infer_datetime_format=True)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, parse_dates=True, infer_datetime_format=True)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, delimiter="\t", parse_dates=True, infer_datetime_format=True)
            else:
                st.warning(f"Unsupported file format for {uploaded_file.name}")
                continue

            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            all_columns.update(df.columns)
            dataframes.append(df)

            st.write(f"Preview of {uploaded_file.name}:")
            st.write(df.head())

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if dataframes:
        # Merge DataFrames on common columns, performing an outer join
        consolidated_df = dataframes[0]
        for df in dataframes[1:]:
            consolidated_df = pd.merge(consolidated_df, df, on='date', how='outer', suffixes=("", "_dup"))

        st.success(f"Uploaded and consolidated {len(uploaded_files)} file(s).")
        st.write("Consolidated Dataset Preview:", consolidated_df.head())

# Step 2: Option to select variables for time series plot
if 'consolidated_df' in locals():
    time_var = st.selectbox("Select time variable", consolidated_df.columns)
    value_vars = st.multiselect("Select value variables", consolidated_df.columns)

if st.button("Generate Time Series Plot") and value_vars:
    for value_var in value_vars:
        if value_var in consolidated_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=consolidated_df[time_var], 
                                     y=consolidated_df[value_var], 
                                     mode="lines", 
                                     name=value_var))
            
            # Update layout to set x-axis to 'Month' and y-axis to the variable name
            fig.update_layout(
                title=f"Time Series Plot for {value_var}",
                xaxis_title="Month",
                yaxis_title=value_var,  # Dynamic y-axis title
                template="plotly_dark"
            )
            st.plotly_chart(fig)

# Step 3: Option to select variables for correlation
if 'consolidated_df' in locals():
    corr_vars = st.multiselect("Select variables for correlation", consolidated_df.columns)

    if len(corr_vars) > 1:
        corr_matrix = consolidated_df[corr_vars].corr()
        st.write("Correlation Matrix:", corr_matrix)

        significant_pairs = []
        for i in range(len(corr_vars)):
            for j in range(i + 1, len(corr_vars)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    significant_pairs.append((corr_vars[i], corr_vars[j], corr_value))

        if significant_pairs:
            st.subheader("Variable pairs with significant correlation")
            for pair in significant_pairs:
                st.write(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

        # Plotting the correlation heatmap
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

# Step 4: Feature importance plot for target variable
if 'consolidated_df' in locals():
    target_var = st.selectbox("Select target variable", consolidated_df.columns)
    feature_vars = st.multiselect("Select feature variables", [col for col in consolidated_df.columns if col != target_var])

    if st.button("Plot Feature Importance") and feature_vars:
        X = consolidated_df[feature_vars].fillna(0)
        y = consolidated_df[target_var].fillna(0)

        model = RandomForestRegressor()
        model.fit(X, y)

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({"Feature": feature_vars, "Importance": importances})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation="h", 
                     title="Feature Importance", color="Importance", 
                     color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

# 5. Forecasting
# Forecasting with models
if 'consolidated_df' in locals():
    time_var = st.selectbox("Select time variable", consolidated_df.columns, key="time_var_select")
    target_var = st.selectbox("Select target variable", [col for col in consolidated_df.columns if col != time_var], key="target_var_select")

    forecast_steps = st.slider("Select number of forecast steps", min_value=1, max_value=120, value=12, step=1, key="forecast_steps")
if st.button("Forecast"):
        models = {}

        # Preparing data
        df = consolidated_df[[time_var, target_var]].dropna()
        df[time_var] = pd.to_datetime(df[time_var])
        df = df.sort_values(by=time_var)
        train = df.iloc[:-12]
        test = df.iloc[-forecast_steps:]

        # ARIMA
        try:
            arima_model = ARIMA(train[target_var], order=(1, 1, 1)).fit()
            arima_forecast = arima_model.forecast(steps=forecast_steps)
            arima_mape = mean_absolute_percentage_error(test[target_var], arima_forecast)
            models['ARIMA'] = {'MAPE': arima_mape, 'Forecast': arima_forecast}
        except Exception as e:
            st.error(f"ARIMA error: {e}")

        # SARIMAX
        try:
            sarimax_model = SARIMAX(train[target_var], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
            sarimax_forecast = sarimax_model.forecast(steps=forecast_steps)
            sarimax_mape = mean_absolute_percentage_error(test[target_var], sarimax_forecast)
            models['SARIMAX'] = {'MAPE': sarimax_mape, 'Forecast': sarimax_forecast}
        except Exception as e:
            st.error(f"SARIMAX error: {e}")

        # Linear Regression
        try:
            train['time_index'] = range(len(train))
            test['time_index'] = range(len(train), len(train) + len(test))

            lr_model = LinearRegression()
            lr_model.fit(train[['time_index']], train[target_var])
            lr_forecast = lr_model.predict(test[['time_index']])
            lr_mape = mean_absolute_percentage_error(test[target_var], lr_forecast)
            models['Linear Regression'] = {'MAPE': lr_mape, 'Forecast': lr_forecast}
        except Exception as e:
            st.error(f"Linear Regression error: {e}")

        # Prophet
        try:
            prophet_df = train.rename(columns={time_var: 'ds', target_var: 'y'})
            prophet_model = Prophet()
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=forecast_steps)
            forecast = prophet_model.predict(future)
            prophet_forecast = forecast['yhat'][-forecast_steps:]
            prophet_mape = mean_absolute_percentage_error(test[target_var], prophet_forecast)
            models['Prophet'] = {'MAPE': prophet_mape, 'Forecast': prophet_forecast}
        except Exception as e:
            st.error(f"Prophet error: {e}")

        # Display results
        if models:
            # Create a DataFrame with only 'Model' and 'MAPE' columns
            results = pd.DataFrame([{'Model': model, 'MAPE': details['MAPE']} for model, details in models.items()])
            results['Accuracy'] = (1 - results['MAPE']) * 100
            st.write("Forecasting Results:", results[['Model', 'MAPE']])

            # Best model
            best_model = results.loc[results['MAPE'].idxmin()]
            st.subheader("Best Selected Model")
            st.write(f"Model: {best_model['Model']}")
            st.write(f"Accuracy: {best_model['Accuracy']:.2f}%")


            # Plot actual vs forecast
            best_forecast = models[best_model['Model']]['Forecast']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test[time_var], y=test[target_var], mode='lines', name='Actual', line=dict(shape='spline')))

            # Prepare the x-axis data for the entire period from actual data to forecast
            full_date_range = pd.date_range(start=df[time_var].min(), end='2024-07-01', freq='M')
            forecast_months = pd.date_range(start='2024-07-01', periods=forecast_steps, freq='M')

            # Combine actual and forecast months
            combined_date_range = full_date_range.append(forecast_months)

            # Create the figure
            fig = go.Figure()

            # Add the actual data line up to June 2024
            fig.add_trace(go.Scatter(x=df[time_var], y=df[target_var], mode='lines', name='Actual', line=dict(shape='spline')))

            # Add the forecast data starting from July 2024
            fig.add_trace(go.Scatter(x=forecast_months, y=best_forecast, mode='lines+markers', name='Forecast', line=dict(dash='dash', color='red', shape='spline')))

            # Update the x-axis to display both actual and forecast months
            # Only show labels at a reduced frequency for better readability
            step = 2  # Adjust this value to increase/decrease label frequency
            tickvals = combined_date_range[::step]  # Select every 'step'-th date for the labels
            ticktext = [date.strftime('%b %Y') for date in tickvals]  # Format the date labels

            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                title_text="Month",
                tickangle=-45,  # Rotate labels for better visibility
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig)

            # Generate forecast months starting from Jul 2024
            forecast_dates = pd.date_range(start='2024-07-01', periods=forecast_steps, freq='M')

            # Create a DataFrame with the forecasted values and formatted months
            forecast_table = pd.DataFrame({
                "Month": forecast_dates.strftime("%b %Y"),
                "Forecast": best_forecast
            }).round(0)

            # Reset the index and drop the old index column
            forecast_table = forecast_table.reset_index(drop=True)

            st.write("Forecasted Values:", forecast_table)

            # Download option
            csv = forecast_table.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecasted Values", csv, "forecast.csv", "text/csv")
