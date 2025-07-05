# install.packages(c("tidyverse", "lubridate", "forecast", "prophet", "yardstick"))
library(tidyverse)
library(lubridate)
library(forecast)
library(prophet)
library(yardstick)


########################################
# Data Preparation
########################################

# Read file without a header, the first two rows are data components.
raw_data <- read.csv(
  "Galunggong Quarterly Farmgate Price 2002-2025.csv", 
  header = FALSE,
  na.strings = c("", "NA")
)

# Extract first three rows containing all the information.
years_row <- as.character(raw_data[1, ])
quarters_row <- as.character(raw_data[2, ])
prices_row <- as.numeric(raw_data[3, ])

# Create temporary dataframe.
temp_df <- data.frame(
  Year_raw = years_row,
  Quarter_str = quarters_row,
  Price = prices_row,
  stringsAsFactors = FALSE
)

# Fill years which are listed once.
filled_df <- temp_df %>%
  fill(Year_raw, .direction = "down")

# Create final, tidy dataframe for modeling.
galunggong_df <- filled_df %>%
  # Remove NA rows in Price column.
  filter(!is.na(Price)) %>% 
  # Convert Year to numeric data type.
  mutate(Year = as.numeric(Year_raw)) %>% 
  # Extract the number from the string Quarter 1, Quarter 2, and so on.
  mutate(Quarter_num = parse_number(Quarter_str)) %>%
  # Create a Date column from Year and Quarter_num column.
  mutate(Date = yq(paste(Year, Quarter_num))) %>%
  # Select and rename final columns for clarity.
  select(Date, Price)

head(galunggong_df)
tail(galunggong_df)

# Create Time Series object.
galunggong_ts <- ts(
  galunggong_df$Price,
  start = c(2002, 1), 
  frequency = 4       
)

print(galunggong_ts)


########################################
# Decomposition
########################################

# Decompose the time series into trend, seasonal, and random components. Use multiplicative model.
decomposition <- decompose(galunggong_ts, type = "multiplicative")
plot(decomposition)


######################################################
# Time Series Forecasting (SARIMA, and ETS)
######################################################

# Calculate the split point at approximately 80% of the dataset.
split_point <- floor(0.8 * length(galunggong_ts))

# Create training set.
train_ts <- window(galunggong_ts, end = time(galunggong_ts)[split_point])

# Create testing set.
test_ts <- window(galunggong_ts, start = time(galunggong_ts)[split_point + 1])

print(train_ts)
print(test_ts)

# SARIMA Model
# auto.arima will automatically find the best (p,d,q)(P,D,Q) parameters.
sarima_model <- auto.arima(train_ts)

# Generate forecasts of SARIMA model.
sarima_forecast <- forecast(sarima_model, h = length(test_ts))

# Plot the SARIMA forecast along with actual values.
autoplot(sarima_forecast) +
  labs(
    title = "SARIMA: Actual vs. Predicted Price",
    subtitle = "Black = Actual Data, Blue = Forecast",
    x = "Year",
    y = "Price (PHP/kg)"
  ) +
  theme_bw()

# Compare SARIMA predictions to the actual values in the test set.
sarima_comparison_df <- data.frame(
  truth = as.numeric(test_ts),
  estimate = as.numeric(sarima_forecast$mean)
)
print(sarima_comparison_df)

# ETS Model
# ets() will automatically find the best Error, Trend, and Seasonality combination.
ets_model <- ets(train_ts)

# Generate forecasts of ETS model.
ets_forecast <- forecast(ets_model, h = length(test_ts))

# Plot the ETS forecast along with actual values.
autoplot(ets_forecast) +
  labs(
    title = "ETS: Actual vs. Predicted Price",
    subtitle = "Black = Actual Data, Blue = Forecast",
    x = "Year",
    y = "Price (PHP/kg)"
  ) +
  theme_bw()

# Compare ETS predictions to the actual values in the test set.
ets_comparison_df <- data.frame(
  truth = as.numeric(test_ts),
  estimate = as.numeric(ets_forecast$mean)
)
print(ets_comparison_df)


########################################
# Time Series Forecasting (Prophet)
########################################

# Rename columns to the required 'ds' and 'y' for Prophet.
prophet_df <- galunggong_df %>%
  rename(ds = Date, y = Price)

# Specify split point for training and testing.
split_date <- as.Date("2020-04-01")
train_df <- prophet_df %>% filter(ds < split_date)
test_df <- prophet_df %>% filter(ds >= split_date)

print(train_df)
print(test_df)

# Fit Prophet with default parameters.
prophet_model <- prophet(train_df)

# Create a dataframe for the future dates to predict.
future_df <- make_future_dataframe(
  prophet_model, 
  periods = nrow(test_df),
  freq = "quarter"  
)

# Generate Prophet forecast.
prophet_forecast <- predict(prophet_model, future_df)

# Plot the Prophet forecast along with actual values.
plot(prophet_model, prophet_forecast) +
  labs(title = "Prophet Forecast vs. Actuals", x = "Date", y = "Price") +
  add_changepoints_to_plot(prophet_model)

# Compare Prophet predictions to the actual values in the test set.
comparison_df <- prophet_forecast %>%
  select(ds, yhat, yhat_lower, yhat_upper) %>%
  right_join(test_df, by = "ds")

prophet_comparison_df <- comparison_df %>%
  select(truth = y, estimate = yhat)

print(prophet_comparison_df)


###############################################################
# Evaluation of Forecasting Models (SARIMA, ETS, and Prophet)
###############################################################

# Create a set of the metrics to calculate.
metrics <- metric_set(rmse, mae, mape)

# Calculate metrics for SARIMA model. 
sarima_accuracy <- sarima_comparison_df %>%
  metrics(truth = truth, estimate = estimate, metric_set = metrics) %>%
  mutate(Model = "SARIMA")

# Calculate metrics for ETS model. 
ets_accuracy <- ets_comparison_df %>%
  metrics(truth = truth, estimate = estimate, metric_set = metrics) %>%
  mutate(Model = "ETS")

# Calculate metrics for Prophet model. 
prophet_accuracy <- prophet_comparison_df %>%
  metrics(truth = truth, estimate = estimate, metric_set = metrics) %>%
  mutate(Model = "Prophet")

# Combine the results into a single table.
all_metrics <- bind_rows(sarima_accuracy, ets_accuracy, prophet_accuracy)
print(all_metrics)

# Pivot the table to make models the columns.
final_comparison_table <- all_metrics %>%
  select(Model, .metric, .estimate) %>%
  pivot_wider(names_from = Model, values_from = .estimate) %>%
  mutate(.metric = str_to_upper(.metric))
print(final_comparison_table)


########################################
# Prophet Model Fine Tuning
########################################

# Fit Prophet with custom parameters.
prophet_model_tuned <- prophet(
  train_df, 
  changepoint.prior.scale = 0.07,  
  changepoint.range = 0.95,        
  n.changepoints = 40 
)

# Create a dataframe for the future dates to predict.
tuned_future_df <- make_future_dataframe(
  prophet_model_tuned, 
  periods = nrow(test_df),
  freq = "quarter"  
)

# Generate Prophet forecast.
tuned_prophet_forecast <- predict(prophet_model_tuned, tuned_future_df)

# Plot the Prophet forecast along with actual values.
plot(prophet_model_tuned, tuned_prophet_forecast) +
  labs(title = "Tuned-Prophet Forecast vs. Actuals", x = "Date", y = "Price") +
  add_changepoints_to_plot(prophet_model_tuned)

# Compare Prophet predictions to the actual values in the test set.
tuned_comparison_df <- tuned_prophet_forecast %>%
  select(ds, yhat, yhat_lower, yhat_upper) %>%
  right_join(test_df, by = "ds")

tuned_prophet_comparison_df <- tuned_comparison_df %>%
  select(truth = y, estimate = yhat)

print(tuned_prophet_comparison_df)

# Calculate metrics for Prophet model. 
tuned_prophet_accuracy <- tuned_prophet_comparison_df %>%
  metrics(truth = truth, estimate = estimate, metric_set = metrics) %>%
  mutate(Model = "Tuned Prophet")

# Combine the results into a single table.
all_metrics2 <- bind_rows(sarima_accuracy, ets_accuracy, prophet_accuracy, tuned_prophet_accuracy)
print(all_metrics2)

# Pivot the table to make models the columns.
final_comparison_table2 <- all_metrics2 %>%
  select(Model, .metric, .estimate) %>%
  pivot_wider(names_from = Model, values_from = .estimate) %>%
  mutate(.metric = str_to_upper(.metric))
print(final_comparison_table2)


################################################
# Tuned Prophet Model Forecast into 2025 - 2026
################################################

# Re-train the Prophet model on all the data.
final_prophet_model <- prophet(
  prophet_df, 
  changepoint.prior.scale = 0.07,  
  changepoint.range = 0.95,        
  n.changepoints = 40 
)

# Create the future dataframe.
future_forecast_df <- make_future_dataframe(
  final_prophet_model,
  periods = 8, 
  freq = "quarter"
)

# Generate the future forecast.
final_forecast <- predict(final_prophet_model, future_forecast_df)

# 2025 - 2026 forecast predictions.
tail(final_forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')], 8)

# Plot for 2025 - 2026 forecast.
plot(final_prophet_model, final_forecast) +
  labs(title = "Final Galunggong Price Forecast (into 2026)", x = "Date", y = "Price (PHP/kg)")

future_predictions <- select(final_forecast, ds, yhat_lower, yhat_upper, yhat)
print(tail(future_predictions, 8))

