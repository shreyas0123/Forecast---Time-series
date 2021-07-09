################################ problem1 #########################################
library(readr)

#load the dataset
Airlines_data <- readxl::read_excel("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\Airlines Data.xlsx") 
View(Airlines_data) # Seasonality 12 months
colnames(Airlines_data)
# Pre Processing
# input t
Airlines_data["t"] <- c(1:96)
View(Airlines_data)

Airlines_data["t_square"] <- Airlines_data["t"] * Airlines_data["t"]
Airlines_data["log_Passengers"] <- log(Airlines_data["Passengers"])


# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

Airlines_data_Passengers <- cbind(Airlines_data, X)
colnames(Airlines_data_Passengers)

View(Airlines_data_Passengers)
## Pre-processing completed

attach(Airlines_data_Passengers)

# partitioning
train <- Airlines_data_Passengers[1:77, ]
test <- Airlines_data_Passengers[78:96, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Passengers ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Passengers - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_Passengers ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Passengers - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Passengers ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Passengers-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Passengers - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Passengers - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Passengers ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Passengers - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(Airlines_data_Passengers, file = "AirlinesPassengers.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data = Airlines_data_Passengers)
summary(Add_sea_Quad_model_final)

####################### Predicting new data #############################
install.packages("xlsx")
library(xlsx)
test_data <- read.xlsx("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\New_Airlines_data.xlsx", 1)
View(test_data)

# Pre Processing
# input t
test_data["t"] <- c(97:156)
View(test_data)

test_data["t_square"] <- test_data["t"] * test_data["t"]

# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

test_data <- cbind(test_data, X)
colnames(test_data)

pred_new <- predict(Add_sea_Quad_model_final, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)
pred_new$fit
plot(Add_sea_Quad_model_final)

############
# ACF plot
acf(Add_sea_Quad_model_final$residuals, lag.max = 12) # take all residual value of the model built & plot ACF plot

A <- arima(Add_sea_Quad_model_final$residuals, order = c(1, 0, 0))
summary(A)
A$coef
A$residuals

ARerrors <- A$residuals

acf(ARerrors, lag.max = 12)

# predicting next 12 month errors using arima(order=c(1,0,0))
install.packages("forecast")
library(forecast)
errors_12 <- forecast(A, h = 12)
?forecast
View(errors_12)

future_errors <- data.frame(errors_12)
class(future_errors)
future_errors <- future_errors$Point.Forecast

# predicted values for new data + future error values 

predicted_new_values <- pred_new$fit + future_errors

write.csv(predicted_new_values, file = "predicted_new_values.csv", row.names = F)
getwd()

################################ problem2 ##############################################
library(readr)

#load cocacola data 
cocacola_data <- readxl::read_excel("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\CocaCola_Sales_Rawdata.xlsx") 
View(cocacola_data) # Seasonality 12 months
colnames(cocacola_data)

# Pre Processing
# input t
cocacola_data["t"] <- c(1:42)
View(cocacola_data)

cocacola_data["t_square"] <- cocacola_data["t"] * cocacola_data["t"]
cocacola_data["log_Sales"] <- log(cocacola_data["Sales"])


# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 42), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

cocacola_data_Sales <- cbind(cocacola_data, X)
colnames(cocacola_data_Sales)

View(cocacola_data_Sales)
## Pre-processing completed

attach(cocacola_data_Sales)

# partitioning
train <- cocacola_data_Sales[1:33, ]
test <- cocacola_data_Sales[34:42, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_Sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(cocacola_data_Sales, file = "cocacola_Sales.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Sales ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data = cocacola_data_Sales)
summary(Add_sea_Quad_model_final)

####################### Predicting new data #############################
library(xlsx)
test_data <- read.xlsx("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\New_Cocacola.xlsx", 1)
View(test_data)

# Pre Processing
# input t
test_data["t"] <- c(43:52)
View(test_data)

test_data["t_square"] <- test_data["t"] * test_data["t"]

# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 10), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

test_data <- cbind(test_data, X)
colnames(test_data)

pred_new <- predict(Add_sea_Quad_model_final, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)
pred_new$fit
plot(Add_sea_Quad_model_final)


############
# ACF plot
acf(Add_sea_Quad_model_final$residuals, lag.max = 12) # take all residual value of the model built & plot ACF plot

A <- arima(Add_sea_Quad_model_final$residuals, order = c(1, 0, 0))
summary(A)
A$coef
A$residuals

ARerrors <- A$residuals

acf(ARerrors, lag.max = 12)

# predicting next 12 month errors using arima(order=c(1,0,0))

library(forecast)
errors_12 <- forecast(A, h = 12)
?forecast
View(errors_12)

future_errors <- data.frame(errors_12)
class(future_errors)
future_errors <- future_errors$Point.Forecast

# predicted values for new data + future error values 

predicted_new_values <- pred_new$fit + future_errors

write.csv(predicted_new_values, file = "predicted_new_values.csv", row.names = F)
getwd()

################################ problem3 ###########################################
library(readr)

#load PlasticSales data 
PlasticSales_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\PlasticSales.csv") 
View(PlasticSales_data) # Seasonality 12 months
colnames(PlasticSales_data)

# Pre Processing
# input t
PlasticSales_data["t"] <- c(1:60)
View(PlasticSales_data)

PlasticSales_data["t_square"] <- PlasticSales_data["t"] * PlasticSales_data["t"]
PlasticSales_data["log_Sales"] <- log(PlasticSales_data["Sales"])


# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

PlasticSales_data <- cbind(PlasticSales_data, X)
colnames(PlasticSales_data)

View(cocacola_data_Sales)
## Pre-processing completed

attach(PlasticSales_data)

# partitioning
train <- PlasticSales_data[1:48, ]
test <- PlasticSales_data[49:60, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_Sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(PlasticSales_data, file = "PlasticSales.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Sales ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data = PlasticSales_data)
summary(Add_sea_Quad_model_final)

####################### Predicting new data #############################
library(xlsx)
test_data <- readxl::read_excel("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\new_pastic.xlsx", 1)
View(test_data)

# Pre Processing
# input t
test_data["t"] <- c(61:72)
View(test_data)

test_data["t_square"] <- test_data["t"] * test_data["t"]

# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 12), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

test_data <- cbind(test_data, X)
colnames(test_data)

pred_new <- predict(Add_sea_Quad_model_final, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)
pred_new$fit
plot(Add_sea_Quad_model_final)


############
# ACF plot
acf(Add_sea_Quad_model_final$residuals, lag.max = 12) # take all residual value of the model built & plot ACF plot

A <- arima(Add_sea_Quad_model_final$residuals, order = c(1, 0, 0))
summary(A)
A$coef
A$residuals

ARerrors <- A$residuals

acf(ARerrors, lag.max = 12)

# predicting next 12 month errors using arima(order=c(1,0,0))

library(forecast)
errors_12 <- forecast(A, h = 12)
?forecast
View(errors_12)

future_errors <- data.frame(errors_12)
class(future_errors)
future_errors <- future_errors$Point.Forecast

# predicted values for new data + future error values 

predicted_new_values <- pred_new$fit + future_errors

write.csv(predicted_new_values, file = "predicted_new_values.csv", row.names = F)
getwd()

##################################### problem4 ###################################################
library(readr)

#load PlasticSales data 
solar_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\solarpower_cumuldaybyday2.csv") 
View(solar_data) # Seasonality 12 months
colnames(solar_data)

# Pre Processing
# input t
solar_data["t"] <- c(1:2558)
View(solar_data)

solar_data["t_square"] <- solar_data["t"] * solar_data["t"]
solar_data["log_cum_power"] <- log(solar_data["cum_power"])


# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 2558), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

solar_data <- cbind(solar_data, X)
colnames(solar_data)

View(solar_data)
## Pre-processing completed

attach(solar_data)

# partitioning
train <- solar_data[1:2047, ]
test <- solar_data[2048:2558, ]

########################### LINEAR MODEL #############################

linear_model <- lm(cum_power ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$cum_power - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_cum_power ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$cum_power - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(cum_power ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$cum_power-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(cum_power ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$cum_power - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_cum_power ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$cum_power - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(cum_power ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$cum_power - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# LINEAR MODEL has least RMSE value

write.csv(solar_data, file = "solar.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(cum_power ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data =solar_data)
summary(Add_sea_Quad_model_final)

####################### Predicting new datas #############################
library(xlsx)
test_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Forecasting-Time Series\\solarpower_cumuldaybyday2.csv", 1)
View(test_data)
pred_new <- predict(Add_sea_Quad_model_final, newdata = test_data, interval = 'predict')
pred_new <- as.data.frame(pred_new)
pred_new$fit
plot(Add_sea_Quad_model_final)

############################################### END #####################################################


