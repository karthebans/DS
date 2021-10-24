# Importing the libraries
import pandas as pd
import pmdarima as pm
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

#Declaring the variables
attr1 = "Date"
attr2 = 'Headend_Id'
attr3 = "Enduser_count"

# declaring the ratio for train and test split
split_ratio = 0.80
n_periods = 1830 # (366 days X 5 years)

output_csv = "GEA Forecasting SARIMA2.csv"

#SQL connection details
host_name="10.54.8.77"
user_name="root"
user_password="MySQL@123"
Port_no = "3306"
db_name = "CMA"

no_of_profiles = 2
num_of_threads = 11

# get data from database
# engine = create_engine("mysql+pymysql://" + user_name + ":" + user_password + "@" + host_name + "/" + db_name)
# fetch_all_query = "SELECT * FROM HEADEND_VIEW_DAILY"
# df = pd.read_sql(fetch_all_query, con=engine)

# Data preparation for the model
try:
    df = pd.read_csv("GEM Headedend.csv")     # old table
    df = df.loc[:, ['CREATED_DATE', 'HEADEND_ID', 'NPM_PRODUCT_TYPE', 'CUSTOMER_COUNT']]   # old table
    # df = df.loc[:, ['CREATED_DATE', 'HEADEND_ID', 'NPM_PRODUCT_TYPE', 'CUSTOMER_COUNT']]
    columns = {'CREATED_DATE' : "Date", 'HEADEND_ID': 'Headend_Id', 'NPM_PRODUCT_TYPE':"Product",
                     'CUSTOMER_COUNT':"Enduser_count"}   # old table
    # columns = {'CREATED_DATE' : "Date", 'HEADEND_ID': 'Headend_Id', 'NPM_PRODUCT_TYPE':"Product",
    #            'CUSTOMER_COUNT':"Enduser_count"}
    df.rename(columns=columns, inplace=True)
    a = df['Headend_Id'] + '&' + df['Product']
    df['Headend_Id'] = a
    df.dropna(inplace=True)
    df.drop(columns=[ 'Product'], inplace=True)    
    headend_list = df['Headend_Id'].unique()
    
except Exception as e:
         print(e)

# FUNCTION TO PREPARE THE TRAIN AND TEST DATA
def prepareheadendData(headend):
    X = df[df[attr2] == headend]    
    X = X.sort_values(by=attr1)
    X = X.set_index(attr1)
    X_train = X[:int(X.shape[0]*split_ratio)]
    X_test = X[int(X.shape[0]*split_ratio):]
    return X_train, X_test

#FUNCTION TO FIT THE MODEL USING ALGORITHM
def forecastAll(X_train,X_test,headend):
    smodel = pm.auto_arima(X_train, start_p=0, start_q=0,d=0, 
                                           max_p=5, max_q=5, max_d=5,  
                                           start_P=0, D=0,
                                           max_P=2, max_Q=2, max_D=1,  m=0,
                                           seasonal=False,test='adf',trace=False,
                                           error_action='ignore',
                                           suppress_warnings=True,stepwise=True, n_jobs = -1)
    
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    date = pd.date_range(X_test.index[-1], periods = n_periods, freq='D')
    
    fitted_series = pd.Series(fitted)
    lower_series = pd.Series(confint[:, 0])
    upper_series = pd.Series(confint[:, 1])
    
    output_fc = pd.DataFrame({'FORECAST_DATE': date, 'FORECAST' : fitted_series, 'LOWER_CONFLICT': lower_series, 
                              'UPPER_CONFLICT': upper_series})
    output_fc[attr2] = headend
    return output_fc

#FUNCTION TO FORECAST
def forecast(headend):    
    X_train, X_test = prepareheadendData(headend)
    if (len(X_train)> 5):          
        df_forecast = forecastAll(X_train,X_test,attr3)
        df_forecast1 = df_forecast['Headend_Id'].str.split('&',expand = True)
        df_forecast = pd.concat([df_forecast, df_forecast1], axis=1)
        df_forecast.drop(columns= ['Headend_Id'], inplace=True)
        df_forecast.rename({0 : 'HEADEND_ID' , 1 : 'PRODUCT'}, axis=1, inplace=True)        
        df_forecast['Date'] = pd.to_datetime(df['Date']).dt.floor('D').dt.strftime('%Y-%m-%d %H:%M:%S')
        df_forecast = df_forecast[['FORECAST_DATE', 'HEADEND_ID','PRODUCT', 'FORECAST', 
                                   'LOWER_CONFLICT', 'UPPER_CONFLICT']]
        df_forecast["ALGORITHM"] = "SARIMA"
# O/P to CSV
        df_forecast.to_csv(output_csv, mode='a', header=True, index=False)
# O/P to SQL
        # engine = create_engine("mysql+pymysql://" + user_name + ":" + user_password + "@" + host_name + "/" + db_name)
        # df_forecast.to_sql('GEA_FORECASTING_ANALYTICS_DAILY', con = engine, if_exists = 'replace',
        #                    index = False, chunksize = 1000)

        return "Completed for profile: " + headend + " Test: " +str(len(X_train)) +" Train: "+str(len(X_test))
    else:
        print(headend +" does not have sufficient data to train")

# executing the model with each headend and product
# try:
#     from concurrent.futures import ThreadPoolExecutor
#     executor = ThreadPoolExecutor(num_of_threads)

#     counter = 0
#     no_of_profiles = len(headend_list)
#     l = ['BAAALI&FTTC_LT_100','BAABAU&FTTC_LT_100', 'BAABDI&FTTP_GE_500']
#     for headend in headend_list:
#         if counter < no_of_profiles:
#             if headend not in l:
#                 future = executor.submit(forecast, (headend))
#                 print(future.result())
#                 counter = counter + 1
#             else:
#                 pass
#         else:
#             break
# except Exception as e:
#     print(e)



import concurrent
with concurrent.futures.ThreadPoolExecutor(3) as executor:
    results = executor.map(forecast, headend_list[1:8])
    
    for result in results:
        print(result)