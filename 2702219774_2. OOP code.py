#Liliana Djaja Witama
#2702219774

#UTS Model Deployment No 2

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class Hotel:
    def __init__(self, train_path):
        self.train_path = train_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=SEED)
        self.train_data = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
    
    def head(self, n=5):
        print(self.train_data.head(n))
    
    def information(self):
        self.train_data.info()
    
    def duplicated(self):
        return self.train_data.duplicated().sum()

    def null_value(self):
        return self.train_data.isna().sum()
    
    def num_fillna(self, column):
        self.train_data[column] = self.train_data[column].fillna(self.train_data[column].median())                      
    
    def cat_fillna(self, column):
        self.train_data[column] = self.train_data[column].fillna(self.train_data[column].mode()[0])

    def drop_column(self, column):
        self.train_data.drop(column, axis=1, inplace=True)
    
    def drop_row(self, column, value):
        self.train_data = self.train_data[self.train_data[column] != value]

    def separate_num_cat(self):
        num_cols = []
        cat_cols = []
        for i in self.train_data.columns:
            if 'int' in str(self.train_data[i].dtype) or 'float' in str(self.train_data[i].dtype):
                num_cols.append(i)
            else:
                cat_cols.append(i)
        return num_cols, cat_cols

    def remove_append(self, column, num_cols, cat_cols):
        for i in column:
            num_cols.remove(i)
            cat_cols.append(i)

    def num_description(self, column):
        return self.train_data[column].describe()
    
    def unique_values(self, column):
        for i in column:
            print(self.train_data[i].value_counts(), '\n')
    
    def look_outliers_distribution(self, column):
        for col in column:
            plt.figure(figsize=(10,3))

            plt.subplot(1,2,1)
            sns.histplot(self.train_data[col], bins=30)
            plt.title(f'Histogram of {col}')

            plt.subplot(1,2,2)
            sns.boxplot(y=self.train_data[col])
            plt.title(f'Boxplot of {col}')

            plt.show()

            print(f'skewness: {self.train_data[col].skew()}')
            print(f'kurtosis: {self.train_data[col].kurt()}')

    def shapiro_wilk(self, column):
        for i in column:
            print(i, shapiro(self.train_data[i]))

    def split_dataset(self, target):
        x = self.train_data.drop([target], axis=1)
        y = self.train_data[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
        return x_train, x_test, y_train, y_test


    def encode(self, x_train, x_test, column):
        encoders = {}
        for i in column:
            enc = LabelEncoder()
            x_train[i] = enc.fit_transform(x_train[i])
            x_test[i] = enc.transform(x_test[i])
            encoders[i] = enc

        with open('encoders_oop.pkl', 'wb') as f:
            pickle.dump(encoders, f)

        return x_train, x_test
    

    def train_pred_eval(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        print(classification_report(y_test, y_pred))

    def save_model(self):
        with open('model_ranfor_oop.pkl', 'wb') as file:
            pickle.dump(self.model, file)


hotel = Hotel('C:/Users/lenovo/OneDrive - Bina Nusantara/SEMESTER 4/Model Deployment/uts/Dataset_B_hotel.csv')
hotel.load_data()
hotel.head()

hotel.information()

print(hotel.duplicated())
print(hotel.null_value())

hotel.num_fillna('avg_price_per_room')
hotel.cat_fillna('type_of_meal_plan')
hotel.cat_fillna('required_car_parking_space')
print(hotel.null_value())

hotel.drop_column('Booking_ID')

num_cols, cat_cols = hotel.separate_num_cat()
to_cat = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'required_car_parking_space', 
          'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 
          'no_of_previous_bookings_not_canceled', 'no_of_special_requests']
hotel.remove_append(to_cat, num_cols, cat_cols)
print('numerical: ', num_cols)
print('categorical: ', cat_cols)

hotel.num_description(num_cols)

hotel.unique_values(cat_cols)
hotel.drop_row('no_of_adults', 0)

hotel.look_outliers_distribution(num_cols)
hotel.shapiro_wilk(num_cols)

x_train, x_test, y_train, y_test = hotel.split_dataset('booking_status')

to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_year']
x_train, x_test = hotel.encode(x_train, x_test, to_encode)

hotel.train_pred_eval(x_train, x_test, y_train, y_test)
hotel.save_model()