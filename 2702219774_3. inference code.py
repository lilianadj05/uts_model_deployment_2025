#Liliana Djaja Witama
#2702219774

#UTS Model Deployment No 3


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import random


model = joblib.load('model_ranfor_oop.pkl')
encoder = joblib.load('encoders_oop.pkl')


def main():
    st.title('Hotel Booking Cancellation Prediction')

    #random booking ID
    def generate_booking_id():
        return f"BK{random.randint(1000, 9999)}"
    
    #inisialisasi session_state
    if 'new_booking_started' not in st.session_state:
        st.session_state.new_booking_started = False

    if 'arrival_month' not in st.session_state:
        st.session_state.arrival_month = 1    #default januari
    if 'prev_arrival_date' not in st.session_state:
        st.session_state.prev_arrival_date = None
        
    #button New Booking
    if st.button('New Booking'):
        #bersihkan semua session state kecuali model dan encoder
        keys_to_keep = ['model', 'encoder']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        # Inisialisasi ulang state yang diperlukan
        st.session_state.new_booking_started = True
        st.session_state.booking_id = generate_booking_id()
        st.session_state.pop('avg_price_per_room', None)    #reset harga
        
        
    #simpan state button New Booking
    if st.session_state.new_booking_started:
        booking_id = st.session_state.booking_id
        st.write(f"New Booking ID: {booking_id}")


        no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
        no_of_children = st.number_input("Number of Children", min_value=0, value=0)
        no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
        no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
        
        
        type_of_meal_plan = st.selectbox(
            "Meal Plan",
            ["-- Select Meal Plan --", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"]
        )
        #untuk value Not Selected
        if type_of_meal_plan == "-- Select Meal Plan --":
            type_of_meal_plan = "Not Selected"
        

        required_car_parking_space = st.radio("Car Parking Required", ["Yes", "No"])
        if required_car_parking_space == "-Yes":
            required_car_parking_space = 1
        else:
            required_car_parking_space = 0


        room_type_reserved = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
        

        #tanggal-tanggal dan menghitung lead_time
        max_booking_date = datetime.date(2018, 12, 31)
        booking_date = st.date_input(
            "Booking Date",
            value=datetime.date(2017, 1, 1),
            max_value=max_booking_date
        )
        arrival_date = st.date_input(
            "Arrival Date",
            value=booking_date + datetime.timedelta(days=7),
            min_value=booking_date
        )

        if arrival_date <= booking_date:
            st.warning("Arrival date must be after booking date.")
            lead_time = None
        elif arrival_date.year < 2017 or arrival_date.year > 2018:
            st.warning("Arrival year must be 2017 or 2018.")
            lead_time = None
        else:
            lead_time = (arrival_date - booking_date).days
            st.success(f"Lead Time (days): {lead_time}")

        if lead_time is not None:
            arrival_year = arrival_date.year
            arrival_month = arrival_date.month
            arrival_day = arrival_date.day
            st.session_state.arrival_month = arrival_month
            

        market_segment_type = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
        

        repeated_guest = st.radio("Repeated Guest", ["Yes", "No"])
        if repeated_guest == "-Yes":
            repeated_guest = 1
        else:
            repeated_guest = 0
        

        no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("Previous Non-Canceled", min_value=0, value=0)
        

        #avg_price_per_room berdasarkan arrival_month
        if 'avg_price_per_room' not in st.session_state or st.session_state.arrival_month != arrival_month or st.session_state.prev_arrival_date != arrival_date:
            month_price_range = {
                1: (100, 250),    
                2: (60, 160),    
                3: (80, 200),    
                4: (100, 250),   
                5: (120, 300),   
                6: (150, 400),  
                7: (200, 500),   
                8: (200, 550),   
                9: (130, 350),   
                10: (100, 250),  
                11: (70, 200),   
                12: (200, 500),   
            }
            min_price, max_price = month_price_range[arrival_month]
            st.session_state.avg_price_per_room = round(random.uniform(min_price, max_price), 0)
            st.session_state.prev_arrival_date = arrival_date
        avg_price_per_room = st.session_state.avg_price_per_room
        st.write(f"Average Price per Room: {avg_price_per_room}")

        
        no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)
        
        
        user_input = {
            'no_of_adults': int(no_of_adults),
            'no_of_children': int(no_of_children),
            'no_of_weekend_nights': int(no_of_weekend_nights),
            'no_of_week_nights': int(no_of_week_nights),
            'type_of_meal_plan': [type_of_meal_plan],
            'required_car_parking_space': int(required_car_parking_space),
            'room_type_reserved': [room_type_reserved],
            'lead_time': int(lead_time),
            'arrival_year': [arrival_year],
            'arrival_month': int(arrival_month),
            'arrival_date': int(arrival_day),
            'market_segment_type': [market_segment_type],
            'repeated_guest': int(repeated_guest),
            'no_of_previous_cancellations': int(no_of_previous_cancellations),
            'no_of_previous_bookings_not_canceled': int(no_of_previous_bookings_not_canceled),
            'avg_price_per_room': int(avg_price_per_room),
            'no_of_special_requests': int(no_of_special_requests),
        }

        df = pd.DataFrame(user_input)

        df['avg_price_per_room'] = df['avg_price_per_room'].fillna(df['avg_price_per_room'].median())                      
        df['type_of_meal_plan'] = df['type_of_meal_plan'].fillna(df['type_of_meal_plan'].mode()[0])
        df['required_car_parking_space'] = df['required_car_parking_space'].fillna(df['required_car_parking_space'].mode()[0])

        df = df[df['no_of_adults'] != 0]

        to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_year']
        for col in to_encode:
            enc = encoder[col]
            df[col] = enc.transform(df[col])
        
        if st.button('Make Prediction'):
            features = df      
            result = make_prediction(features)
            st.success(f'Booking Status: {result}')


def make_prediction(features):
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()
