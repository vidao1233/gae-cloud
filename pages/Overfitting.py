import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://image.winudf.com/v2/image/Y29tLnNpbXBsZWRyb2lkLndhbGxwYXBlcmdyYWRpZW50YmFja2dyb3VuZF9zY3JlZW5fMF8xNTI2OTY5MDEyXzAwMA/screen-0.jpg?fakeurl=1&type=.webp');
background-size: 80%;
background-position: right;
background-repeat: initial;
background-attachment: fixed;
background-repeat: no-repeat;
}"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Overfitting")
st.sidebar.markdown("# Overfitting")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',["Bài 01a", "Bài 01b", "Bài 01c", "Bài 01d"]) 



if(app_mode=='Bài 01a'):
    st.title('Bài 1a')
    np.random.seed(100)

    N = 30
    X = np.random.rand(N, 1)*5
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

    poly_features = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    N_test = 20 
    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    lin_reg.fit(X_poly, y)
    
    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)

    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)

    fig, ax = plt.subplots()
    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 2')
    
    st.pyplot(fig)
    st.write('Sai số bình phương trung bình - tập training: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.write('Sai số bình phương trung bình - tập test: %.6f' % (sai_so_binh_phuong_trung_binh/2))
elif(app_mode=='Bài 01b'):
    st.title('Bài 1b')
      
    np.random.seed(100)
    N = 30
    X = np.random.rand(N, 1)*5
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

    poly_features = PolynomialFeatures(degree=4, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    N_test = 20 
    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    lin_reg.fit(X_poly, y)

    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)

    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)

    fig, ax = plt.subplots()
    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 4')
    st.pyplot(fig)
    st.write('Sai số bình phương trung bình - tập training: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.write('Sai số bình phương trung bình - tập test: %.6f' % (sai_so_binh_phuong_trung_binh/2))
elif(app_mode=='Bài 01c'):
    st.title('Bài 1c')
    
    np.random.seed(100)
    N = 30
    X = np.random.rand(N, 1)*5
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

    poly_features = PolynomialFeatures(degree=8, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    N_test = 20 
    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
    X_poly_test = poly_features.fit_transform(X_test)

    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    lin_reg.fit(X_poly, y)

    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)

    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)


    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    
    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)

    fig, ax = plt.subplots()
    plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])
    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 8')
    st.pyplot(fig)
    st.write(np.min(y_test), np.max(y) + 100)
    st.write('Sai số bình phương trung bình - tập training: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.write('Sai số bình phương trung bình - tập test: %.6f' % (sai_so_binh_phuong_trung_binh/2))
else:
    st.title('Bài 1d')
    np.random.seed(100)

    N = 30
    X = np.random.rand(N, 1)*5
    y = 3*(X -2) * (X - 3)*(X-4) +  10*np.random.randn(N, 1)

    poly_features = PolynomialFeatures(degree=8, include_bias=True)
    X_poly = poly_features.fit_transform(X)

    N_test = 20 

    X_test = (np.random.rand(N_test,1) - 1/8) *10
    y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)
    X_poly_test = poly_features.fit_transform(X_test)
    lin_reg = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    lin_reg.fit(X_poly, y)

    x_ve = np.linspace(-2, 10, 100)
    y_ve = np.zeros(100, dtype = np.float64)
    y_real = np.zeros(100, dtype = np.float64)
    x_ve_poly = poly_features.fit_transform(np.array([x_ve]).T)
    y_ve = np.matmul(x_ve_poly, lin_reg.coef_.T)

    for i in range(0, 100):
        y_real[i] = 3*(x_ve[i]-2) * (x_ve[i]-3)*(x_ve[i]-4)


    
    fig, ax = plt.subplots()
    plt.axis([-4, 10, np.min(y_test) - 100, np.max(y) + 100])

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)

    # Tinh sai so cua scikit-learn
    y_test_predict = lin_reg.predict(X_poly_test)
    # print(y_test_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y_test, y_test_predict)

    plt.plot(X,y, 'ro')
    plt.plot(X_test,y_test, 's')
    plt.plot(x_ve, y_ve, 'b')
    plt.plot(x_ve, y_real, '--')
    plt.title('Hồi quy đa thức bậc 16')

    st.pyplot(fig)
    st.write(np.min(y_test), np.max(y) + 100)
    st.write('Sai số bình phương trung bình - tập training: %.6f' % (sai_so_binh_phuong_trung_binh/2))
    st.write('Sai số bình phương trung bình - tập test: %.6f' % (sai_so_binh_phuong_trung_binh/2))