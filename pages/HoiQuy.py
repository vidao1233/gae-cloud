import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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
st.title("Hồi quy")
st.sidebar.markdown("# Hồi quy")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',["Tính toán", "Nghiệm Scikit-learn", "Hồi quy bậc 2", "Sự nhiễu", "Khắc phục"]) 


if(app_mode == 'Tính toán'):
    st.title("Bài 01: Tính toán")
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
    mot = np.ones((1, 13), dtype = np.int32)
    X_bar = np.vstack((mot, X))
    X_bar_T = X_bar.T
    A = np.matmul(X_bar, X_bar_T)
    y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    b = np.matmul(X_bar, y)
    A_inv = np.linalg.pinv(A)
    w = np.matmul(A_inv, b)
    w_0, w_1 = w[0,0], w[1,0]
    x =[]
    e = []
    for i in range(0,13):
        x.append(X[0,i])
    for i in range(0,13):
        e.append(y.T[0,i])
    df = pd.DataFrame({
        'Chiều cao(cm)':x,
        'Cân nặng tương ứng(kg)':e,
    })
    st.header('Bảng số liệu tham khảo:')
    st.write(df.T)
    
    x1 = X[0, 0]
    y1 = x1*w[1, 0] + w[0, 0]
    x2 = X[0, -1]
    y2 = x2*w[1, 0] + w[0, 0]
    plt.plot(X, y.T, 'ro')
    plt.plot([x1, x2], [y1, y2])  
    st.write("Nghiệm tính theo công thức: w_0 = ",w_0," w_1 = ", w_1)
    st.header('Đồ thị')  
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    hight = st.number_input('Chiều cao:')
    btn_giai = st.button('Kết quả')
    if btn_giai: 
        st.session_state.btn_giai = True
        ket_qua = hight*w[1, 0] + w[0, 0]
        st.write('Chiều cao là %d thì cân nặng là: %d' % (hight, ket_qua))
elif(app_mode == 'Nghiệm Scikit-learn'):
    st.title("Bài 02: Nghiệm theo thư viện scikit - learn")
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    X = X[:,0]
    fig, ax = plt.subplots()
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig)
    # Compare two results
    st.write("Nghiệm theo thư viện scikit-learn : w_0 = ", regr.intercept_, "w_1 = ", regr.coef_[0])
elif(app_mode == 'Hồi quy bậc 2'):
    st.title("Bài 03: Hồi quy bậc 2")
    st.header("Giới thiệu")
    st.write("Hàm số y ≈ f(x)= ((w)^T)x là một hàm tuyến tính theo cả w và x. Trên thực tế, Linear Regression có thể áp dụng cho các mô hình chỉ cần tuyến tính theo w")
    st.write("Ví dụ: y ≈ w[0]*x^2 + w[1]*x + w[2] cũng là một hàm tuyến tính theo w do đó có thể sử dụng Hồi quy để giải bài toán này")
    st.header("Ví dụ")
    st.write("Thực hiện ví dụ: y ≈ w[0]*x^2 + w[1]*x + w[2] với các dữ liệu được lấy từ thư viện:") 
    st.button("linear_model.LinearRegression()")
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    X2 = X**2
    X_poly = np.hstack((X, X2))
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, y)
    st.write("w[0] =",lin_reg.coef_[0,1],", w[1] =",lin_reg.coef_[0,0], " và w[2] =",lin_reg.intercept_[0])
    st.write("=> Hàm tuyến tính y ≈ ",lin_reg.coef_[0,1],"*x^2 + ",lin_reg.coef_[0,0],"*x +",lin_reg.intercept_[0])
    a = lin_reg.intercept_[0]
    b = lin_reg.coef_[0,0]
    c = lin_reg.coef_[0,1]
    x_ve = np.linspace(-3,3,m)
    y_ve = a + b*x_ve + c*x_ve**2
    fig, ax = plt.subplots()
    plt.plot(X, y, 'o')
    plt.plot(x_ve, y_ve, 'r')

    # Tinh sai so
    loss = 0 
    for i in range(0, m):
        y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
        sai_so = (y[i] - y_mu)**2 
        loss = loss + sai_so
    loss = loss/(2*m)
    st.write('Với sai số tính được là loss = %.6f' % loss)

    # Tinh sai so cua scikit-learn
    y_train_predict = lin_reg.predict(X_poly)
    # print(y_train_predict)
    sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
    st.write('Sai số phương sai trung bình: %.6f' % (sai_so_binh_phuong_trung_binh/2)," cũng bằng với sai số loss tính được ở trên")
    st.pyplot(fig)
    st.latex("Example")
elif(app_mode == 'Sự nhiễu'):
    st.title("Bài 04: Hạn chế của Hồi Quy")
    #Hạm chế của hồi quy
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
    regr = linear_model.LinearRegression()
    regr.fit(X, y) # in scikit-learn, each sample is one row
    fig, ax = plt.subplots()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = regr.coef_[0]
    b = regr.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.header("Giới thiệu")
    st.write("Hồi quy rất nhạy cảm với sensitive to noise (sự nhiễu)")
    st.header("Ví dụ")
    st.write("Sự nhiễu của 1 cặp dữ liệu trong model (150cm, 90kg)")
    st.pyplot(fig)
    st.latex("Example")
else:
    st.title("Bài 05: Cách khắc phục")
    
    # height (cm), input data, each row is a data point
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

    huber_reg = linear_model.HuberRegressor()
    huber_reg.fit(X, y) # in scikit-learn, each sample is one row
    # Compare two results
    st.write("scikit-learn’s solution : w_1 = ", huber_reg.coef_[0], "w_0 = ", huber_reg.intercept_)
    fig, ax = plt.subplots()
    X = X[:,0]
    plt.plot(X, y, 'ro')
    a = huber_reg.coef_[0]
    b = huber_reg.intercept_
    x1 = X[0]
    y1 = a*x1 + b
    x2 = X[12]
    y2 = a*x2 + b
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)
    st.pyplot(fig)
