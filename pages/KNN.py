import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from PIL import ImageTk, Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2
import joblib
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import io
from sklearn import datasets
from skimage import exposure
import imutils
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
st.title("K-Nearest Neighbors")
st.sidebar.markdown("# K-Nearest Neighbors")
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',["Bai01", "Bai02", "Bai03", "Bai04", "Bai08"]) 
if app_mode=='Bai01':
    st.header('BÀI 01')
    N = 150
    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(N, centers=np.array(centers), random_state=1)
    nhom_0 = []
    nhom_1 = []
    nhom_2 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])
        else:
            nhom_2.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)
    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)
    plt.plot(nhom_2[:,0], nhom_2[:,1], 'ob', markersize = 2)
    plt.legend(['Nhóm 0', 'Nhóm 1', 'Nhóm 2'])
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    res = train_test_split(data, labels, 
                            train_size=0.8,
                            test_size=0.2,
                            random_state=1)
    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels) 
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    predicted = knn.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.latex('Biểu đồ tham khảo- Sai số: %.0f%%' % sai_so)
    df = pd.DataFrame((*nhom_0, *nhom_1, *nhom_2), columns=["x", "y"])
    st.expander("Bảng dữ liệu tham khảo ứng với các điểm trên biểu đồ ").write(df.T)

    
    x = st.number_input("Nhập x")
    y = st.number_input("Nhập y")
    btnKQ = st.button("Kết quả:")
    if btnKQ:
        my_test = np.array([[x, y]])
        st.write("Kết quả cặp [x,y] thuộc nhãn :")
        ket_qua = st.text(knn.predict(my_test))
elif (app_mode=='Bai02'):
    st.title("BÀI 02") 
    # take the MNIST data and construct the training and testing split, using 75% of the
    # data for training and 25% for testing
    mnist = datasets.load_digits()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
        mnist.target, test_size=0.25, random_state=42)

    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)

    st.write("training data points: ", len(trainLabels))
    st.write("validation data points: ", len(valLabels))
    st.write("testing data points: ", len(testLabels))

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    st.write("accuracy = %.2f%%" % (score * 100))

    # loop over a few random digits
    btnD = st.button("Data and Prediction:")
    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
        # grab the image and classify it
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels so we can see it better
        image = image.reshape((8, 8)).astype("uint8")

        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
        # show the prediction
        if btnD:
            st.image(image, clamp=True)
            st.write("I think that digit is: {}".format(prediction))
elif(app_mode=='Bai03'):
    st.header("BÀI 03")
    uploaded_file = st.file_uploader("OPEN MODEL",type=['pkl'])
    if uploaded_file is not None:
        mnist = keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        index = None
        knn = joblib.load(uploaded_file)
        col1, col2 = st.columns([15,20])
        mnist = keras.datasets.mnist 
        digit = np.zeros((10*28,10*28), np.uint8)
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
        index = np.random.randint(0, 9999, 100)
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                k = k + 1
        with col1:
            st.latex("IMAGE")
            cv2.imwrite('digit.jpg',digit)
            image = Image.open('digit.jpg')
            st.image(image, caption='IMAGE')
            sample = np.zeros((100,28,28), np.uint8)
            for i in range(0, 100):
                sample[i] = X_test[index[i]]
            RESHAPED = 784
            sample = sample.reshape(100, RESHAPED) 
            predicted = knn.predict(sample)
            k = 0
            with col2:
                st.latex("Ket qua nhan dang")
                for x in range(0, 5):
                    ketqua = ''
                    for y in range(0, 20):
                        ketqua = ketqua + '%3d' % (predicted[k])
                        k = k + 1
                    st.subheader(ketqua )
elif(app_mode=='Bai04'):
    st.header("BÀI 04")
    uploaded_file = st.file_uploader("OPEN FILE",type=['pkl'])
    if uploaded_file is not None:
        mnist = keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        index = None
        knn = joblib.load(uploaded_file)
        btn1 = st.button('Create Digit and Recognition')
        if btn1:
            col1,col2 = st.columns([15,20])
            index = np.random.randint(0, 9999, 100)
            digit = np.zeros((10*28,10*28), np.uint8)
            k = 0
            for x in range(0, 10):
                for y in range(0, 10):
                        digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                        k = k + 1  
            with col1:
                st.latex("IMAGE")
                st.write()
                st.write()
                cv2.imwrite('digit.jpg', digit)
                image = Image.open('digit.jpg')
                st.image(image, caption='IMAGE')
                sample = np.zeros((100,28,28), np.uint8)
                for i in range(0, 100):
                    sample[i] = X_test[index[i]]
                    
                RESHAPED = 784
                sample = sample.reshape(100, RESHAPED) 
                predicted = knn.predict(sample)
                k = 0
                with col2:
                    st.latex("Ket qua nhan dang")
                    for x in range(0, 5):
                        ketqua = ''
                        for y in range(0, 20):
                            ketqua = ketqua + '%3d' % (predicted[k])
                            k = k + 1
                        st.subheader(ketqua )
else:
    st.header("BÀI 08")
    bottom_image = st.file_uploader('', type='jpg', key=6)
    
    if bottom_image is not None:
        image = Image.open(bottom_image)
        def pickle_model(model):
            f = io.BytesIO()
            joblib.dump(model, f)
            return f 
        st.image(image)
        x = int(st.number_input("Nhập width: "))
        y = int(st.number_input("Nhập height: "))
        btnRe = st.button("Resize")
        if btnRe:
            st.write("Resize: width =", x," - height = ",y)
            new_image = image.resize((x, y))
            st.image(new_image)
            st.download_button("Download Image", data=pickle_model(new_image), file_name="imageresize.jpg")