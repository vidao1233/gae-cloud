import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

#import source.GiamDanDaoHam.Bai01 as bai1
import source.GiamDanDaoHam.Bai02 as bai2
import source.GiamDanDaoHam.Bai02a as bai2a

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
st.sidebar.markdown("Đạo hàm giảm dần")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Menu',["Bài 01", "Bài 02", "Bài 02a", "Bài 03", "Bài 04", "Bài 05", "Temp"]) 
if(app_mode == 'Bài 01'):
    st.title("BÀI 01")
    def grad(x):
        return 2*x+ 5*np.cos(x)
    def cost(x):
        return x**2 + 5*np.sin(x)

    def myGD1(x0, eta):
        x = [x0]
        for it in range(100):
            x_new = x[-1] - eta*grad(x[-1])
            if abs(grad(x_new)) < 1e-3: # just a small number
                break
            x.append(x_new)
        return (x, it)
    x0 = -5
    eta = 0.1
    (x, it) = myGD1(x0, eta)
    x = np.array(x)
    y = cost(x)
    n = 101
    xx = np.linspace(-6, 6, n)
    yy = xx**2 + 5*np.sin(xx)
    z = [xx, yy]
    fig = plt.subplot(2,4,1)
    plt.plot(xx,yy)
    index = 0
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])



    plt.subplot(2,4,2)
    plt.plot(xx, yy)
    index = 1
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,3)
    plt.plot(xx, yy)
    index = 2
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,4)
    plt.plot(xx, yy)
    index = 3
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,5)
    plt.plot(xx, yy)
    index = 4
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,6)
    plt.plot(xx, yy)
    index = 5
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,7)
    plt.plot(xx, yy)
    index = 7
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.subplot(2,4,8)
    plt.plot(xx, yy)
    index = 11
    plt.plot(x[index], y[index], 'ro')
    s = ' iter%d/%d, grad=%.3f ' % (index, it, grad(x[index]))
    plt.xlabel(s, fontsize = 8)
    plt.axis([-7, 7, -10, 50])

    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Gradient Descent cho hàm 1 biến")
    st.write("Ví dụ")
    st.write("Xét hàm số f(x)=x^2+5sin(x), với đạo hàm f′(x)=2x+5cos(x)(một lý do tôi chọn hàm này vì nó không dễ tìm nghiệm của đạo hàm bằng 0 như hàm phía trên). Giả sử bắt đầu từ một điểm x0 nào đó, tại vòng lặp thứ t")
    st.write("Ta các các hàm số: ")
    st.write("1. grad để tính đạo hàm")
    st.write("2. cost để tính giá trị của hàm số. Hàm này không sử dụng trong thuật toán nhưng thường được dùng để kiểm tra việc tính đạo hàm của đúng không hoặc để xem giá trị của hàm số có giảm theo mỗi vòng lặp hay không.")
    st.write("3. myGD1 là phần chính thực hiện thuật toán Gradient Desent nêu phía trên. Đầu vào của hàm số này là learning rate và điểm bắt đầu. Thuật toán dừng lại khi đạo hàm có độ lớn đủ nhỏ.")
    st.pyplot()
    
elif(app_mode == 'Bài 02'):
    st.title("BÀI 02")
    st.subheader("Gradient Descent cho hàm nhiều biến")
    st.write("Quy tắc cần nhớ: luôn luôn đi ngược hướng với đạo hàm")
    st.write("Sau đây là ví dụ trên Python")
    st.write("Chúng ta tạo 1000 điểm dữ liệu được chọn gần với đường thẳng y = 4 + 3x, hiển thị chúng và tìm nghiệm theo công thức")

    X = np.random.rand(1000)

    y = 4 + 3 * X + .5*np.random.randn(1000)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    x0 = 0
    x1 = 1
    y0 = w*x0 + b
    y1 = w*x1 + b

    z = [x0, x1]
    t = [y0, y1]
    plt.plot(X, y, 'bo', markersize = 2)
    b = plt.plot([x0, x1], [y0, y1], 'r')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write("Đường thẳng tìm được là đường có màu đỏ có phương trình y ≈ 4 + 2.998x")
elif(app_mode == 'Bài 02a'):
    st.title("BÀI 02a")
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000) # noise added

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    w, b = model.coef_[0][0], model.intercept_[0]
    sol_sklearn = np.array([b, w])
    print('Solution found by sklearn:', sol_sklearn)

    # Building Xbar 
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)

    st.write("Solution found by GD: w = ", w1[-1], ',\nafter %d iterations.' %(it1+1))
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))

elif(app_mode == 'Bài 03'):
    st.title('Bài 3')
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    st.write('b = %.4f & w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    st.write('Sol found by GD: w = ', w1[-1], ',\tafter %d iterations.' %(it1+1))
    # for item in w1:
    #     st.write(item, cost(item))

    # st.write(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    temp = w1[0]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax = plt.axes(projection="3d")
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[1]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[2]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)

    temp = w1[3]
    bb = temp[0]
    ww = temp[1]
    zz = cost(temp) 
    ax.plot3D(bb, ww, zz, 'ro', markersize = 3)


    ax.plot_wireframe(b, w, z)
    ax.set_xlabel("b")
    ax.set_ylabel("w")

    st.pyplot(fig=None, clear_figure=None)

elif(app_mode == 'Bài 04'):
    st.title("BÀI 04")
    x = np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    plt.contour(X, Y, Z, 10)
    plt.show()
    st.subheader("Đường đồng mức (level sets)")
    st.write("Với đồ thị của một hàm số với hai biến đầu vào cần được vẽ trong không gian ba chiều, nhều khi chúng ta khó nhìn được nghiệm có khoảng tọa độ bao nhiêu. Trong toán tối ưu, người ta thường dùng một cách vẽ sử dụng khái niệm đường đồng mức (level sets).")
    st.write("Các vòng nhỏ màu đỏ hơn thể hiện các điểm ở trên cao hơn.")
    st.write("Trong toán tối ưu, người ta cũng dùng phương pháp này để thể hiện các bề mặt trong không gian hai chiều.")
    st.pyplot()
elif(app_mode == 'Bài 05'):
    st.title("BÀI 05")
    np.random.seed(100)
    N = 1000
    X = np.random.rand(N)
    y = 4 + 3 * X + .5*np.random.randn(N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    w, b = model.coef_[0][0], model.intercept_[0]
    print('b = %.4f va w = %.4f' % (b, w))

    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

    def grad(w):
        N = Xbar.shape[0]
        return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

    def cost(w):
        N = Xbar.shape[0]
        return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

    def myGD(w_init, eta):
        w = [w_init]
        for it in range(100):
            w_new = w[-1] - eta*grad(w[-1])
            if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
                break 
            w.append(w_new)
        return (w, it)

    w_init = np.array([0, 0])
    (w1, it1) = myGD(w_init, 1)
    print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
    for item in w1:
        print(item, cost(item))

    print(len(w1))

    A = N/(2*N)
    B = np.sum(X*X)/(2*N)
    C = -np.sum(y)/(2*N)
    D = -np.sum(X*y)/(2*N)
    E = np.sum(X)/(2*N)
    F = np.sum(y*y)/(2*N)

    b = np.linspace(0,6,21)
    w = np.linspace(0,6,21)
    b, w = np.meshgrid(b, w)
    z = A*b*b + B*w*w + C*b*2 + D*w*2 + E*b*w*2 + F

    plt.contour(b, w, z, 45)
    bdata = []
    wdata = []
    for item in w1:
        plt.plot(item[0], item[1], 'ro', markersize = 3)
        bdata.append(item[0])
        wdata.append(item[1])

    plt.plot(bdata, wdata, color = 'b')

    plt.xlabel('b')
    plt.ylabel('w')
    plt.axis('square')
    plt.show()
    st.write("Quay trở lại với hình minh họa thuật toán GD cho bài toán Liner Regression bên trên, hình dưới đây là hình biểu diễn các level sets. Tức là tại các điểm trên cùng một vòng, hàm mất mát có giá trị như nhau. Trong ví dụ này, tôi hiển thị giá trị của hàm số tại một số vòng. Các vòng màu xanh có giá trị thấp, các vòng tròn màu đỏ phía ngoài có giá trị cao hơn. Điểm này khác một chút so với đường đồng mức trong tự nhiên là các vòng bên trong thường thể hiện một thung lũng hơn là một đỉnh núi (vì chúng ta đang đi tìm giá trị nhỏ nhất).")
    st.pyplot()
    st.write("Tốc độ hội tụ đã chậm đi nhiều")
    st.write('Trong các bài toán thực tế, chúng ta cần nhiều vòng lặp hơn 99 rất nhiều, vì số chiều và số điểm dữ liệu thường là rất lớn.')
else:
    st.title("TEMP")
    ax = plt.axes(projection="3d")

    X = np.linspace(-2, 2, 21)
    Y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(X, Y)
    Z = X*X + Y*Y
    ax.plot_wireframe(X, Y, Z)
    plt.show()
    st.pyplot()