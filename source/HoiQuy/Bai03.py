from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
X2 = X**2
# print(X)
# print(X2)
X_poly = np.hstack((X, X2))
# print(X_poly)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)
a = lin_reg.intercept_[0]
b = lin_reg.coef_[0,0]
c = lin_reg.coef_[0,1]
print(a)
print(b)
print(c)

x_ve = np.linspace(-3,3,m)
y_ve = a + b*x_ve + c*x_ve**2

plt.plot(X, y, 'o')
plt.plot(x_ve, y_ve, 'r')

# Tinh sai so
loss = 0 
for i in range(0, m):
    y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
    sai_so = (y[i] - y_mu)**2 
    loss = loss + sai_so
loss = loss/(2*m)
print('loss = %.6f' % loss)

# Tinh sai so cua scikit-learn
y_train_predict = lin_reg.predict(X_poly)
# print(y_train_predict)
sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
print('sai so binh phuong trung binh: %.6f' % (sai_so_binh_phuong_trung_binh/2))
plt.show()
