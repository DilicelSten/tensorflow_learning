# TensorFlow learning


**TensorFlow** is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

## Learning process


### **Linear Regression**

$$
 \varGamma(x) = \frac{\int_{\alpha}^{\beta} g(t)(x-t)^2\text{ d}t }{\phi(x)\sum_{i=0}^{N-1} \omega_i} \tag{2}
$$

$$
 H(x)=Wx + b
$$

\\ (cost(W,b) = \frac{1}{m}\sum_{i=1}^m(H(x^i)-y^i)^2\\)


### **Logistic Regression**


\\ (H(X) = sigmoid(XW) = \frac{1}{1+e^-XW} \\)

\\ (cost(W) = -\frac{1}{m}\sum ylog(H(X) + (1-y)(log(1 - H(X)\\)

updating...
