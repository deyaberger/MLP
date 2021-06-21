try:
    import numpy as np
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return (a)


def sigmoid_derivative(a):
    da = (a) * (1 - a)
    _, n = da.shape
    da = np.einsum('ij,jk->ijk' , da, np.eye(n, n))
    return da


def softmax(z):
    z = z - z.max(axis = 1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis = 1, keepdims=True)
    return (e / s)


def softmax_derivative(a):
    m, n = a.shape # m = nb examples, n = nb features

    # First we create for each example feature vector, it's outer product with itself:
    # ( a1^2  a1*p2  a1*p3 .... )
    # ( a2*p1 a2^2   a2*p3 .... )
    # ( ...                     )
    tensor1 = np.einsum('ij,ik->ijk', a, a)  # (m, n, n)

    # Second we need to create an (n,n) identity of the feature vector
    # ( a1  0  0  ...  )
    # ( 0   a2 0  ...  )
    # ( ...            )
    tensor2 = np.einsum('ij,jk->ijk', a, np.eye(n, n))  # (m, n, n)

    # Then we need to subtract the first tensor from the second
    # ( a1 - a1^2   -a1*a2   -a1*a3  ... )
    # ( -a1*a2     a2 - a2^2   -a2*a3 ...)
    # ( ...                              )
    da = tensor2 - tensor1
    return da


def identity(z):
    return (z)


def identity_derivative(a):
    _, n = a.shape
    da = np.einsum('ij,jk->ijk' , np.ones(a.shape), np.eye(n, n))
    return da


def get_activation(activation):
        a = None
        if activation == "sigmoid":
            a, d_a = sigmoid, sigmoid_derivative
        elif activation == "softmax":
            a, d_a = softmax, softmax_derivative
        elif activation == None or activation == "identity":
            a, d_a = identity, identity_derivative
        return (a, d_a)