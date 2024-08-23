import warnings
import autograd.numpy as np
from autograd import grad, hessian
from numpy.linalg import inv,norm


def diff(f,x,sig=1e-4):
    """first-order derivative of function `f` at point `x`
    ---
    * f: target function
    * x: point
    * sig: neighbor range of x
    """
    return (f(x+sig/2)-f(x-sig/2))/sig


def diff2(f,x,sig=1e-4):
    """second-order derivative of function `f` at point `x`
    ---
    * f: target function
    * x: point
    * sig: neighbor range of x
    """
    return (diff(f,x+sig/2) - diff(f,x-sig/2))/sig


def Newton(x0,f,error=1e-4):
    """Newton method optimization
    ---
    * x0: start point
    * f: target function
    * error: absolute error of optimization
    """
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    delta = float('inf')
    x = x0
    while delta>error:
        if x > 1e7:
           raise RuntimeError(f"At iteration {iter}, optimization appears to be diverging")
        if abs(x) > 100:
           warnings.warn(f"abs({x}) is greater than 100.")
        
        delta = diff(f,x)/diff2(f,x)
        while f(x-delta)>f(x):
            delta /= 2
        x -= delta
    return x


def Newton_multi(x0:np.ndarray,f,error=1e-4):
    """Multivariate Newton’s method
    make sure x0 is float
    """
    f_H = hessian(f)
    f_g = grad(f)
    delta = float('inf')
    x = x0
    while delta>error:
        step = inv(f_H(x)) @ f_g(x)
        delta = norm(step)
        x -= step
    return x


def main():
    def f(x): return x**4/4-x**3-x
    def f_multi(x): return (x[0]-1)**2 + x[1]**2 + 3*x[0]*x[1]
    print(Newton(10,f))
    print(Newton_multi(np.array([10.,1.]),f_multi))



if __name__ == '__main__':
    main()