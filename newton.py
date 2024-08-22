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
    delta = float('inf')
    x = x0
    while delta>error:
        delta = diff(f,x)/diff2(f,x)
        x -= delta
    return x

def main():
    def f(x): return (x-1.533)**2
    print(Newton(10,f))

if __name__ == '__main__':
    main()