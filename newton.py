def diff(f,x,sig=1e-4):
    return (f(x+sig/2)-f(x-sig/2))/sig

def diff2(f,x,sig=1e-4):
    return (diff(f,x+sig/2) - diff(f,x-sig/2))/sig

def Newton(x0,f,error=1e-4):
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