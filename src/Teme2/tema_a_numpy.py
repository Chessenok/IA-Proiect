import numpy as np
rand = np.random.RandomState(15)

def calcSum(C):
    s = np.sum(C)
    print(f'suma C = {s}')
    return s

def calcMean(C):
    m = np.mean(C,axis=0)
    print(f'media pe coloane - {m}')
    return m

def calcMults(a,b):
    m = a @ b
    print(f'a @ b = {m}')
    return m

def bonus(mat):
    print(f'BONUS - e data matricea mat - {mat}')
    inversa = np.linalg.inv(mat)
    print(f'inversa - {inversa}')
    det = np.linalg.det(inversa)
    print(f'det - {det}')
    print(f'E aproape de matricea unitate? - {np.allclose((mat @ inversa),np.eye(3))}')


a = np.array([[rand.randint(0,10) for _ in range(3)] for _ in range(4)])
b = np.array([[rand.randint(0,10) for _ in range(5)]for _ in range(3)])
C = calcMults(a,b)
mat = np.array([[rand.randint(0,10) for _ in range(3)]for _ in range(3)])

calcSum(C)
calcMean(C)
bonus(mat)

