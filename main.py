lane_rate = 0.02
epochs = 4

X1 = 3
X2 = 22
y = 0
b = 3


def expoWhile(x, y, w1, w2, b, _m, cd):
    while True:
        result = w1 * x + w2 * y + b
        if result > 0:
            return
        print("El resultado es: ", result)
        if cd == 'S': 
            w1 += _m
            w2 += _m
            b += _m
        elif cd == 'R':
            w1 -= _m
            w2 -= _m
            b -= _m
expoWhile(3, 22, -1.86545988, -0.25588569, 3.5, 0.02, 'S')

#Esto es 





result = (X1*0.02) + (X2*0.02) + b
print(result)
