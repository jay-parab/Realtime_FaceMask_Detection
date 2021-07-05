age = [5, 10, 15, 18, 20, 56, 90]


def newFunc(x):
    if x < 18:
        return False
    else:
        return True


adults = filter(newFunc, age)

for x in adults:
    print(x)
