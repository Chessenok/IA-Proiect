from scipy.spatial import distance



'''
Manhattan distance between two vectors, x and y, where x and y have the same length.
'''
def calculateFromFile(filename = "in.txt"):
    x = []
    y = []
    with open(filename, "r") as f:
        line = f.readline().removesuffix('\n').split(" ")
        for i in range(0, len(line)):
            x.append(int(line[i]))
        line = f.readline().split(" ")
        for i in range(0, len(line)):
            y.append(int(line[i]))
    sum = 0;
    for i in range(0, len(x)):
        sum += abs(x[i] - y[i])

    return sum

def calculateAutomaticallyFromFile(filename = "in.txt"):
    x = []
    y = []
    with open(filename, "r") as f:
        line = f.readline().removesuffix('\n').split(" ")
        for i in range(0, len(line)):
            x.append(int(line[i]))
        line = f.readline().split(" ")
        for i in range(0, len(line)):
            y.append(int(line[i]))
    sum = distance.cityblock(x,y)
    return sum