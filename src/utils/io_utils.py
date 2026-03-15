import numpy


'''Reads a matrix in clear format(no other lines), and returns a numpy array, floats'''
def citesteMatriceNumpyFloat(cale):
    with open(cale, "r") as f:
        lines = f.readlines()
        a = numpy.zeros((len(lines), len(lines[0].strip().split())))
        for i in range(len(lines)):
            for j in range(len(lines[i].strip().split())):
                a[i, j] = float(lines[i].strip().split()[j])

        return a

'''Reads a matrix in clear format, returns a simple python matrix, integers'''
def citesteMatriceInt(cale):
    with open(cale, "r") as f:
        lines = f.readlines()
        a = []
        for i in range(len(lines)):
            row = []
            for j in range(len(lines[i].strip().split())):
                row.append(int(lines[i].strip().split()[j]))
            a.append(row)
        return a

