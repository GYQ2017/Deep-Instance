'''
@time:2018-10-09
@author:MrGao
@decribe:
    数组的切片操作
'''
import numpy as np

def fun(array,sRow,eRow,sCol,eCol):
    b = array[sRow:eRow,sCol:eCol]
    print(b)
    return b

if __name__ == '__main__':
    list_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    array_a = np.array(list_a)
    fun(array_a,0,2,1,3)
