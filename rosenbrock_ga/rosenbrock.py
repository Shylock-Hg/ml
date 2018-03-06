import numpy as np
import random

''' ********** problem construct ********** '''
#constraint condition
iv_set = (-2.048,2.048)

#coefficients of rosenbrock math function
a = 1
b = 100

#def rosenbrock math function
rosenbrock = lambda x1,x2 : (a-x1)**2+b*(x2-x1**2)**2

''' ********** hyper-parameters ********** '''
drsl = 2**12 #resolution of sample
size = 300 #count of sample
size_sel = 100 #count of choose
gens = 300 #count of loop 
p_c = 0.6 #probability of cross gen
p_m = 0.001 #probability of variation

''' ********** encode & decode method ********** '''
#code set
code = range(drsl)
code_ori = np.arange(iv_set[0],iv_set[1],(iv_set[1]-iv_set[0])/drsl)

#encode the sample -- 2*float --> int
#encode_one = lambda x : int((iv_set[1]-iv_set[0])/(x-iv_set[0]) * drsl)
#encode = lambda x1,x2 : ((encode_one(x1)&0xFFF)<<12) + encode_one(x2)&0xFFF
find = lambda x : np.abs(np.array([x-i for i in code_ori])).argmin()

def encode(x1,x2):
    return (find(x1)<<12) + find(x2)   

#decode the code -- int --> 2*float
#decode_x1 = lambda x : float((x>>12)-drsl/2)/drsl*iv_set[1]
#decode_x2 = lambda x : float((x&0xFFF)-drsl/2)/drsl*iv_set[1]
def decode_x1(code):
    return code_ori[code>>12]

def decode_x2(code):
    return code_ori[code&0xFFF]

def main():
    setx1 = [code_ori[random.randrange(drsl)] for i in range(size)]
    setx2 = [code_ori[random.randrange(drsl)] for i in range(size)]

    init_code = [ (find(x1)<<12)+find(x2) for x1 in setx1 for x2 in setx2 ]
    print(init_code)

    score = []
    #for _ in range(gens):
        #calculus score
        #score = [ rosenbrock(x1,x2) for x1 in setx1 for x2 in setx2 ]

        #print(score)
        #choose 
        

if __name__ == '__main__':
    main()
