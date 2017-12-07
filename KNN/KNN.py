import numpy as np
import csv
import sys

'''
The model overview:
    input[n0,image_v] -> batch[n1,image_v] -> get class by L2 -> calculate accuracy
The predict one image:
    input[image_v] -> batch[n1,image_v] -> get class by L2 -> print
'''

TRAIN_SET_FILE = 'train.csv'
TEST_SET_FILE  = 'test.csv'

IMAGE_LEN = 28*28

K = 3

def one_hot(num):
    label = [0]*10
    label[num] = 1
    return label 

#load data   train_set{'images':images,'labels':labels},test_set{'images':images}
def load_train_data(file_train):
    '''load training data to train_set{'images':[n,v],'labels':[n,v]}'''
    #data set
    images = []
    labels = []
    with open(file_train) as f:
        #load csv and reformat
        reader = csv.reader(f)
        next(reader)
        for raw in reader:
            #labels
            raw_digit =[l for l in map(lambda x:int(x),raw)]
            label = raw_digit[0]
            labels.append(label)
            #images
            images.append(raw_digit[1:])
    return {'images':images,'labels':labels}

def load_test_data(file_test):
    '''load test data to test_set{'images':[n,v]}'''
    #data set
    test_images = []
    with open(file_test) as f:
        #load csv and reformat
        test_reader = csv.reader(f)
        next(test_reader)
        for raw in test_reader:
            #images
            test_raw_digit = [l for l in map(lambda x:int(x),raw)]
            test_images.append(test_raw_digit)
    return {'images':test_images}


def main(argv):
    """
    classify argv image by KNN algorithm
    if argv[1] = none compute the accuracy of test set  
    """
    train_set = load_train_data(TRAIN_SET_FILE)

    if 1 == len(argv) :  #none image input,calculate the accuracy of test set
        print('do predictions\r\n')
        test_set = load_test_data(TEST_SET_FILE)                 
        predictions = []
        for test_vector in test_set['images']: 
            #store the distances of each class
            dists = np.array([0]*10)
            #store the index of min k distances&value of min k distances
            ks = {'indexs':[],'values':[],'labels':[]}
            #vectors represent distance of two images vector
            vectors = np.array(train_set['images']) - np.array(test_vector)
            L2s = np.linalg.norm(vectors,ord=2,axis=1)
            for i in range(K):
                ks['indexs'].append(L2s.argmin())
                ks['values'].append(L2s[ks['indexs'][i]]) 
                ks['labels'].append(train_set['labels'][ks['indexs'][i]])
                np.delete(L2s,ks['indexs'][i])
                dists[ks['labels'][i]] += ks['values'][i]     
            predictions.append(dists.argmax())
        with open(r'./predictions.csv','w') as f:
            f.write('ImageId,Label\n')
            for i,v in enumerate(predictions):
                f.write(str(i+1)+','+str(v)+'\n')

#    else:  #predict a image from cmd args
            

main(sys.argv)
