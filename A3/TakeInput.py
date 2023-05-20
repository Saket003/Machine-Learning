import cv2
import os
import numpy as np
import csv

def take_input(path,n_samples):
    #Labelling paths
    dog_path = path + "/dog"
    airplane_path = path + "/airplane"
    person_path = path + "/person"
    car_path = path + "/car"

    #Initializing and Assigning output valies
    feature_vector = np.zeros(shape = (n_samples, 3072))
    output_vector = np.zeros(n_samples)
    output_vector[int((3*n_samples)/4):n_samples] = np.ones(int(n_samples/4))

    i = 0
    for file in os.listdir(car_path):
        img = cv2.imread(car_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(airplane_path):
        img = cv2.imread(airplane_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(dog_path):
        img = cv2.imread(dog_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(person_path):
        img = cv2.imread(person_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1

    return feature_vector, output_vector

def take_input_multiclass(path,n_samples):
    #Labelling paths
    dog_path = path + "/dog"
    airplane_path = path + "/airplane"
    person_path = path + "/person"
    car_path = path + "/car"

    #Initializing and Assigning output valies
    feature_vector = np.zeros(shape = (n_samples, 3072))
    output_vector = np.zeros(n_samples)

    output_vector[int((n_samples)/4):int(n_samples/2)] = np.zeros(int(n_samples/4)) + 1
    output_vector[int(n_samples/2):int((3*n_samples)/4)] = np.zeros(int(n_samples/4)) + 2
    output_vector[int((3*n_samples)/4):n_samples] = np.zeros(int(n_samples/4)) + 3

    i = 0
    for file in os.listdir(car_path):
        img = cv2.imread(car_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(person_path):
        img = cv2.imread(person_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(airplane_path):
        img = cv2.imread(airplane_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(dog_path):
        img = cv2.imread(dog_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1


    return feature_vector, output_vector

def take_input_test(path):
    n_samples = 0
    for file in os.listdir(path):
        n_samples += 1
    
    test_vector = np.zeros((n_samples,3072))
    test_labels = []

    i = 0
    for file in os.listdir(path):
        img = cv2.imread(path +"/" +file)
        test_vector[i] = img.reshape(1,3072)
        test_labels.append(file.strip(".png"))
        i +=1

    return test_vector,test_labels

def csvoutput(predicted_vector,test_lables,file_name):
    test_numbers = np.zeros_like(predicted_vector)
    for i in range(len(test_numbers)):
        test_numbers[i] = int(test_lables[i].strip("img_"))

    sort = np.unravel_index(np.argsort(test_numbers, axis=None), test_numbers.shape)
    #test_lables = test_lables[sort]
    predicted_vector = predicted_vector[sort]

    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(len(predicted_vector)):
            csvwriter.writerow(["img_"+str(i+1),int(predicted_vector[i])])

def take_input_multiclass_2(path):
    dog_path = path + "/dog"
    airplane_path = path + "/airplane"
    person_path = path + "/person"
    car_path = path + "/car"

    d_samples = 0
    a_samples = 0
    p_samples = 0
    c_samples = 0
    for file in os.listdir(car_path):
        c_samples += 1
    for file in os.listdir(person_path):
        p_samples += 1
    for file in os.listdir(airplane_path):
        a_samples += 1
    for file in os.listdir(dog_path):
        d_samples += 1

    n_samples = d_samples+a_samples+p_samples+c_samples
    feature_vector = np.zeros(shape = (n_samples, 3072))
    output_vector = np.zeros(n_samples)

    output_vector[c_samples:c_samples+p_samples] = np.zeros(p_samples) + 1
    output_vector[c_samples+p_samples:c_samples+p_samples+d_samples] = np.zeros(a_samples) + 2
    output_vector[c_samples+p_samples+d_samples:n_samples] = np.zeros(d_samples) + 3

    i = 0
    for file in os.listdir(car_path):
        img = cv2.imread(car_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(person_path):
        img = cv2.imread(person_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(airplane_path):
        img = cv2.imread(airplane_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(dog_path):
        img = cv2.imread(dog_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1

    return feature_vector, output_vector

def take_input_2(path):
    dog_path = path + "/dog"
    airplane_path = path + "/airplane"
    person_path = path + "/person"
    car_path = path + "/car"

    p_samples = 0
    not_p_samples = 0
    for file in os.listdir(car_path):
        not_p_samples += 1
    for file in os.listdir(person_path):
        p_samples += 1
    for file in os.listdir(airplane_path):
        not_p_samples += 1
    for file in os.listdir(dog_path):
        not_p_samples += 1

    feature_vector = np.zeros(shape = (not_p_samples+p_samples, 3072))
    output_vector = np.zeros(not_p_samples+p_samples)
    output_vector[not_p_samples:not_p_samples+p_samples] = np.ones(p_samples)

    i = 0
    for file in os.listdir(car_path):
        img = cv2.imread(car_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(airplane_path):
        img = cv2.imread(airplane_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(dog_path):
        img = cv2.imread(dog_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1
    for file in os.listdir(person_path):
        img = cv2.imread(person_path +"/" +file)
        feature_vector[i] = img.reshape(1,3072)
        i +=1

    return feature_vector, output_vector