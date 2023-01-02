import os
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mahotas as mt
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Response

def init_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask

def preprocess_background(filename): 
    original_image = cv2.imread("img/" + filename)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image , (500,500))
    image = original_image
    
    h, w = image.shape[:2]
    mask = init_mask(h, w)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    backgroundModel = np.zeros((1, 65), np.float64) 
    foregroundModel = np.zeros((1, 65), np.float64) 

    rectangle = (100, 10, 300, 480) 

    cv2.grabCut(image, mask, rectangle,   
                backgroundModel, foregroundModel, 
                10, cv2.GC_INIT_WITH_RECT) 

#     mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
    mask2 = (mask==2) | (mask==0)

#     image = image * mask2[:, :, np.newaxis] 

    image[mask2] = 255
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
    

def feature_extract(image):
    
    #Membuat perulangan untuk kolom feature glcm
    glcm_feature = ['correlation', 'homogeneity','dissimilarity','contrast','ASM','energy']
    angle = ['0','45','90','135']
    
    # Membuat variabel kolom untuk dataset
    names = ['physiological_length','physiological_width','aspect_ratio','rectangularity',\
             'eccentricity',\
             'hue','saturation','value',\
            ]
    
    #Pemanggilan perulangan kolom glcm
    for i in glcm_feature:
        for j in angle:
            names.append(i + ' ' + j)
    
    # Membuat dataframe berdasarkan nama kolom yang dibuat
    df = pd.DataFrame([], columns=names)
    
    #Preprocessing
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    blur = cv2.GaussianBlur(grayscale, (25,25),0)
    ret, img1 = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

    b, g, r=cv2.split(img)
    rgba = [b,g,r,img1]
    dst = cv2.merge(rgba, 4)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    select = max(contours, key=cv2.contourArea)  
    x,y,w,h = cv2.boundingRect(select)

    png = dst[y:y+h,x:x+w]
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

    #shape
    area = cv2.contourArea(select)
    aspect_ratio = float(w)/h
    rectangularity = w*h/area
    
    #shape eccentricity
    dimension = png.shape
    height = png.shape[0]
    width = png.shape[1]
    mayor = max(height,width)
    minor = min(height,width)
    eccentricity = math.sqrt(1-((minor*minor)/(mayor*mayor)))    

    #hsv color
    hsv = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
    height=png.shape[0]
    width=png.shape[1]
    H=hsv[:,:,0]
    S=hsv[:,:,1]
    V=hsv[:,:,2]

    hue = np.reshape(H,(1,height*width))
    mode_h = stats.mode(hue[0])
    if int(mode_h[0])==0:
        mode_hue = np.mean(H)
    else:
        mode_hue = int(mode_h[0])

    mean_s = np.mean(S)
    mean_v = np.mean(V)

    #glcm
    distance = [5]
    angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
    levels = 256
    symetric = True
    normed = True

    glcm = greycomatrix(gray, distance, angles, levels, symetric, normed)

    # Membuat dataset berdasarkan variabel kolom
    glcm_props = [propery for name in glcm_feature for propery in greycoprops(glcm,name)[0]]
    vector = [w,h,aspect_ratio,rectangularity,eccentricity,mode_hue,mean_s,mean_v] + glcm_props

    df_temp = pd.DataFrame([vector],columns=names)
    df = df.append(df_temp)  
            
    return df


def get_predict_image(imageFile):
    ext = imageFile.filename.split(".")[1]
    filename = str("image_temp."+ ext)
    print("filename: " +filename)
    with open("img/"+filename, "wb+") as image_obj:
        shutil.copyfileobj(imageFile.file, image_obj)

    bg_rem_img = preprocess_background(filename)

    features_of_img = feature_extract(bg_rem_img)

    drop_features = ['physiological_length',
    'saturation',
    'correlation 0',
    'correlation 45',
    'correlation 135']

    df_train = pd.read_csv('dataset/Dataset10Kelas.csv')
    # Menyimpan fitur atribut ke dalam variabel X_train
    X = df_train.drop(labels = ['class'],axis = 1) 
    #menyimpan class (label) pada y_train
    y = df_train['class']
    
    # Melakukan pembagian data dengan train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
        
    # Melakukan standarisasi data
    X_train = scaler.fit_transform(X_train)

    features_of_img = features_of_img.drop(drop_features, axis=1)
    
    # ft = features_of_img.iloc[0]
    # print(ft)

    scaled_features = scaler.transform(features_of_img)

    model_ovr = pickle.load(open('svm_model/Model10Kelas.pkl', 'rb'))
    # print('\nmodel loaded...')

    prob = model_ovr.predict_proba(scaled_features)
    cls = model_ovr.classes_
    # print(f"\nKelas : {cls} \n\nProbabilitas : {prob}")

    prob_best = np.sort(prob)[:,:-3-1:-1]*100
    cls_best = np.argsort(prob)[:,:-3-1:-1]+1
    # print(f"\nKelas dengan 3 Probabilitas Tertinggi : {cls_best} = {prob_best}")

    names = {
    1 : 'Bidara',
    2 : 'Jambu',
    3 : 'Miana',
    4 : 'Sirih',
    5 : 'Anting-anting',
    6 : 'Bayam Duri',
    7 : 'Kirinyuh',
    8 : 'Daun Ungu',
    9 : 'Sidaguri',
    10 : 'Sirsak'
    }

    x = cls_best[0]
    y = prob_best[0]

    # print(f" 1. {names[x[0]]} = {round(y[0], 3)}\n 2. {names[x[1]]} = {round(y[1],3)}\n 3. {names[x[2]]} = {round(y[2],3)}")

    y_pred_mobile = model_ovr.predict(scaled_features)
    result = y_pred_mobile[0]
    
    # print(f"\nHasil Klasifikasi adalah Kelas = {result} (Tanaman Obat {names[result]})\n")

    value = {
        "result": names[result],
        "probabilitas": [
            {
                "name" : names[x[0]],
                "presentage": str(round(y[0], 3))
            },
            {
                "name" : names[x[1]],
                "presentage": str(round(y[1], 3))
            },
            {
                "name" : names[x[2]],
                "presentage": str(round(y[2], 3))
            },
        ]
    }

    json_str = json.dumps(value)
    print(json_str)
    return Response(content=json_str, media_type='application/json')