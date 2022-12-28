#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi dengan Multi-SVM

# In[1]:


# Library
# Untuk mengolah data 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Untuk mengimport SVM
from sklearn import svm

# Untuk digunakan pada SVM dengan parameter tuning 
from sklearn.model_selection import GridSearchCV
# Standarisasi dengan metode StandardScaler
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
# Untuk memanggil metrik akurasi
from sklearn import metrics
# Untuk visualisasi data
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

# Menghitung nilai akurasi untuk model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# ## Memanggil Dataset

# In[2]:


# Memuat data train pada dataframe
df_train = pd.read_csv('dataset/DataTraining_FlipCorr.csv')

df_train.head(5)


# In[3]:


# #Menghilangkan kolom hsv
# df_train = df_train.drop(columns=['hue','saturation','value'])
# df_train.head(5)


# In[4]:


# Memuat informasi pada dataframe train
df_train.info()


# In[5]:


# Memuat deskripsi pada dataframe train
df_train.describe()


# In[6]:


# Menyimpan fitur atribut ke dalam variabel X_train
X = df_train.drop(labels = ['class'],axis = 1) 
#menyimpan class (label) pada y_train
y = df_train['class']


# ## Melakukan pembagian data pada dataset dengan train_test_split

# In[7]:


# Melakukan pembagian data dengan train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('X_train = ' + str(len(X_train)) + ' , ' + 'y_train = ' + 
      str(len(y_train)) + ' , ' + 'X_test = ' + str(len(X_test)), 'y_test = ' + str(len(y_test)))


# In[8]:


y_test = y_test.tolist()
y_train = y_train.tolist()


# In[9]:


print(X_train.iloc[:,:])
print(type(X_train))


# ## Standarisasi dengan metode StandardScaler

# In[10]:


# inisiasi StandardScaler
scaler = StandardScaler()
# Melakukan standarisasi data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## GridSearchCV untuk Hyper Parameter

# In[11]:


parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]


# In[12]:


# Menggunakan Gridsearch dengan memanggil class SVC
model_param = GridSearchCV(svm.SVC(),parameters,cv=5)
#melakukan training pada objek dan label
model_param.fit(X_train, y_train)


# In[13]:


# Menampilkan hasil dari model SVM berdasarkan Hyper Parameter Tuning
means = model_param.cv_results_['mean_test_score']
stds = model_param.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, model_param.cv_results_['params']):
    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))


# In[14]:


# Hasil hyperparameter tuning dengan skor terbaik yang di dapatkan
print(f"Best parameter {model_param.best_params_} with score {model_param.best_score_}")


# In[15]:


# Melakukan prediksi pada data testing
y_pred1 = model_param.predict(X_test)
print('hasil prediksi train dari best_param : ', metrics.accuracy_score(y_test, y_pred1))


# In[16]:


conf1= confusion_matrix(y_test, y_pred1)
#confusion matrix
conf_matrix = pd.DataFrame(conf1, ('1','2','3','4'), ('1','2','3','4'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 14}, fmt='d', cmap='Blues')

plt.title('Confusion Matrix untuk  Model SVM\n(Dengan hyperparameter tuning)\n', fontsize=18)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# ## Model SVM OneVsOneClassifier dengan BestParameter

# In[17]:


from sklearn.multiclass import OneVsOneClassifier

model_ovo = OneVsOneClassifier(svm.SVC(C=100, kernel='linear'))
model_ovo.fit(X_train, y_train)                       

y_pred4 = model_ovo.predict(X_test)                          
metrics.accuracy_score(y_test, y_pred4)


# ## Model SVM OneVsRestClassifier dengan BestParameter

# In[18]:


from sklearn.multiclass import OneVsRestClassifier
model_ovr = OneVsRestClassifier(svm.SVC(C=100, kernel='linear'))
model_ovr.fit(X_train, y_train)                       

y_pred3 = model_ovr.predict(X_test)                          
metrics.accuracy_score(y_test, y_pred3)                


# In[19]:


conf3= confusion_matrix(y_test, y_pred3)

#confusion matrix
conf_matrix = pd.DataFrame(conf3, ('1','2','3','4'), ('1','2','3','4'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 14}, fmt='d', cmap='Blues')

plt.title('Confusion Matrix untuk  Model SVM\nOneVsRestClassifier\n', fontsize=18)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# ## Testing Image

# In[20]:


import os
import cv2


# In[21]:


def init_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask


# In[22]:


def preprocess_grabcut(filename): 
    test_img_path = 'Testing/' + filename
    original_image = cv2.imread(test_img_path)
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
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image)
    plt.show(block=True)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     cv2.imwrite(filename, image)
    
    return image
    


# In[32]:




# In[24]:


import mahotas as mt
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats


# In[33]:


def feature_extract(image):
    
    #Membuat perulangan untuk kolom feature glcm
    glcm_feature = ['correlation', 'homogeneity','dissimilarity','contrast','ASM','energy']
    angle = ['0','45','90','135']
    
    # Membuat variabel kolom untuk dataset
    names = ['area','perimeter','physiological_length','physiological_width','aspect_ratio','rectangularity','circularity',             'eccentricity','metric',             'hue','saturation','value',            ]
    
    #Pemanggilan perulangan kolom glcm
    for i in glcm_feature:
        for j in angle:
            names.append(i + ' ' + j)
    
    # Membuat dataframe berdasarkan nama kolom yang dibuat
    df = pd.DataFrame([], columns=names)
    
    #Preprocessing
#     test_img_path = 'Bidara.jpg'
#     original_image = cv2.imread(test_img_path)
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
    perimeter = cv2.arcLength(select,True)
    aspect_ratio = float(w)/h
    rectangularity = w*h/area
    circularity = ((perimeter)**2)/area
    
    #shape eccentricity
    dimension = png.shape
    height = png.shape[0]
    width = png.shape[1]
    mayor = max(height,width)
    minor = min(height,width)
    eccentricity = math.sqrt(1-((minor*minor)/(mayor*mayor)))

    #shape metric
    height1=png.shape[0]
    width1=png.shape[1]
    edge = cv2.Canny(img,100,200)
    k=0
    keliling=0
    while k<height1:
        l=0
        while l<width1:
            if edge[k,l]==255:
                keliling=keliling+1
            l=l+1
        k=k+1
    k=0
    luas = 0
    while k<height1:
        l=0
        while l<width1:
            if img1[k,l]==255:
                luas=luas+1
            l=l+1
        k=k+1
    metric = (4*math.pi*luas)/(keliling*keliling)    

#     #hsv color
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
    vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,eccentricity,metric,mode_hue,mean_s,mean_v] + glcm_props

    df_temp = pd.DataFrame([vector],columns=names)
    df = df.append(df_temp)  
            
    return df
            


# In[34]:




# In[35]:



# In[36]:



# In[37]:




# ## Save Model

# In[30]:




# In[31]:



# In[ ]:

def get_predict_image():
    filename = 'Jambu1.jpg' 
    bg_rem_img = preprocess_grabcut(filename)

    features_of_img = feature_extract(bg_rem_img)

    ft = np.array(features_of_img)

    drop_features = ['area',
    'physiological_length',
    'hue',
    'correlation 45',
    'correlation 90',
    'correlation 135',
    'dissimilarity 0',
    'dissimilarity 45',
    'dissimilarity 90',
    'dissimilarity 135',
    'contrast 0',
    'contrast 45',
    'contrast 90',
    'contrast 135']

    scaler = StandardScaler()
    

    # features_of_img = features_of_img.drop(drop_features, axis=1)
    # scaled_features = scaler.fit_transform(features_of_img)
    features_of_img = features_of_img.drop(drop_features, axis=1)
    scaled_features = scaler.transform(features_of_img)

    # y_pred_mobile = model_ovr.predict(scaled_features)

    import pickle
    model_ovr = pickle.load(open('svm_model/Model1.pkl', 'rb'))
    print('model loaded...')
    y_pred_mobile = model_ovr.predict(scaled_features)

    result = y_pred_mobile[0]
    print(result)

    names = {
        1 : 'Bidara',
        2 : 'Jambu',
        3 : 'Miana',
        4 : 'Sirih',
    }

    return names[result]


