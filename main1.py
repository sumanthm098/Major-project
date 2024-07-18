import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
import seaborn as sns
from skimage import io, transform
from sklearn import preprocessing
import cv2
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##Creating a User interface

main=tk.Tk()
main.title("Pneumonia Detection")
main.geometry("1650x1000")
main.config(bg='teal')


image = Image.open(r"C:\Users\m.sumanth\Downloads\pexels-cottonbro-studio-7579834.jpg")
image = image.resize((1650, 1000), Image.ANTIALIAS) 
photo = ImageTk.PhotoImage(image)

#Create a label with the image as background
background_label = tk.Label(main, image=photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

title=tk.Label(main, text="Machine Learning Model For Pneumonia Detection Using Chest X-rays",justify='center')


##Functions of buttons

def upload():
    global categories
    filename=filedialog.askdirectory(initialdir=".")
    path=filename
    model_folder='model1'
    categories=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    text.delete('1.0',END)
    text.insert(END,"Dataset Loaded Successfully. \n\n")
    text.insert(END,"Total categories found in the dataset"+str(categories)+'\n\n')
def preprocess():
    global X,Y
    global model_folder
    path=r"dataset"
    model_folder='model1'
    categories=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    X_file = os.path.join(model_folder, "X.npy")
    Y_file = os.path.join(model_folder, "Y.npy")
    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        text.delete('1.0',END)
        text.insert(END,"Total images found in the dataset  :  "+str(X.shape[0])+'\n\n')
        text.insert(END,"Total categories found in the dataset : "+str(categories)+'\n\n')
    else:
        X =[]#input array
        Y =[]#output array
        for i in categories:
            text.delete('1.0',END)
            text.insert(END,f'Loading category:{i}\n\n')
            category_path=os.path.join(path,i)
            for img in os.listdir(category_path):
                img_array=imread(os.path.join(category_path, img))
                img_resized=resize(img_array,(256,256,3))
                X.append(img_resized.flatten())
                Y.append(categories.index(i))
                text.insert(END,f'Loaded image:{img}successfully\n\n')
            text.insert(END,f'Loaded category: {i} sucessfully\n\n')

        os.makedirs(model_folder)
        np.save(X_file,X)
        np.save(Y_file,Y)
        text.insert(END,"image precessing  done succesufully")
        text.insert(END,"Total images found in the dataset  : "+str(X.shape[0]))
        text.insert(END,"Total categories found in the dataset : "+str(categories))
        
def data_splitting():
        global X_train,X_test,Y_train,Y_test
        text.delete('1.0',END)
        X_file = os.path.join(model_folder, "X.npy")
        Y_file = os.path.join(model_folder, "Y.npy")

        if os.path.exists(X_file) and os.path.exists(Y_file):
            X = np.load(X_file)
            Y = np.load(Y_file)
            text.delete('1.0',END)
            text.insert(END,"X and Y arrays loaded successfully.\n\n")
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=77)
            text.insert(END,"Total trained dataset  : "+str(X_train.shape[0])+"\n")
            text.insert(END,"Total tested dataset  : "+str(X_test.shape[0])+'\n')
        else:
            X =[]#input array
            Y =[]#output array
            text.delete('1.0',END)
            for i in categories:
                text.insert(END,f'Loading category:{i}\n\n')
                category_path=os.path.join(path,i)
                for img in os.listdir(category_path):
                    img_array=imread(os.path.join(category_path, img))
                    img_resized=resize(img_array,(256,256,3))
                    X.append(img_resized.flatten())
                    Y.append(categories.index(i))
                    text.insert(END,f'Loaded image:{img} succesfully\n\n')
                text.insert(END,f'Loaded category: {i} sucesfully\n\n')

            os.makedirs(model_folder)
            np.save(X_file,X)
            np.save(Y_file,Y)

def navibayes(): 
    
    #check if the pkl file exists
    Model_file = os.path.join(model_folder,"NBC_Model.pkl")
    text.delete('1.0',END)
    if os.path.exists(Model_file):
        #load the model from the pkl file 
        nb_classifier = joblib.load(Model_file)
        y_pred = nb_classifier.predict(X_test)
        accuracy = accuracy_score(Y_test,y_pred)
        report = classification_report(Y_test, y_pred, target_names=categories)
        # Print the classification report
        text.insert(END,'  Report for Gaussian Navi_Bayes : \n\n')
        text.insert(END,str(report)+'\n')
        text.insert(END,"Data trained successfully\n\n")
        accuracy = accuracy_score(Y_test, y_pred)*100
        precision = precision_score(Y_test, y_pred, average='weighted')*100
        recall = recall_score(Y_test, y_pred, average='weighted')*100
        f1 = f1_score(Y_test, y_pred, average='weighted')*100
        text.insert(END,f'Accuracy: {accuracy:.3f}\n')
        text.insert(END,f'Precision: {precision:.3f}\n')
        text.insert(END,f'Recall: {recall:.3f}\n')
        text.insert(END,f'F1 Score: {f1:.3f}\n')
    else:
        #Create a gaussian Naivebayes classifier
        nb_classifier= BernoulliNB()
        #train the classifier on the training data 
        nb_classifier.fit(X_train,Y_train)
        joblib.dump(nb_classifier, Model_file)
        y_pred = nb_classifier.predict(X_test)
        accuracy = accuracy_score(Y_test,y_pred)
        report = classification_report(Y_test, y_pred, target_names=categories)
        text.delete('1.0',END)
        # Print the classification report
        text.insert(END,'  Report for Gaussian Navi_Bayes : \n\n')
        text.insert(END,str(report)+'\n')
        text.insert(END,"Data trained successfully\n\n")
        accuracy = accuracy_score(Y_test, y_pred)*100
        precision = precision_score(Y_test, y_pred, average='weighted')*100
        recall = recall_score(Y_test, y_pred, average='weighted')*100
        f1 = f1_score(Y_test, y_pred, average='weighted')*100
        text.insert(END,f'Accuracy: {accuracy:.3f}\n')
        text.insert(END,f'Precision: {precision:.3f}\n')
        text.insert(END,f'Recall: {recall:.3f}\n')
        text.insert(END,f'F1 Score: {f1:.3f}\n')
        
def rfc():
    global model1
    text.delete('1.0',END)
    text.insert(END,'  Report for Random Forest Classification : \n\n')
    model1=RandomForestClassifier()
    model1.fit(X_train,Y_train)
    y_pre=model1.predict(X_test)
    report1 = classification_report(Y_test, y_pre, target_names=categories)
    accuracy_rfc = accuracy_score(Y_test, y_pre)*100
    precision_rfc = precision_score(Y_test, y_pre, average='weighted')*100
    recall_rfc = recall_score(Y_test, y_pre, average='weighted')*100
    f1_rfc = f1_score(Y_test, y_pre, average='weighted')*100
    text.insert(END,"Data trained successfully\n\n")
    text.insert(END,str(report1)+'\n')
    text.insert(END,f'Accuracy: {accuracy_rfc:.3f}\n')
    text.insert(END,f'Precision: {precision_rfc:.3f}\n')
    text.insert(END,f'Recall: {recall_rfc:.3f}\n')
    text.insert(END,f'F1 Score: {f1_rfc:.3f}\n')
    #y_acc=accuracy_score(Y_test,y_pre)*100
    #text.insert(END,str(y_acc))
    cm = confusion_matrix(Y_test, y_pre)
    class_names=categories
    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names,yticklabels=class_names,fmt='d')
    # Set the axis labels and title
    plt.xlabel("Predicted Class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    # Display the heatmap
    plt.show()

def test1():
    global Test_img1
    Test_img1=filedialog.askopenfilename(initialdir=".")
    img=imread(Test_img1)
    img_resize=resize(img,(256,256,3))
    img_preprocessed = [img_resize.flatten()]
    text.delete('1.0',END)
    text.insert(END,Test_img1+'Image loaded Successfully.\n\n')
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    #img=cv2.flip(img,-1)
    output_number=model1.predict(img_preprocessed)[0]
    output_name=categories[output_number]

    plt.imshow(img)
    plt.text(10,10,f'predicted output:{output_name}', color='white',fontsize=12,weight='bold',backgroundcolor='black')
    plt.axis('off')
    plt.show()
    

##Styling
    
title.grid(column=0,row=0)
font=('Algerian',17)
title.config(fg='navy blue',bg='orange')
title.config(font=font)
title.config(height=3,width=110)
title.place(x=0,y=0)
s=Scale(main, from_=0 , to=100)

##Creating buttons

uploadButton1=Button(main,text="Upload Dataset",command=upload)
uploadButton1.place(x=95,y=110)
uploadButton1.config(bg='cyan',font=font,fg='red')

font1=('Bell MT',12,'bold')
text=Text(main,height=15.5,width=60)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,bg='white',fg='black')
text.place(x=703,y=400)
text.config(font=font1)

uploadButton2=Button(main,text="Image Pre-Process",command=preprocess)
uploadButton2.place(x=320,y=110)
uploadButton2.config(bg='green',font=font,fg='yellow')

    
uploadButton3=Button(main,text="Data Splittng",command=data_splitting)
uploadButton3.place(x=575,y=110)
uploadButton3.config(bg='violet',font=font,fg='black')

uploadButton4=Button(main,text="Naive Bayes",command=navibayes)
uploadButton4.place(x=125,y=250)
uploadButton4.config(bg='light green',font=font,fg='green')

uploadButton5=Button(main,text="Random Forest Classifier",command=rfc)
uploadButton5.place(x=45,y=350)
uploadButton5.config(bg='teal',font=font,fg='aqua')

uploadButton5=Button(main,text="Prediction",command=test1)
uploadButton5.place(x=130,y=450)
uploadButton5.config(bg='blue',font=font,fg='white')





main.mainloop()