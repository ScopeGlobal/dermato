from logging import debug
from flask import Flask, request, render_template,session, redirect
from werkzeug.utils import secure_filename
from flask.helpers import url_for
from flask_sqlalchemy import SQLAlchemy
import pymysql
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, \
                          Dropout, Input,GlobalAveragePooling2D,BatchNormalization)
from tqdm import tqdm
from tensorflow.keras import Sequential, Model
import tensorflow as tf
import cv2
import warnings
warnings.filterwarnings('ignore')
import os
from keras.models import load_model
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img

from keras.utils.vis_utils import plot_model
from glob import glob
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
# from scipy.misc import imsave, imread, imresize
import re
import base64

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'GGFGFJ77Thh68jbbb'
db = pymysql.connect(host= "localhost",user= "root",passwd= "", database= 'dermato')
my_cursor = db.cursor()


csrf = CSRFProtect(app)

@app.route('/')
def home():
    return render_template('Home.html')

# @app.route('/')
# def someName():
#     cursor = db.cursor()
#     sql = "SELECT * FROM bills"
#     cursor.execute(sql)
#     results = cursor.fetchall()
#     return results

@app.route('/login')
def login():
    return render_template('Login.html')

# ---------------------------------------------------------------------------------------
# @app.route('/register', methods = ['POST', 'GET'])
# def login():
#     # return render_template('Login.html')
#     my_cursor = db.cursor()
#     if request.method == 'POST':
#         register_data = request.form
#         user_name = register_data['name']
#         contact = register_data['contact']
#         email = register_data['email']
#         password = register_data['password']

#         my_cursor.execute("INSERT INTO users(user_name, user_contact, user_email, user_password) VALUES(%s,%s,%s,%s)", (user_name, contact, email, password))
#         db.commit()
#         my_cursor.close()
#         return "Registration Successful!!"


#---------------------------------------------------------------------------------------

@app.route('/loginuser', methods =['GET', 'POST'])
def loginuser():
    msg = ''
    if request.method == 'POST': #and 'email' in request.form and 'password' in request.form:
        print("Now, you are in an if block of the fuction")
        print(request)
        # username = request.form['name']
        # contact = request.form['contact']
        email = request.form['email']
        password = request.form['password']
        # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        db = pymysql.connect(host= "localhost",user= "root",passwd= "", database= 'dermato')
        my_cursor = db.cursor()
        my_cursor.execute('SELECT * FROM users WHERE user_email = %s AND user_password = %s', (email, password ))
        # db.commit()
        account = my_cursor.fetchone()

        print("session:", session, "account:", account)
        if account:
            session['loggedin'] = True
            session['user_email'] = account[2]
            session['user_password'] = account[3]
            msg = 'Logged in successfully !'
            return render_template('dashboard.html')
        else:
            msg = 'Incorrect username / password !'
            return render_template('login.html', msg = msg)
    # return url_for('dashboard')
  
@app.route('/login')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'contact' in request.form:
        username = request.form['name']
        password = request.form['password']
        email = request.form['email']
        contact = request.form['contact']
        my_cursor = db.cursor()
        my_cursor.execute('SELECT * FROM users WHERE user_email = %s', (email ))
        # db.commit()
        account = my_cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        # elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        #     msg = 'Invalid email address !'
        # elif not re.match(r'[A-Za-z0-9]+', username):
        #     msg = 'Username must contain only characters and numbers !'
        elif not password or not email:
            msg = 'Please fill all the fields!'
        else:
            my_cursor.execute('INSERT INTO users(`user_name`, `user_contact`, `user_email`, `user_password`) VALUES (%s, %s, %s, %s)', (username, contact, email, password))
            db.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('login.html', msg = msg)



@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


res = VGG16(weights ='imagenet', include_top = False, 
               input_shape = (256, 256, 3)) 
x= res.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x)

x = Dense(256, activation ='relu')(x)
x = BatchNormalization()(x)

x = Dense(3, activation ='softmax')(x)
model = Model(res.input, x)
model.compile(optimizer =tf.keras.optimizers.RMSprop(learning_rate=0.0001),  #'Adam'
              loss ="categorical_crossentropy",  #sparse_categorical_crossentropy
              metrics =["categorical_accuracy"])  #sparse_categorical_accuracy
model.load_weights('model.h5')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    # return preds
    pred_class = decode_predictions(preds, top=1)   
    result = str(pred_class[0][0][1])
    return result

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))

labels = ["Papular Acne", "Pustular Acne", "Comedonal Acne"]
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    mdg = ''
    img = request.files['fileUpload']
    img.save("img.jpg")
    image = cv2.imread("img.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256,256))
    image = np.reshape(image, (1,256,256,3))

    pred = model.predict(image)
    pred = np.argmax(pred)
    msg = labels[pred]
    return render_template('dashboard.html', msg = msg)
    


    # print(request)
    # if request.method == 'POST':
    #     f = request.files['fileUpload']
    #     f.save(secure_filename(f.filename))
    #     print(f)
    #     # return 'file uploaded successfully'
    #     # model1.predict(f)

    #     basepath = os.path.dirname(__file__)
    #     file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    #     f.save(file_path)
    #     # return "Uploaded!!"

    #     pred = model_predict(file_path, model)
    #     return pred


if __name__ == "__main__":
    # app = Flask(__name__)
    csrf = CSRFProtect(app)
    app.run(debug = True)
