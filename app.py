from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
#import StockPrice as SP
import re
import sqlite3
import pandas as pd
import numpy as np
import requests

import fakenews as fr
import PredictionNews as PN
import FetchReal as cd

from flask_table import Table, Col

    
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
        if not session.get('logged_in'):
                return render_template("login.html")
        else:
                return render_template('main.html')
        #return render_template('main.html')
@app.route('/registerpage',methods=['POST'])
def reg_page():
    return render_template("register.html")
        
@app.route('/loginpage',methods=['POST'])
def log_page():
    return render_template("login.html")

@app.route('/createdataset',methods=['POST'])
def createdataset():
        cd.process()
        return render_template('main.html')


@app.route('/main',methods=['POST'])
def main_page():
        path=request.form['datasetfile']
        print(path)
        return render_template('main.html')

@app.route('/countmodel',methods=['POST'])
def count_page():
        path=request.form['datasetfile']
        print(path)
        fr.MainProcessCount(path)
        return render_template('main.html')
   
@app.route('/tfidfmodel',methods=['POST'])
def tfidf_page():
        path=request.form['datasetfile']
        print(path)
        fr.MainProcessTfidf(path)
        return render_template('main.html')

@app.route('/ngrammodel',methods=['POST'])
def ngram_page():
        path=request.form['datasetfile']
        print(path)
        fr.MainProcessNgram(path)
        return render_template('main.html')



@app.route('/prediction',methods=['POST'])
def prediction_page():
        path=request.form['datasetfile']
        news=request.form['news']
        print(path)
        print(news)
        res,res1=PN.process(path,news)
        print(res)
        return render_template('main.html',message=res,message1=res1,message2=news)
@app.route('/register',methods=['POST'])
def reg():
        name=request.form['name']
        username=request.form['username']
        password=request.form['password']
        email=request.form['emailid']
        mobile=request.form['mobile']
        conn= sqlite3.connect("Database")
        cmd="SELECT * FROM login WHERE username='"+username+"'"
        print(cmd)
        cursor=conn.execute(cmd)
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Username Already Exists")
                return render_template("usernameexist.html")
        else:
                print("insert")
                cmd="INSERT INTO login Values('"+str(name)+"','"+str(username)+"','"+str(password)+"','"+str(email)+"','"+str(mobile)+"')"
                print(cmd)
                print("Inserted Successfully")
                conn.execute(cmd)
                conn.commit()
                conn.close() 
                return render_template("inserted.html")

@app.route('/login',methods=['POST'])
def log_in():
        #complete login if name is not an empty string or doesnt corss with any names currently used across sessions
        if request.form['username'] != None and request.form['username'] != "" and request.form['password'] != None and request.form['password'] != "":
                username=request.form['username']
                password=request.form['password']
                conn= sqlite3.connect("Database")
                cmd="SELECT username,password FROM login WHERE username='"+username+"' and password='"+password+"'"
                print(cmd)
                cursor=conn.execute(cmd)
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                if(isRecordExist==1):
                        session['logged_in'] = True
                        # cross check names and see if name exists in current session
                        session['username'] = request.form['username']
                        return redirect(url_for('index'))

        return redirect(url_for('index'))
        
@app.route("/logout")
def log_out():
    session.clear()
    return redirect(url_for('index'))

# /////////socket io config ///////////////
#when message is recieved from the client    
@socketio.on('message')
def handleMessage(msg):
    print("Message recieved: " + msg)
 
# socket-io error handling
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass


  
  
if __name__ == '__main__':
    socketio.run(app,debug=True,host='127.0.0.1', port=4000)
