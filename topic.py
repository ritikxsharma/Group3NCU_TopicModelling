
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import lda as ld
from pprint import pprint


app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method=='POST':
        txt = request.form['article']
        tpc = ld.chnge_to_list(txt) 
        # res = tpc
        # print(pprint(tpc))
        # print(len(tpc))
        return render_template('index.html', result = tpc)
    else:
        return render_template('index.html')


"""@app.route('/rslt', methods=['GET', 'POST'])
def show():
    if request.method=='POST':
        txt = request.form['article']
        tpc = ld.chnge_to_list(txt) 
        res = tpc
        print(res)
    return render_template('result.html',res=res)"""



if __name__ == "__main__":
    app.run(debug=True)