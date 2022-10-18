import pandas as pd
from flask import Flask, request, render_template
from keras.models import Model
import tensorflow as tf

app = Flask("__name__")


@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']

    
    cnn_model = tf.keras.models.load_model('cnn_model')
    
    data = [[float(inputQuery1), float(inputQuery2), float(inputQuery3), float(inputQuery4), 
             float(inputQuery5), float(inputQuery6), float(inputQuery7), float(inputQuery8),
             float(inputQuery9), float(inputQuery10), float(inputQuery11), float(inputQuery12)]]
    
    df = pd.DataFrame(data, 
                      columns = [ 'credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
                                            'credit_card', 'active_member', 'estimated_salary',
                                            'country_France', 'country_Germany', 'country_Spain'])
    
    
    #final_df=pd.concat([new_df__dummies, new_dummy], axis=1)
  
    probablity = cnn_model.predict(data)
    if probablity>=0.5:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'])
    
app.run()