from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt 
import numpy as np

app = Flask(__name__)

dic = {0 : 'Ayam Bakar', 1 : 'Bakso', 2 : 'Gado gado', 3 : 'Rendang', 4 : 'Sate'}

model = load_model('cnn_adam_categorical_crossentropy')

def predict_label(img_path):
   img = cv2.imread(img_path)
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cmap=plt.cm.binary)
   img = cv2.resize(img,(198,259))
   img = np.expand_dims(img, axis=0)
   pred = model.predict(img)
   prediction  = np.argmax(pred, axis=1)
   return dic[int(prediction)], pred


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p,pred = predict_label(img_path)

	return render_template("home.html", prediction = p, img_path = img_path,pred=pred)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)