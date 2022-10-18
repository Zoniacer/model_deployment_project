import pandas as pd
import pickle
from keras.models import Model
import tensorflow as tf

cnn_model = tf.keras.models.load_model('cnn_model')

data = [[0.538	, 1, 0.324324, 0.2, 0, 0, 1, 
             1, 0.506735, 1, 0, 0]]
    
df = pd.DataFrame(data, columns = [ 'credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
                                            'credit_card', 'active_member', 'estimated_salary',
                                            'country_France', 'country_Germany', 'country_Spain'])
print((cnn_model.predict(df)))

