from flask import Flask,render_template,request
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

# @app.route('/test', methods=['GET'])
# def test():
#     return 'Pinging Model Application!!'

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
	return render_template("index.html",static_url_path='/static')

@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
def predict():
		try:
			#reading the inputs given by the user
			vehicle_name = str(request.form['vehiclename'])
			battery = float(request.form['battery'])
			acceleration = float(request.form['acceleration'])
			topspeed = float(request.form['topspeed'])
			distance = float(request.form['distance'])
			efficiency = float(request.form['efficiency'])
			fastcharge = float(request.form['fastcharge'])

			# load df_col list
			with open('df_cols.bin', 'rb') as f_in:
				df_cols = pickle.load(f_in)
				f_in.close()
			# load model
			with open('ev_model.bin', 'rb') as f_in:
				loaded_model = pickle.load(f_in)
				f_in.close()


			def preporcess_vehiclename_col(arr):
				vehicle_name = arr[0]
				col = f"vehicle_name_{vehicle_name}"
				idx = df_cols.index(col)
				encoded_arr = arr + [0.0 for i in range(len(df_cols)-len(arr)+1)]
				encoded_arr[idx+1] = 1.0
				del encoded_arr[0]
				# print(encoded_arr)
				return encoded_arr

			# def scale_input(arr):
			# 	scaler = MinMaxScaler()
			# 	# convert inputs to array
			# 	arr = np.array(arr)
			# 	# print(arr)
			# 	scaled_inputs = scaler.fit_transform(arr.reshape(-1,1))
			# 	# scaled_inputs = scaler.transform(arr.reshape(1,-1))
			# 	scaled_inputs = [[i[0] for i in list(scaled_inputs)]]

			# 	return scaled_inputs

			def reshape(arr):
				arr = np.array(arr)
				reshaped = arr.reshape(1,-1)
				return reshaped
			
			# get inputs
			user_inputs = [vehicle_name,battery,acceleration,topspeed,distance,efficiency,fastcharge]
			# print(user_inputs)

			#preprocess vehicle name
			data = preporcess_vehiclename_col(user_inputs)

			# # scaled data
			# scaled_data = scale_input(data)

			# reshape data
			reshaped_data = reshape(data)

			# loading the model file from the storage
			# predictions using the loaded model file
			prediction = loaded_model.predict(reshaped_data)

			# showing the prediction results in a UI
			return render_template('predict.html',prediction=round(prediction[0]))
		except Exception as e:
			print('The Exception message is: ',e)
			return 'something is wrong'


# if __name__ == '__main__':
# 	app.run(debug=True)