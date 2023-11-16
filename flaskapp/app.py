from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('../tylers_best_model_Log_reg/saved_log_reg_model/model.pkl')
scaler = joblib.load('../tylers_best_model_Log_reg/saved_log_reg_model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract and scale features from form
            features = np.array([[
                float(request.form['battery_power']),
                int(request.form['blue']),
                float(request.form['clock_speed']),
                int(request.form['dual_sim']),
                int(request.form['fc']),
                int(request.form['four_g']),
                int(request.form['int_memory']),
                float(request.form['m_dep']),
                int(request.form['mobile_wt']),
                int(request.form['n_cores']),
                int(request.form['pc']),
                int(request.form['px_height']),
                int(request.form['px_width']),
                int(request.form['ram']),
                int(request.form['sc_h']),
                int(request.form['sc_w']),
                int(request.form['talk_time']),
                int(request.form['three_g']),
                int(request.form['touch_screen']),
                int(request.form['wifi']),
            ]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            # output the prediction in the terminal
            print("Model prediction :", prediction)
        except ValueError:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template('home.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run(debug=True, port= 5002)
