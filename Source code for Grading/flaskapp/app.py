from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('./saved_model/logistic.pkl')
scaler = joblib.load('./saved_model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract and scale features from form
            blue = (lambda x: 1 if x == 'on' else 0)(request.form.get('blue'))
            dualSim = (lambda x: 1 if x == 'on' else 0)(request.form.get('dual_sim'))
            fourG = (lambda x: 1 if x == 'on' else 0)(request.form.get('four_g'))
            threeG = (lambda x: 1 if x == 'on' else 0)(request.form.get('three_g'))
            touchScreen = (lambda x: 1 if x == 'on' else 0)(request.form.get('touch_screen'))
            wifi = (lambda x: 1 if x == 'on' else 0)(request.form.get('wifi'))

            features = np.array([[
                int(request.form.get('battery_power')),
                blue,
                float(request.form.get('clock_speed')),
                dualSim,
                int(request.form['fc']),
                fourG,
                int(request.form.get('int_memory')),
                float(request.form.get('m_dep')),
                int(request.form.get('mobile_wt')),
                int(request.form.get('n_cores')),
                int(request.form.get('pc')),
                int(request.form.get('px_height')),
                int(request.form.get('px_width')),
                int(request.form.get('ram')),
                int(request.form.get('sc_h')),
                int(request.form.get('sc_w')),
                int(request.form.get('talk_time')),
                threeG,
                touchScreen,
                wifi,
            ]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            # output the prediction in the terminal
            print("Model prediction :", prediction)

            switch = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }

            prediction = switch.get(prediction)
        except ValueError:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template('home.html', 
                            prediction_result=prediction)
    
if __name__ == '__main__':
    app.run(debug=True, port= 5002)