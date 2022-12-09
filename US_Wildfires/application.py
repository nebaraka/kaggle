from flask import Flask
import pandas as pd
import sklearn
import pickle

from flask import request, jsonify

application = app = Flask(__name__)

classes_decoder = {0: 'Arson',
                   1: 'Campfire',
                   2: 'Children',
                   3: 'Debris Burning',
                   4: 'Equipment Use',
                   5: 'Fireworks',
                   6: 'Lightning',
                   7: 'Miscellaneous',
                   8: 'Powerline',
                   9: 'Railroad',
                   10: 'Smoking',
                   11: 'Structure'}


def predict_fire_cause(fire_size, lat, lon, discovery_doy, dow, random_col, fire_year, month):
    pfile = open("model.pkl", "rb")
    model = pickle.load(pfile)

    y_predict = model.predict(pd.DataFrame({'FIRE_SIZE': [fire_size],
                                            'LATITUDE': [lat],
                                            'LONGITUDE': [lon],
                                            'DISCOVERY_DOY': [discovery_doy],
                                            'MY_DOW': [dow],
                                            'RANDOM': [random_col],
                                            'FIRE_YEAR': [fire_year],
                                            'MY_MONTH': [month]}))[0]

    return classes_decoder[y_predict]


@app.route("/")
def hello():
    return "A simple web service for accessing a ML model to classify fire cause."


@app.route("/fire", methods=["GET"])
def api_all():
    fire_size = request.args["fire_size"]
    lat = request.args["lat"]
    lon = request.args["lon"]
    discovery_doy = request.args["discovery_doy"]
    dow = request.args["dow"]
    random_col = request.args["random_col"]
    fire_year = request.args["fire_year"]
    month = request.args["month"]
    cause = predict_fire_cause(fire_size, lat, lon, discovery_doy, dow, random_col, fire_year, month)

    return jsonify(cause=cause)


if __name__ == '__main__':
    app.run()
