from flask import Flask,redirect,url_for

app = Flask(__name__)

@app.route("/")
def Home():
    return "<p>CREDIT_CARD _DEFAULT _PREDICTOR</p>"













if __name__ == '__main__':
    app.run(debug=True)
