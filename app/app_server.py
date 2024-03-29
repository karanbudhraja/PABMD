import os
import base64
import glob

from flask import Flask, render_template, request
from werkzeug import secure_filename


UPLOAD_FOLDER = "sketches/"
UPLOAD_FILENAME = "demonstration.png"
MATCH_FOLDER = "matches/"
ABM = "flocking"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def root():
    return "You are here: /"


@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():

    if (request.method == "POST"):
        # get the base64 encoded image
        encodedImage = request.form["encodedImage"]

        # save to disk
        with open("demonstration.jpg", "wb") as outFile:
            outFile.write(base64.b64decode(encodedImage))

        # convert to png
        os.system("convert demonstration.jpg demonstration.png")
        os.system("rm -f demonstration.jpg")
            
        # clear old data
        os.system("rm -f " + UPLOAD_FOLDER + "*")
        os.system("rm -f " + MATCH_FOLDER + "*")

        # save file in selected location
        fileName = os.path.join(app.config["UPLOAD_FOLDER"], UPLOAD_FILENAME)
        os.system("mv demonstration.png " + fileName)

        # initiate matching and call pipeline background process
        os.system("rm -rf " + "../data/predictions/" + ABM + "/" + MATCH_FOLDER + "/*")
        os.system('curl localhost:5001/match')
        os.system("./do_pipeline.sh &")
        
    # refresh will be a get request
    return render_template("redirect.html")


@app.route("/redirect")
def redirect():
    # get background process status
    fileExists = os.path.isfile("../data/predictions/" + ABM + "/" + MATCH_FOLDER + "predicted_alps.txt")

    return str(fileExists)


@app.route('/results')
def results():
    return render_template("results.html")


@app.route('/alps')
def get_suggested_alps():
    # get suggested ALPs from file
    # this will be the response sent
    with open("../data/predictions/" + ABM + "/" + MATCH_FOLDER + "predicted_alps.txt") as inFile:
        prediction = inFile.readlines()[0]
        suggestedAlps = prediction.split(" (")[0]

    return suggestedAlps


@app.route('/match')
def get_match():
    # return matched image as json
    imageName = glob.glob("matches/*.png")[0]
    with open(imageName, "rb") as imageFile:
        imageString = base64.b64encode(imageFile.read())

    return imageString


@app.route('/prediction')
def get_prediction():
    # return matched image as json
    imageFolder = glob.glob("../data/predictions/" + ABM + "/" + MATCH_FOLDER + "/images_*")[0]
    imageName = imageFolder + "/0_200.png"
    with open(imageName, "rb") as imageFile:
        imageString = base64.b64encode(imageFile.read())

    return imageString
