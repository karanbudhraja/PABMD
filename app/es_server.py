from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES
import glob
from flask import Flask, render_template, request
import os
import time

app = Flask(__name__)
app.config["INPUT_FILE"] = "sketches/demonstration.png"
app.config["ABM"] = "flocking"
app.config["SES"] = SignatureES(Elasticsearch(), distance_cutoff=1.0)

@app.route('/initialize', methods=["GET"])
def initialize():
    # clear existing indexes
    os.system("curl -XDELETE 'http://localhost:9200/_all'")

    # sleep to be sure of synchronization
    time.sleep(60)
    
    # index items
    #folderName = "/home/karan/storage/workspaces/bitbucket/_swarm-lfd-data/" + app.config["ABM"] + "/images_10"
    folderName = "/home/karan/storage/workspaces/bitbucket/_swarm-lfd-data/" + app.config["ABM"] + "/images"
    allFileNames = glob.glob(folderName + "/*.*")

    for imageName in sorted(allFileNames):
        print('elastic search initialize: ' + imageName)
        app.config["SES"].add_image(imageName)

    # sleep to be sure of synchronization
    time.sleep(60)
        
    return "initialization complete"


@app.route('/match', methods=["GET"])
def match():
    results = app.config["SES"].search_image(app.config["INPUT_FILE"])

    distanceResults = []
    for result in results:
        distanceResults.append((result['dist'], result['path']))

    # get top result
    [distance, path] = sorted(distanceResults)[0]
    os.system('cp ' + path + ' matches/')
    
    return "matched"

# code run at app start
initialize()
