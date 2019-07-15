sudo service elasticsearch start

export FLASK_APP=es_server.py
flask run --host=0.0.0.0 --port=5001
