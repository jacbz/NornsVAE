import json
from flask import Flask
from flask import request
from interface import Interface

app = Flask(__name__)

@app.route("/")
def base():
  return 'Server is running...'

@app.route("/lookahead")
def lookahead():
  attr_values = json.loads(request.args.get("attr_values"))
  attribute = request.args.get("attribute")
  return json.dumps(interface.lookahead(attr_values, attribute))

@app.route("/reload")
def reload():
  interface.init()
  return {}


interface = Interface()