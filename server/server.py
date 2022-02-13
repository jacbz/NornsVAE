import json
from flask import Flask
from flask import request
from interface import Interface

app = Flask(__name__)

@app.route("/")
def base():
  return 'Server is running...'

@app.route("/sample")
def sample():
  n = int(request.args.get("n"))
  samples = interface.sample(n)
  return json.dumps(samples)

@app.route("/sample_and_interpolate")
def sample_and_interpolate():
  n = int(request.args.get("n"))
  samples = interface.sample_and_interpolate(n)
  return json.dumps(samples)

@app.route("/interpolate", methods=['POST'])
def interpolate():
  data = request.get_json()
  num_outputs = int(request.args.get('n'))
  samples = interface.interpolate(data[0], data[1], num_outputs)
  return json.dumps(samples)


@app.route("/interpolate_existing")
def interpolate_existing():
  num_outputs = int(request.args.get('n'))
  hash1 = int(request.args.get('hash1'))
  hash2 = int(request.args.get('hash2'))
  samples = interface.interpolate_existing(hash1, hash2, num_outputs)
  return json.dumps(samples)

@app.route("/attribute_arithmetics")
def attribute_arithmetics():
  attribute = request.args.get('attribute')
  num_outputs = int(request.args.get('n'))
  hash = int(request.args.get('hash'))
  samples = interface.attribute_arithmetics(attribute, hash, num_outputs)
  return json.dumps(samples)


interface = Interface()