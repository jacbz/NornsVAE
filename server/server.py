import json
import random
import threading

from flask import Flask
from flask import request
from interface import Interface

app = Flask(__name__)

current_job = 0
current_data = None

@app.route("/")
def base():
  return 'Server is running...'


@app.route("/sync")
def sync():
  global current_data
  if current_data is None:
    return {}

  response = {
    'job_id': current_job,
    'data': current_data
  }
  response_str = json.dumps(response)
  current_data = None
  return response_str


@app.route("/lookahead")
def lookahead():
  attr_values = json.loads(request.args.get("attr_values"))
  attribute = request.args.get("attribute")

  thread = threading.Thread(target=lambda: do_work(attr_values, attribute))
  thread.daemon = True
  thread.start()

  global current_job
  job_id = str(random.randint(0, 10000000000))
  current_job = job_id

  return job_id


@app.route("/reload")
def reload():
  interface.init()
  return "Reloaded!"


def do_work(attr_values, attribute):
  global current_data
  current_data = interface.lookahead(attr_values, attribute)


interface = Interface()