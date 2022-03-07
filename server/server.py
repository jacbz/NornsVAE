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


def do_job(job):
    global current_job
    job_id = str(random.randint(0, 10000000000))
    current_job = job_id

    def target():
        global current_data
        data = job()
        if current_job == job_id:
            current_data = data

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

    return job_id


@app.route("/lookahead")
def lookahead():
    attr_values = json.loads(request.args.get("attr_values"))
    attribute = request.args.get("attribute")

    def lookahead_target(attr_values, attribute):
        return interface.lookahead(attr_values, attribute)

    return do_job(lambda: lookahead_target(attr_values, attribute))


@app.route("/replace")
def replace():
    dict1 = json.loads(request.args.get("dict1"))
    dict2 = json.loads(request.args.get("dict2"))

    def replace_target():
        interface.replace(dict1, dict2)
        return interface.lookahead()

    return do_job(lambda: replace_target())


@app.route("/reload")
def reload():
    def reload_target():
        interface.init()
        return interface.lookahead()
    return do_job(lambda: reload_target())


interface = Interface()
