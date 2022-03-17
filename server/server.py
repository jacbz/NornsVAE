import json
import random
import threading
import time
from datetime import datetime
from uuid import getnode as get_mac

from flask import Flask
from flask import request
from interface import Interface
import requests
import hashlib

app = Flask(__name__)

current_job = 0
current_data = None

uid = hashlib.md5(hex(get_mac()).encode('utf-8')).hexdigest()[0:6]
app_log = []

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

    append_to_log(response, "lookahead")

    return response_str


def do_job(title, job):
    global current_job
    job_id = str(random.randint(0, 10000000000))
    current_job = job_id

    append_to_log({
        "job_id": job_id
    }, title)

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

    return do_job("request_lookahead", lambda: lookahead_target(attr_values, attribute))


@app.route("/replace")
def replace():
    dict1 = json.loads(request.args.get("dict1"))
    dict2 = json.loads(request.args.get("dict2"))

    def replace_target():
        return interface.replace(dict1, dict2)

    return do_job("request_replace", lambda: replace_target())


@app.route("/reload")
def reload():
    def reload_target():
        interface.init()
        return interface.lookahead()
    return do_job("request_reload", lambda: reload_target())


@app.route("/log")
def log():
    log_data = request.args.get("data")

    def add_to_log(log_data):
        for json_data in log_data.split(";"):
            data = json.loads(json_data)
            data["source"] = "client"
            append_to_log(data)

    thread = threading.Thread(target=lambda: add_to_log(log_data))
    thread.daemon = True
    thread.start()
    return 'OK'


def send_log_to_server():
    print(f"Sending {len(app_log)} log items to server...")
    requests.post("http://localhost:3000/log", json=app_log)
    app_log.clear()


def logging_thread():
    while True:
        if len(app_log) > 0:
            send_log_to_server()
        time.sleep(5)


def append_to_log(data, type=None):
    if "source" not in data:
        data["source"] = "server"
    if "time" not in data:
        data["time"] = datetime.utcnow().isoformat()
    if type is not None:
        data["type"] = type
    data["uid"] = uid
    app_log.append(data)
    print(data)

threading.Thread(target=logging_thread).start()
interface = Interface()
append_to_log({}, "init_server")