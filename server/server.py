import json
import os
import random
import re
import sys
import threading
import time
import datetime
from uuid import getnode as get_mac
from pathlib import Path
from flask import Flask
from flask import request
import requests
import hashlib
import webbrowser

from musicvae_interface.console_filter import ConsoleFilter
from musicvae_interface.interface import Interface

app = Flask(__name__)

current_job = 0
current_data = None

app_log_buffer = []
uid = hashlib.md5(hex(get_mac()).encode('utf-8')).hexdigest()

client_time_offset = datetime.timedelta(0)


@app.route("/")
def base():
    return 'Server is running...'


@app.route("/init")
def init():
    global client_time_offset
    client_unix_time = datetime.datetime.utcfromtimestamp(float(request.args.get("time")))
    client_time_offset = (datetime.datetime.utcnow() - client_unix_time)
    print(f"Client has reconnected, client time offset {client_time_offset}")
    return 'OK'


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
        interface.init_random()
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
    count = len(app_log_buffer)
    print(f"Logging {count} item{'s' if count != 1 else ''}")
    response = requests.post("https://nornsvae-logging.medien.ifi.lmu.de", json=app_log_buffer)
    if response.status_code == 200:
        app_log_buffer.clear()
    else:
        print(f"Error sending log to server: {response.status_code} {response.reason}", file=sys.stderr)


def logging_thread():
    while True:
        if len(app_log_buffer) > 0:
            send_log_to_server()
        time.sleep(5)


def append_to_log(data, type=None):
    if "source" not in data:
        data["source"] = "server"
        data["time"] = datetime.datetime.utcnow().isoformat()
    else:
        time = datetime.datetime.utcfromtimestamp(float(data["time"])) + client_time_offset
        data["time"] = time.isoformat()
    if type is not None:
        data["type"] = type
    data["uid"] = uid
    app_log_buffer.append(data)
    print(f"Log entry: {data}")


def ask_for_email(email):
    regex = re.compile(r'[^@]+@[^@]+\.[^@]+')
    while not re.fullmatch(regex, email):
        print("Please enter your email:")
        email = input()
    append_to_log({"data": { "email": email} }, "email")
    return email


if __name__ == '__main__':
    # init console filter and log
    log_filename = Path(f"log_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    log_filename.touch(exist_ok=True)

    sys.stdout = ConsoleFilter(sys.stdout, log_filename)
    sys.stderr = ConsoleFilter(sys.stderr, log_filename)

    opened_pre_questionnaire = os.path.exists("pre_questionnaire")

    print("Welcome to NornsVAE!")
    print("As part of my master thesis, I'm researching how machine learning can be applied "
          "to an interactive music generation context.\n")
    if not opened_pre_questionnaire:
        print("You are kindly asked to fill out a Pre-Questionnaire first before you use NornsVAE.")
    print("After each session with NornsVAE, you will be asked to fill out a short questionnaire on the specific session.")
    print("After a few weeks, you will be asked to fill out a Post-Questionnaire on your experiences as a whole.")
    print("Bandcamp vouchers will be randomly awarded to participants after the end of the user study.\n")

    # email handling
    email_filename = Path('email')
    email_filename.touch(exist_ok=True)
    with open("email", "r+") as email_file:
        email = ask_for_email(email_file.read())
        email_file.seek(0)
        email_file.write(email)
        email_file.truncate()

    if not opened_pre_questionnaire:
        pre_questionnaire_url = f'https://jacobz.limesurvey.net/257256?newtest=Y&uid={uid}&email={email}'
        print("Please complete the Pre-Questionnaire before using NornsVAE. Thank you so much!")
        print(f"Please press the [ENTER] key now to open the Pre-Questionnaire. You can also open it manually: {pre_questionnaire_url}")
        x = input()
        webbrowser.open(pre_questionnaire_url)
        with open('pre_questionnaire', 'w') as f:
            f.write(pre_questionnaire_url)

    print("All actions performed on the client and server are sent to the logging server for evaluation as part of my thesis.")
    print("By using this software, you agree to logging. When the user study is completed, logging will be removed.")
    print("Thank you!\n")

    # load MusicVAE model
    print("Loading machine learning model...")
    interface = Interface("assets")

    # log server init
    append_to_log({}, "init_server")
    print(f"Your user ID is {uid}")

    # start logging thread
    threading.Thread(target=logging_thread).start()

    # run Flask server without showing banner
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app.run(host="0.0.0.0", port=6123)
