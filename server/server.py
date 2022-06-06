import datetime
import glob
import hashlib
import json
import os
import random
import sys
import threading
import time
import webbrowser
import zipfile
from pathlib import Path
from uuid import getnode as get_mac

from flask import Flask
from flask import request

from musicvae_interface.console_filter import ConsoleFilter
from musicvae_interface.interface import Interface

app = Flask(__name__)

current_job = 0
current_data = None

uid = hashlib.md5(hex(get_mac()).encode('utf-8')).hexdigest()

client_time_offset = datetime.timedelta(0)

log_has_changed = False


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
        log_array = log_data.split(";")
        for json_data in log_array:
            data = json.loads(json_data)
            data["source"] = "client"
            append_to_log(data)

    thread = threading.Thread(target=lambda: add_to_log(log_data))
    thread.daemon = True
    thread.start()
    return 'OK'


def log_zipping_thread():
    global log_has_changed
    while True:
        if log_has_changed:
            Path('logs.zip').unlink(missing_ok=True)
            with zipfile.ZipFile('logs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as log_zip:
                for f in glob.glob("*.log"):
                    log_zip.write(f)
            log_has_changed = False
        time.sleep(60)


def append_to_log(data, type=None):
    global log_has_changed
    if "source" not in data:
        data["source"] = "server"
        data["time"] = datetime.datetime.utcnow().isoformat()
    else:
        time = datetime.datetime.utcfromtimestamp(float(data["time"])) + client_time_offset
        data["time"] = time.isoformat()
    if type is not None:
        data["type"] = type
    data["uid"] = uid
    log_has_changed = True
    print(f"Log entry: {data}\n")


if __name__ == '__main__':
    # init console filter and log
    log_filename = Path(f"log_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    log_filename.touch(exist_ok=True)

    sys.stdout = ConsoleFilter(sys.stdout, log_filename)
    sys.stderr = ConsoleFilter(sys.stderr, log_filename)

    user_study = os.path.exists("user_study")

    print("Welcome to NornsVAE!")

    if user_study:
        with open('user_study', 'r+') as file:
            try:
                d = datetime.datetime.fromtimestamp(float(file.read()))
            except ValueError:
                d = None

            if d is not None and (datetime.datetime.now() - d).days > 7:
                post_questionnaire_url = f'https://jacobz.limesurvey.net/179125?newtest=Y&uid={uid}'
                print("It has been seven days since you first used NornsVAE.")
                print("You are kindly asked to complete the Post-Questionnaire now. Thank you so much!")
                print(f"Please press the [ENTER] key now to open the Post-Questionnaire. "
                      f"You can also open it manually: {post_questionnaire_url}")
                y = input()
                webbrowser.open(post_questionnaire_url)
                file.truncate(0)
    else:
        print("As part of my master thesis, I'm researching how machine learning can be applied "
          "to an interactive music generation context.")
        print("For that purpose, I’m conducting a user study. Bandcamp vouchers will be randomly awarded to participants.")
        print("Would you like to participate in the user study? [Y/N]")

        choice = input().lower()
        if 'n' in choice:
            open('user_study', 'a').close()
            print(":( If you change your mind, simply delete the 'user_study' file in the folder.")
        else:
            print("You are kindly asked to fill out a Pre-Questionnaire before you use NornsVAE.")
            pre_questionnaire_url = f'https://jacobz.limesurvey.net/257256?newtest=Y&uid={uid}'
            print(f"Please press the [ENTER] key now to open the Pre-Questionnaire. You can also open it manually: {pre_questionnaire_url}")
            x = input()
            webbrowser.open(pre_questionnaire_url)
            with open('user_study', 'w') as f:
                f.write(str(time.time()))
            print("Thank you! After seven days, you will be asked to fill out a Post-Quesionnaire on your experiences.\n")

    # load MusicVAE model
    print("Loading machine learning model...")
    interface = Interface("assets")

    # log server init
    append_to_log({}, "init_server")
    print(f"Your anonymized user ID is {uid}")

    # start log zipping thread
    threading.Thread(target=log_zipping_thread).start()

    # run Flask server without showing banner
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app.run(host="0.0.0.0", port=6123)
