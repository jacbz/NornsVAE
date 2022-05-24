import re
from datetime import datetime


class ConsoleFilter(object):
    def __init__(self, stream, log_filename):
        self.stream = stream
        self.pattern = re.compile(r'tensorflow|tf\.|layer\.|development server|GET |Log entry')
        self.triggered = False
        self.log_filename = log_filename

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if len(data) > 1:
                data = f"[{datetime.now().strftime('%H:%M:%S')}]\t{data}"

            if "Running on h" in data or self.pattern.search(data) is None:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True
            with open(self.log_filename, "a") as file:
                file.write(data)

    def flush(self):
        self.stream.flush()