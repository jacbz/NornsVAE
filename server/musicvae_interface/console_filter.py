import re
from datetime import datetime


class ConsoleFilter(object):
    def __init__(self, stream):
        self.stream = stream
        self.pattern = re.compile(r'tensorflow|tf\.|layer\.|development server')
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if self.pattern.search(data) is None:
                if len(data) > 1:
                    data = f"[{datetime.now().strftime('%H:%M:%S')}]\t{data}"
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()