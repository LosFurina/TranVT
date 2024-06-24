import requests


class LarkHook:
    def __init__(self, url):
        self.url = url

    def send(self, data: dict):
        r = requests.post(self.url, data=data)
        print(r.text)
