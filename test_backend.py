import urllib.request
import json
try:
    with urllib.request.urlopen('http://127.0.0.1:5002/occasions') as response:
        data = json.loads(response.read().decode())
        print('Occasions:', data)
except Exception as e:
    print('Error:', str(e))