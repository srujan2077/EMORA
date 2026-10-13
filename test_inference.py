import urllib.request
import urllib.parse
import json
import mimetypes
import uuid

def encode_multipart_formdata(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for key, value in fields.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode('utf-8'))
    for key, (filename, filepath) in files.items():
        mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\nContent-Type: {mimetype}\r\n\r\n'.encode('utf-8'))
        with open(filepath, 'rb') as f:
            body.extend(f.read())
        body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
    return body, f'multipart/form-data; boundary={boundary}'

url = 'http://127.0.0.1:8000/predict/raw'
fields = {'text': 'I felt thrilled and overjoyed by the unexpected outcome, but also somewhat cautious about the future.'}
files = {
    'audio': ('sample_audio.wav', 'c:/SRUJAN/multimodal_ai/frontend/public/samples/sample_audio.wav'),
    'video': ('sample_video.mp4', 'c:/SRUJAN/multimodal_ai/frontend/public/samples/sample_video.mp4')
}

body, content_type = encode_multipart_formdata(fields, files)
req = urllib.request.Request(url, data=body, headers={'Content-Type': content_type}, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        print('Status Code:', response.getcode())
        print(json.dumps(json.loads(response.read().decode('utf-8')), indent=2))
except Exception as e:
    print('Error:', e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
