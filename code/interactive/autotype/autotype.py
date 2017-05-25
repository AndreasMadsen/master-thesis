
import sys
import os.path as path
from http.server import BaseHTTPRequestHandler, HTTPServer

thisdir = path.dirname(path.realpath(__file__))
data_content = bytes(sys.stdin.read(), 'utf-8')


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_index()
        elif self.path == '/data.json.newline':
            self.serve_data()
        elif self.path == '/script.js':
            self.serve_script()
        else:
            # Send response status code
            self.send_response(404)
            self.end_headers()

    def serve_index(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        with open(path.join(thisdir, 'index.html'), 'rb') as script_file:
            self.wfile.write(script_file.read())

    def serve_data(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        self.wfile.write(data_content)

    def serve_script(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/javascript')
        self.end_headers()

        with open(path.join(thisdir, 'script.js'), 'rb') as script_file:
            self.wfile.write(script_file.read())


# Start server
server_address = ('127.0.0.1', 8080)
httpd = HTTPServer(server_address, RequestHandler)
print('running server...')
httpd.serve_forever()
