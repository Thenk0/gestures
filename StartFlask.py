from flask.wrappers import Response
from flask import Flask, render_template
from Start import generate, get_frames
import threading

app = Flask(__name__)


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html")


def start_thread():
    t = threading.Thread(target=get_frames)
    t.daemon = True
    t.start()


if __name__ == "__main__":
    print("begin")
    start_thread()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
