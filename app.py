from flask import Flask, render_template, request, Response, jsonify
import os
from detector import process_video

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_stats = {"global_count": 0, "region_count": 0, "classes": []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    if video:
        path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(path)
        return render_template('index.html', video_path=video.filename)
    return "No file uploaded", 400

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(process_video(video_path, current_stats), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(current_stats)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    # app.run(debug=True)
