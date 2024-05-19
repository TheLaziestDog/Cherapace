# This is Cherapace SF (Stream & Feed) code, this includes autofeeder mechanism and object detection
# YoloV8 & OpenCV 
from ultralytics import YOLO
import cv2
import math
from flask import Flask, render_template, Response, jsonify, request
import threading
import argparse
import gc
gc.collect()

# Initialize the Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Use /home/jg2gb-5/yolov8/cherapace/best.pt for Jetson
model = YOLO("/Users/baloon/Programming/Python/projects/Computer-Vision-Projects/cherapace/runs/detect/train4/weights/best.pt")

# Object classes
classNames = ["Lobster_AT_Inside", "Lobster_AT_Outside"]
lobster_inside = 0
lobster_outside = 0
lobster_total = 6
lobster_fed = False

@app.route('/')
def index():
    return render_template('index.html', lobster_inside=lobster_inside, lobster_outside=lobster_outside, lobster_fed=lobster_fed)

@app.route('/get_counts')
def get_counts():
    return jsonify({
        'lobster_inside': lobster_inside,
        'lobster_outside': lobster_outside
    })

@app.route('/feed_lobster', methods=['POST'])
def feed_lobster():
    feed()
    return jsonify({'status': 'success', 'message': 'Lobsters have been fed!'})

def feed():
    # Your feeding logic here
    print("Feeding lobster...")

# Object Detection
def detect_and_stream():
    global lobster_inside, lobster_outside
    while True:
        success, img = camera.read()
        if not success:
            continue

        # Remove device='mps' for Jetson use
        results = model(img, stream=True, device='mps')

        # Coordinates
        for r in results: 
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)
                
                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Object details
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1
                detecttext = f"{classNames[cls]} {confidence}"

                # current_lobster = lobster_inside + lobster_outsid
                # put lobster counter here:
                lobster_inside += 1
                
                # Will only show detection with confidence level of over 0.5 or 50%
                if confidence >= 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, detecttext, org, font, fontScale, color, thickness)
                else:
                    print("Confidence Below 0.5, Specifically --->", confidence)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concat frame one by one and show result

# Web Server
@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_webserver():
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

ap = None
args = None

# Multithreading
if __name__ == "__main__":
    # Construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="IP address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="Ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    
    thread_two = threading.Thread(target=run_webserver)
    thread_two.start()
    thread_two.join()

    camera.release()
    cv2.destroyAllWindows()
