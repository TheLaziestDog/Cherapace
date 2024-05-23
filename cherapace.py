import threading
import argparse
import datetime
import gc
import queue
gc.collect()

# Object Detection
from ultralytics import YOLO
import cv2
import math

# Flask
from flask import Flask, render_template, Response, jsonify, request

# Servo (libraries not included on mac version)
feed_duration = 0.8  # format in second

# Initialize the Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Sensors
water_param = [0, 0, 0]

# Use /home/jg2gb-5/yolov8/cherapace/best.pt for Jetson
model = YOLO("/Users/baloon/Programming/Python/projects/Computer-Vision-Projects/cherapace/runs/detect/train5/weights/best.pt")

# Object classes
classNames = ["Lobster_AT_Inside", "Lobster_AT_Outside"]
lobster_inside = 0
lobster_outside = 0
lobster_total = 6
inConfBelow = 0  # default is 0 to prevent error
outConfBelow = 0  # default is 0 to prevent error
lobster_fed = 0

# shared queue for frames
frame_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html', lobster_inside=lobster_inside, lobster_outside=lobster_outside, lobster_fed=lobster_fed)

@app.route('/get_counts')
def get_counts():
    fed_count = lobster_fed
    return jsonify({
        'lobster_inside': lobster_inside,
        'lobster_outside': lobster_outside,
        "lobster_fed": fed_count
    })

@app.route('/feed_lobster', methods=['POST'])
def feed_lobster():
    feed()
    return jsonify({'status': 'success', 'message': 'Lobsters have been fed!'})

# Track the last feeding times
lastMorningFeed = None
lastEveningFeed = None
pastDay = None

def feed():
    global lobster_fed
    print(f"Feeding.....")
    
    lobster_fed += 1

def detect_and_stream():
    global lastMorningFeed, lastEveningFeed, pastDay, lobster_fed, lobster_inside, lobster_outside, inConfBelow, outConfBelow
    while True:
        # Update time
        currentDay = datetime.date.today()
        currentHour = datetime.datetime.now().hour
        
        # Reset lobster_fed if a new day has started
        if currentDay != pastDay:
            pastDay = currentDay
            
            lobster_fed = 0
            print("New day detected, resetting lobster_fed to 0")
        
        # Capture camera stream
        success, img = camera.read()
        if not success:
            continue

        # Remove device='mps' for Jetson use
        results = model(img, stream=True, device='mps')

        # Coordinates
        for r in results:
            boxes = r.boxes
            # Count how many objects (per class) appear
            class_ids = r.boxes.cls.tolist()
            class_ids = [int(class_id) for class_id in class_ids]
            class_counts = {class_id: class_ids.count(class_id) for class_id in set(class_ids)}

            # If the ml model is not good enough, 
            # please only run this loop when the predict confidence is more than 0.5
            if lobster_inside + lobster_outside <= lobster_total:
                for class_id, count in class_counts.items():
                    class_name = classNames[class_id]
                    if count <= lobster_total:
                        if class_name == classNames[0]:
                            lobster_inside = count - inConfBelow
                        if class_name == classNames[1]:
                            lobster_outside = count - outConfBelow
                    print(f"{class_name}: {count}")

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                inConfBelow = 0  # To not count In the prediction below 0.5
                outConfBelow = 0  # To not count Out the prediction below 0.5
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
                
                # Will only show detection with confidence level of over 0.5 or 50%
                if confidence >= 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, detecttext, org, font, fontScale, color, thickness)
                else:
                    print("Confidence Below 0.5, Specifically --->", confidence)
                
                if confidence <= 0.5 and classNames[cls] == "Lobster_AT_Inside":
                    inConfBelow += 1
                elif confidence <= 0.5 and classNames[cls] == "Lobster_AT_Outside":
                    outConfBelow += 1

                # Morning feeding time: 7 AM to 10 AM
                if 7 <= currentHour < 10:
                    if lastMorningFeed != currentDay:
                        if lobster_fed < 3:
                            if classNames[cls] == "Lobster_AT_Outside" and confidence >= 0.5:
                                feed()
                                lastMorningFeed = currentDay
                            elif classNames[cls] != "Lobster_AT_Outside":
                                print("Lobster not outside in the morning")
                        else:
                            print("Already fed in the morning")
                    else:
                        print("Already fed this morning")
                
                # Evening feeding time: 6 PM to 9 PM
                elif 18 <= currentHour < 21:
                    if lastEveningFeed != currentDay:
                        if lobster_fed < 3:
                            if classNames[cls] == "Lobster_AT_Outside" and confidence >= 0.5:
                                feed()
                                lastEveningFeed = currentDay
                            elif classNames[cls] != "Lobster_AT_Outside":
                                print("Lobster not outside in the evening")
                        else:
                            print("Already fed in the evening")
                    else:
                        print("Already fed this evening")
                else:
                    print("It's not feeding time now.")
        
        print(f"Lobster have been feeded {lobster_fed} time")

        # Encode frame and put it into the queue
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        if frame_queue.full():
            frame_queue.get()  # Discard the oldest frame if the queue is full
        frame_queue.put(frame)

@app.route('/reset_detection', methods=['POST'])
def reset_detection():
    global lobster_inside, lobster_outside
    lobster_inside = 0
    lobster_outside = 0
    return jsonify({"inside": lobster_inside, "outside": lobster_outside})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

args = None
ap = None

def run_webserver():
    global args, ap
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

def main():
    global args, ap
    # Construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="IP address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="Ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # Create threads
    webserver_thread = threading.Thread(target=run_webserver)
    detection_thread = threading.Thread(target=detect_and_stream)

    # Start threads
    webserver_thread.start()
    detection_thread.start()

    # Join threads
    webserver_thread.join()
    detection_thread.join()

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

# Multithreading
if __name__ == "__main__":
    main()
