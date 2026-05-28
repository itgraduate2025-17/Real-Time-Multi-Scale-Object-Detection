from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import os
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from twilio.rest import Client  # ✅ Added for WhatsApp alert

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Distance measurement constants
KNOWN_WIDTH = 0.5  # meters
FOCAL_LENGTH = 200  # pixels

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# Twilio Configuration
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_WHATSAPP_NUMBER = ""
RECIPIENT_WHATSAPP_NUMBER = ""

# Load YOLOv8 model
try:
    model = YOLO("14classes.pt")
    print("Model loaded successfully from best.pt")
except Exception as e:
    model = None
    print(f"Error loading the model: {e}")

# ✅ Send WhatsApp Alert
def send_whatsapp_alert(detected_class, confidence, distance):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message_body = f"⚠️ Alert: '{detected_class}' detected with {confidence:.2f} confidence.\nEstimated Distance: {distance:.2f} meters."
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=RECIPIENT_WHATSAPP_NUMBER
        )
        print(f"WhatsApp alert sent: SID={message.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_distance(known_width, focal_length, pixel_width):
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in {'png', 'jpg', 'jpeg', 'gif'}:
            try:
                img = cv2.imread(file_path)
                results = model(img)

                for result in results:
                    for box in result.boxes:
                        conf = box.conf[0].numpy()
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = box.xyxy[0].numpy()
                            cls = int(box.cls[0].numpy())
                            pixel_width = x2 - x1
                            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width)
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(img, f'{model.names[cls]} {conf:.2f}', (int(x1), int(y1) - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(img, f'Distance: {distance:.2f} m', (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            if model.names[cls] == "No Helmet":
                                send_whatsapp_alert(model.names[cls], conf, distance)

                detected_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detected_{filename}')
                cv2.imwrite(detected_path, img)

                return redirect(url_for('uploaded_file', filename=f'detected_{filename}'))
            except Exception as e:
                print(f"Error processing the uploaded file: {e}")
                return "Error processing the uploaded file."

        elif file_ext in {'mp4', 'avi'}:
            return redirect(url_for('process_video', filename=filename))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/live_detection')
def live_detection():
    return render_template('live_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_video/<filename>')
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(gen_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            try:
                results = model(frame)

                for result in results:
                    for box in result.boxes:
                        conf = box.conf[0].numpy()
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = box.xyxy[0].numpy()
                            cls = int(box.cls[0].numpy())
                            pixel_width = x2 - x1
                            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (int(x1), int(y1) - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f'Distance: {distance:.2f} m', (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            if model.names[cls] == "No Helmet":
                                send_whatsapp_alert(model.names[cls], conf, distance)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error during live detection: {e}")
                break

def gen_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        try:
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    conf = box.conf[0].numpy()
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].numpy()
                        cls = int(box.cls[0].numpy())
                        pixel_width = x2 - x1
                        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (int(x1), int(y1) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f'Distance: {distance:.2f} m', (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        if model.names[cls] == "No Helmet":
                            send_whatsapp_alert(model.names[cls], conf, distance)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Error processing video frame: {e}")
            break
    cap.release()

@app.route('/user')
def user():
    return render_template('user.html')  # file: templates/user.html

if __name__ == '__main__':
    app.run(debug=True)
