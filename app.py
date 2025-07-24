import cv2
import numpy as np
import torch
from ultralytics import YOLO
import socket
import struct
from canlib import canlib, kvadblib
import json
import logging
import datetime
import os
import time
import sys
import signal
from flask import (Flask, render_template, Response,
                    jsonify, request, send_from_directory,
                    stream_with_context, session)
import threading, queue
from flask_session import Session
import subprocess

IGNORE_TIMER = True  # Set to True to ignore timing in the Timer class
stop_event = threading.Event()

with open('config.json') as f:
    default_config = json.load(f)

def list_can_channels():
    num_channels = canlib.getNumberOfChannels()
    result = [f"Found {num_channels} channels"]
    for ch in range(num_channels):
        chd = canlib.ChannelData(ch)
        info = f"{ch}. {chd.channel_name} ({chd.card_upc_no.product()}:{chd.card_serial_no}/{chd.chan_no_on_card})"
        result.append(info)
    return result

class Timer:
    def __init__(self, name="Block", ignore=IGNORE_TIMER):
        self.name = name
        self.ignore = ignore

    def __enter__(self):
        if self.ignore:
            return self
        self.start = time.perf_counter()
        return self  # optional, if you want to access elapsed time mid-block

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ignore:
            return
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"{self.name} took {self.elapsed:.4f} seconds")

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def get_can_busrate(busrate):
    can_baudrates = {
        100000: canlib.canBITRATE_100K,
        125000: canlib.canBITRATE_125K,
        250000: canlib.canBITRATE_250K,
        500000: canlib.canBITRATE_500K,
        1000000: canlib.canBITRATE_1M
    }
    return can_baudrates.get(busrate, canlib.canBITRATE_500K)

def find_intersection(line1_p1, line1_p2, line2_p1, line2_p2=None):
    if isinstance(line2_p1, list):
        line2_p2 = line2_p1[1]
        line2_p1 = line2_p1[0]
    if line2_p2 is None:
        return None  # Invalid input
    # Calculate coefficients for line1 (from lane boundary)
    A1 = line1_p2[1] - line1_p1[1]
    B1 = line1_p1[0] - line1_p2[0]
    C1 = A1 * line1_p1[0] + B1 * line1_p1[1]

    # Calculate coefficients for line2 (vehicle reference line)
    A2 = line2_p2[1] - line2_p1[1]
    B2 = line2_p1[0] - line2_p2[0]
    C2 = A2 * line2_p1[0] + B2 * line2_p1[1]

    matrix_A = np.array([[A1, B1], [A2, B2]])
    matrix_C = np.array([C1, C2])

    if np.linalg.det(matrix_A) == 0:
        return None  # Lines are parallel

    intersection = np.linalg.solve(matrix_A, matrix_C)
    return (int(intersection[0]), int(intersection[1]))


# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class offsetToCenterLine:
    def __init__(self):
        self.DEBUG_LEVEL = 1
        self.logger = None
        self.config = None

        self.sock = None

        self.DBC = None
        self.canMsg = None
        self.can = None

        self.source = None
        self.source_type = None

        self.cap = None

    def setup_logger(self):
        self.logger = logging.getLogger("offsetToCenterLine")
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
        self.logger.setLevel(logging.DEBUG)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.config.get("LOG", {}).get("ENABLED", False):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f"{timestamp}.log"
            os.makedirs("log", exist_ok=True)
            file_name = os.path.join("log", file_name)
            level = self.config.get("LOG", {}).get("LEVEL", "INFO")
            self.file_handler = logging.FileHandler(file_name)
            self.file_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

    def load_config(self, config=None):
        self.config = None
        if config is not None:
            self.config = config
            if self.logger:
                self.logger.info("Configuration loaded from user input.")
            return 0
        try:
            with open('config.json', 'r') as f:
                config_ = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file config.json not found. return 1")
            return 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the configuration file config.json. return 2")
            return 2
        self.config = config_
        return 0
        
    def open_socket(self):
        self.sock = None
        self.logger.info("Opening socket...")
        if self.config is None:
            return 1
        if self.config.get('UDP', {}).get('ENABLED', False) is not True:
            self.logger.warning("UDP is not enabled in the configuration. return 1")
            return 1
        try:
            host = str(self.config.get('UDP', {}).get('ADDR', 'localhost'))
            port = self.config.get('UDP', {}).get('PORT', 5000)
            self.logger.info(f"Connecting to {host}:{port}...")
            if not isinstance(port, int):
                self.logger.error(f"Invalid port number: {port}. It should be an integer. return 2")
                return 2
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.connect((host, port))
            self.logger.info(f"Connected to {host}:{port}")
        except socket.error as e:
            self.logger.error(f"Socket error: {e}, return 3")
            return 3
        return 0

    def close_socket(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            self.logger.info("Socket closed.")
        else:
            self.logger.warning("Socket was not open.")

    def send_udp_message(self, data):
        if not self.sock:
            self.logger.debug("Socket is not initialized., return 1")
            return 1
        try:
            msg = struct.pack('!f', data)
            self.sock.send(msg)
            self.logger.debug(f"Sent UDP message: {data}, return 0")
            return 0
        except socket.error as e:
            self.logger.error(f"Error sending UDP message: {e}, return 2")
            return 2
        except Exception as e:
            self.logger.error(f"Unexpected error sending UDP message: {e}, return 3")
            return 3

    def open_can(self):
        self.can = None
        self.canMsg = None
        self.logger.info("Opening CAN channel...")
        if self.config is None:
            return 1
        if self.config.get('CAN', {}).get('ENABLED', False) is not True:
            self.logger.warning("CAN is not enabled in the configuration. return 1")
            return 1
        try:
            dbc_file = self.config.get('CAN', {}).get('DBC', '')
            if not dbc_file:
                self.logger.error("DBC file path is empty. return 2")
                return 2
            self.DBC = kvadblib.Dbc(dbc_file)
            self.logger.info(f"Opened CAN DBC: {dbc_file}")
        except kvadblib.exceptions.KvdGeneralError as e:
            self.logger.error(f"Error opening CAN DBC: {e}, return 3")
            return 3
        try:
            msg_ = self.DBC.get_message_by_name(self.config.get('CAN', {}).get('MSG', ''))
            self.logger.info(f"Bound CAN message: {msg_.name}")
        except kvadblib.exceptions.KvdNoMessage as e:
            self.logger.error(f"Error binding CAN message: {e}, return 4")
            return 4
        self.canMsg = msg_.bind()
        try:
            self.can = canlib.openChannel(self.config.get('CAN', {}).get('CHANNEL', 0), canlib.canOPEN_ACCEPT_VIRTUAL)
            self.can.setBusOutputControl(canlib.canDRIVER_NORMAL)
            busrate = get_can_busrate(self.config.get('CAN', {}).get('BUSRATE', 500000))
            self.can.setBusParams(busrate)
            self.can.busOn()
            self.logger.info(f"Opened CAN channel: {self.can.getChannelData_Name()}")
            self.logger.info(f"CAN channel bus rate set to: {busrate}")
        except canlib.exceptions.CanNotFound as e:
            self.logger.error(f"Error opening CAN channel: {e}, return 5")
            self.can = None
            self.canMsg = None
            return 5
        except  Exception as e:
            self.logger.error(f"Error opening CAN channel: {e}, return 6")
            self.can = None
            self.canMsg = None
            return 6
        return 0

    def close_can(self):
        if self.can:
            self.can.busOff()
            self.can.close()
            self.can = None
            self.canMsg = None
            self.logger.info("CAN channel closed.")
        else:
            self.logger.warning("CAN channel was not open.")

    def send_can_message(self, data):
        if not self.can:
            self.logger.debug("CAN channel is not initialized., return 1")
            return 1
        try:
            self.canMsg.offset.phys = float(data)
            self.can.write(self.canMsg._frame)
            self.logger.debug(f"Sent CAN message: {data}, return 0")
            return 0
        except Exception as e:
            self.logger.error(f"Error sending CAN message: {e}, return 2")
            return 2


    def set_source(self):
        self.source = self.config.get('SOURCE', '')
        if not self.source:
            self.logger.error("Source is not set in the configuration. return 1")
            return 1
        if isinstance(self.source, str):
            self.source_type = "video"
            self.logger.info(f"Source type set to: {self.source_type}")
            self.logger.info(f"Source set to: {self.source}")
        elif isinstance(self.source, int):
            self.source_type = "camera"
            self.logger.info(f"Source type set to: {self.source_type}")
            self.logger.info(f"Source set to: {self.source}")
        else:
            self.logger.error("Invalid source type. Source must be a string or an integer. return 2")
            return 2

        # check if source is valid
        temp_cap = cv2.VideoCapture(self.source)
        if not temp_cap.isOpened():
            self.logger.error("Failed to open source. return 3")
            return 3
        temp_cap.release()
        self.logger.info("Source is valid.")
        return 0

    def open_source(self):
        if self.source is None:
            self.logger.error("Source is not set. return 1")
            return 1
        if self.source_type == "video":
            self.cap = cv2.VideoCapture(self.source)
        elif self.source_type == "camera":
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.logger.error("Invalid source type. return 2")
            return 2
        if not self.cap.isOpened():
            self.logger.error("Failed to open source. return 3")
            return 3
        self.logger.info("Source opened successfully.")
        return 0

    def close_source(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("Source closed.")
        else:
            self.logger.warning("Source was not open.")

    def init_model(self):
        if self.config is None:
            return 1
        model_path = self.config.get('MODEL_PATH', '')
        if not model_path:
            self.logger.error("Model path is not set in the configuration. return 2")
            return 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        try:
            self.model = YOLO(model_path, verbose=False).to(device)
            self.logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}, return 3")
            self.model = None
            return 3
        return 0

    def get_offset(self, boundary_points, y_threshold):
        valid_points = boundary_points[boundary_points[:, 0] > y_threshold]
        if len(valid_points) == 0:
            self.logger.warning("No valid points found above the y_threshold.")
            return 1, None
        left_point = [valid_points[0, 0], valid_points[0, 1]]
        right_point = [valid_points[0, 0], valid_points[0, 2]]
        center_point = [(left_point[0] + right_point[0]) / 2, (left_point[1] + right_point[1]) / 2]
        lane_width = calculate_distance(left_point, right_point)
        if lane_width == 0:
            self.logger.error("Lane width is zero, cannot calculate offset. return 2")
            return 2, None

        intersect_left = find_intersection(left_point, right_point, self.config['REF']['left'])
        intersect_right = find_intersection(left_point, right_point, self.config['REF']['right'])
        intersect_center = find_intersection(left_point, right_point, self.config['REF']['center'])

        left_offset = calculate_distance(center_point, intersect_left)
        right_offset = calculate_distance(center_point, intersect_right)
        center_offset = calculate_distance(center_point, intersect_center)

        base_left = calculate_distance(intersect_left, intersect_center)
        base_right = calculate_distance(intersect_right, intersect_center)

        d_left = left_offset - base_left
        d_right = right_offset - base_right

        offset = np.nanmean([d_left, d_right, center_offset])
        if np.isnan(offset):
            self.logger.error("Offset calculation resulted in NaN. return 3")
            return 3, None
        
        offset = offset * self.config.get('LANE_WIDTH', 3.6) / lane_width
        offset = offset / self.config.get('SCALE', 1.0) - self.config.get('BIAS', 0.0)
        return 0, offset
        
    def process_frame(self, frame):
        if self.model is None:
            self.logger.error("Model is not initialized. return 1")
            return 1, None
        if frame is None:
            self.logger.error("Frame is None. return 2")
            return 2, None

        ref = self.config['REF'].get('image_size', [1280, 720])
        W, H = ref[0], ref[1]
        
        with Timer("Frame Processing"):
            frame = cv2.resize(frame, (640, 360))
            results = self.model(frame, verbose=False)

        max_contour_length = 0
        max_contour = None
        segmentation_found = False

        with Timer("Contour Detection"):
            for result in results:
                if result.masks is not None:
                    for mask in result.masks.data:
                        mask_cpu = mask.cpu().numpy()
                        mask_cpu = cv2.resize(mask_cpu, (W, H), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours((mask_cpu * 255).astype('uint8'),
                                                        cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > max_contour_length:
                            max_contour_length = len(contour)
                            max_contour = contour
            if max_contour is not None: 
                    segmentation_found = True
        
        if not segmentation_found:
            self.logger.warning("No segmentation found in the frame.")
            return 5, None
        
        with Timer("Contour Processing"):
            red_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(red_mask, [max_contour], -1, (255), thickness=5)
            ys, xs = np.where(red_mask == 255)
            pixel_indices = np.array(list(zip(xs, ys)))
            final_contour = []
            for v in np.unique(pixel_indices[:, 1]):
                u_vals = pixel_indices[pixel_indices[:, 1] == v, 0]
                if len(final_contour) > 0 and np.max(u_vals) - np.min(u_vals) < final_contour[-1][3]:
                    break
                final_contour.append([v, np.min(u_vals), np.max(u_vals), np.max(u_vals) - np.min(u_vals)])
            final_contour = np.array(final_contour)

        with Timer("Offset Calculation"):
            measured = []
            for y_threshold in np.linspace(np.min(final_contour[:, 0]), np.max(final_contour[:, 0]), self.config.get('NUM_OF_LINES', 2)+2)[1:-1]:
                status, offset = self.get_offset(final_contour, y_threshold)
                if status != 0:
                    self.logger.warning(f"Error calculating offset at {y_threshold}: {status}")
                    continue
                measured.append(offset)
        if len(measured) == 0:
            self.logger.error("No valid offsets calculated. return 4")
            return 4, None
        return 0, np.nanmean(measured)
    
    def publish(self, offset):
        status = self.send_udp_message(offset)
        if status in [2, 3]:
            self.logger.error(f"Error sending UDP message: {status}")
            return status
        status = self.send_can_message(offset)
        if status in [2, 3]:
            self.logger.error(f"Error sending CAN message: {status}")
            return status
        self.logger.info(f"Published offset: {offset}")
        return 0
        
    def init(self, config=None):
        self.load_config(config=config)
        self.setup_logger()
        if self.set_source()!= 0:
            self.shutdown(exit_code=1)
        if self.open_socket() > 1:
            self.shutdown(exit_code=2)
        if self.open_can() > 1:
            self.shutdown(exit_code=3)
        if self.open_source()!= 0:
            self.shutdown(exit_code=4)
        if self.init_model()!= 0:
            self.shutdown(exit_code=5)

    def run(self, stop_event=None):
        while True:
            if stop_event and stop_event.is_set():
                self.logger.warning("Stopping early by request.")
                break
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read frame from source. Exiting.")
                break
            with Timer("full processing"):
                status, offset = self.process_frame(frame)
            if status != 0:
                self.logger.warning(f"Error processing frame: {status}")
                continue
            self.publish(offset)

        self.shutdown()
        return 0
    
    def shutdown(self, exit_code=0):
        self.close_source()
        self.close_can()
        self.close_socket()
        if self.logger:
            self.logger.info("Server shutdown complete.")
        else:
            print("Server shutdown complete.")
        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
        if hasattr(self, 'console_handler'):
            self.logger.removeHandler(self.console_handler)
            self.console_handler.close()
            self.console_handler = None
        sys.exit(exit_code)

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

app = Flask(__name__)
camera_capture = {
    "cap": None,
    "init": False
}

log_queue = queue.Queue()
log_stream_handler = QueueLogHandler(log_queue)
log_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'))
log_stream_handler.setLevel(logging.WARNING)
logger = logging.getLogger("offsetToCenterLine")
logger.addHandler(log_stream_handler)

def init_camera_capture(config):
    cam_source = config.get("SOURCE", 0)
    if camera_capture["cap"] is not None:
        camera_capture["cap"].release()
    camera_capture["cap"] = cv2.VideoCapture(cam_source)
    camera_capture["init"] = camera_capture["cap"].isOpened()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_model():
    stop_event.clear()

    config = request.get_json()
    if not config:
        return jsonify({"error": "No config received"}), 400
    
    def run_instance():
        server = offsetToCenterLine()
        server.init(config=config)
        server.run(stop_event=stop_event)

    threading.Thread(target=run_instance).start()
    return jsonify({"status": "started"})

@app.route('/logs')
def stream_logs():
    def event_stream():
        while True:
            msg = log_queue.get()
            yield f"data: {msg}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/stop', methods=['POST'])
def stop_model():
    stop_event.set()
    return jsonify({"status": "stopping"})

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify(default_config)

@app.route('/save', methods=['POST'])
def save_config():
    config = request.get_json()
    if config is None:
        return jsonify({"error": "Invalid JSON"}), 400
    init_camera_capture(config)
    try:
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

@app.route('/set_log_level', methods=['POST'])
def set_log_level():
    data = request.get_json()
    new_level_str = data.get('level')
    if new_level_str in LOG_LEVELS:
        new_level_int = LOG_LEVELS[new_level_str]
        log_stream_handler.setLevel(new_level_int)
        return jsonify({"success": True, "message": f"Log level set to {new_level_str}"}), 200
    else:
        return jsonify({"success": False, "message": "Invalid log level"}), 400

LOG_DIR = './log'

@app.route('/logs.html')
def logs_page():
    return render_template('logs.html')

@app.route('/log_files')
def list_log_files():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
    return jsonify(files)

@app.route('/log_files/<path:filename>')
def serve_log_file(filename):
    return send_from_directory('log', filename)

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/capture_frame')
def capture_frame():
    cap = camera_capture["cap"]
    if cap is None or not cap.isOpened():
        return "Camera not initialized", 400

    success, frame = cap.read()
    if not success:
        return "Failed to capture frame", 500

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/can_channels')
def get_can_channels():
    try:
        data = list_can_channels()
        return jsonify({"channels": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/run_cmd', methods=['POST'])
def run_cmd():
    data = request.get_json()
    cmd = data.get('cmd', '')
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=5)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.decode('utf-8')}"
    except Exception as e:
        return f"Exception: {str(e)}"
    

@app.route('/system/<action>', methods=['POST'])
def system_control(action):
    if action not in ['shutdown', 'reboot']:
        return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

    try:
        cmd = ['sudo', '/usr/sbin/shutdown', 'now'] if action == 'shutdown' else ['sudo', '/usr/sbin/reboot']
        subprocess.run(cmd)
        return jsonify({'status': 'success', 'message': f'{action.capitalize()} initiated.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



        

