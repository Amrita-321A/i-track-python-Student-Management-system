from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import sqlite3, os, csv, json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import cv2
import mediapipe as mp
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ---------------- DATABASE ----------------
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- MEDIAPIPE SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- PARAMETERS ----------------
EAR_THRESHOLD = 0.25
DROWSY_FRAMES = 15
ATTENTION_FRAMES = 15
GAZE_LEFT_THRESHOLD = 0.35
GAZE_RIGHT_THRESHOLD = 0.65

# ---------------- GLOBAL STATE ----------------
drowsy_counter = 0
inattention_counter = 0
current_status = "Attentive"
blink_counter = 0
blink_frame_flag = False

# ---------------- HELPER FUNCTIONS ----------------
def euclidean_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def eye_aspect_ratio(landmarks, img_w, img_h, eye_indices):
    left = (landmarks[eye_indices['left']].x*img_w, landmarks[eye_indices['left']].y*img_h)
    right = (landmarks[eye_indices['right']].x*img_w, landmarks[eye_indices['right']].y*img_h)
    top = (landmarks[eye_indices['top']].x*img_w, landmarks[eye_indices['top']].y*img_h)
    bottom = (landmarks[eye_indices['bottom']].x*img_w, landmarks[eye_indices['bottom']].y*img_h)
    hor = euclidean_distance(left, right)
    ver = euclidean_distance(top, bottom)
    return ver/hor if hor != 0 else 0

def gaze_ratio(landmarks, img_w, img_h, eye_indices, iris_index):
    left_corner = (landmarks[eye_indices['left']].x*img_w, landmarks[eye_indices['left']].y*img_h)
    right_corner = (landmarks[eye_indices['right']].x*img_w, landmarks[eye_indices['right']].y*img_h)
    top_corner = (landmarks[eye_indices['top']].x*img_w, landmarks[eye_indices['top']].y*img_h)
    bottom_corner = (landmarks[eye_indices['bottom']].x*img_w, landmarks[eye_indices['bottom']].y*img_h)
    iris = (landmarks[iris_index].x*img_w, landmarks[iris_index].y*img_h)
    denom_x = (right_corner[0] - left_corner[0]) if (right_corner[0] - left_corner[0]) != 0 else 1
    denom_y = (bottom_corner[1] - top_corner[1]) if (bottom_corner[1] - top_corner[1]) != 0 else 1
    ratio_x = (iris[0]-left_corner[0]) / denom_x
    ratio_y = (iris[1]-top_corner[1]) / denom_y
    return ratio_x, ratio_y

def head_direction(landmarks, img_w):
    nose_x = landmarks[1].x * img_w
    left_x = landmarks[234].x * img_w
    right_x = landmarks[454].x * img_w
    center = (left_x + right_x) / 2
    th = 20
    if nose_x < center - th:
        return "Left"
    elif nose_x > center + th:
        return "Right"
    else:
        return "Center"

# ---------------- VIDEO STREAM ----------------
def video_generator():
    global drowsy_counter, inattention_counter, current_status
    global blink_counter, blink_frame_flag

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        current_status = "No Face"
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            current_status = "No Face"
            break

        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "Attentive"

        if not results.multi_face_landmarks:
            drowsy_counter = 0
            inattention_counter = 0
            blink_frame_flag = False
            status = "No Face"
        else:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye_indices = {'left': 33, 'right': 133, 'top': 159, 'bottom': 145}
            right_eye_indices = {'left': 362, 'right': 263, 'top': 386, 'bottom': 374}

            try:
                ear_left = eye_aspect_ratio(landmarks, img_w, img_h, left_eye_indices)
                ear_right = eye_aspect_ratio(landmarks, img_w, img_h, right_eye_indices)
                ear = (ear_left + ear_right) / 2.0
            except Exception:
                ear = 1.0

            if ear < EAR_THRESHOLD:
                if not blink_frame_flag:
                    blink_counter += 1
                    blink_frame_flag = True
                drowsy_counter += 1
                if drowsy_counter >= DROWSY_FRAMES:
                    status = "Drowsy"
            else:
                blink_frame_flag = False
                drowsy_counter = 0

            direction = head_direction(landmarks, img_w)
            try:
                left_ratio_x, _ = gaze_ratio(landmarks, img_w, img_h, left_eye_indices, 468)
                right_ratio_x, _ = gaze_ratio(landmarks, img_w, img_h, right_eye_indices, 473)
                gaze_x = (left_ratio_x + right_ratio_x) / 2.0
            except Exception:
                gaze_x = 0.5

            if direction != "Center" or gaze_x < GAZE_LEFT_THRESHOLD or gaze_x > GAZE_RIGHT_THRESHOLD:
                inattention_counter += 1
                if inattention_counter >= ATTENTION_FRAMES:
                    status = "Not Attentive"
            else:
                inattention_counter = 0

            if status != "Drowsy" and status != "Not Attentive":
                status = "Attentive"

        current_status = status
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_page'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_page'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/admin')
def admin_page():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    conn = get_db_connection()
    tests = conn.execute("SELECT * FROM tests").fetchall()
    results = conn.execute("SELECT * FROM exam_results ORDER BY id DESC").fetchall()
    conn.close()

    return render_template('admin_dashboard.html', tests=tests, submissions=results)

@app.route('/assign_test', methods=['GET', 'POST'])
def assign_test():
    if request.method == 'POST':
        title = request.form['title']
        questions = request.form.getlist('questions[]')
        conn = get_db_connection()
        conn.execute('INSERT INTO tests (title, q1, q2) VALUES (?, ?, ?)',
                     (title, questions[0] if len(questions) > 0 else None,
                      questions[1] if len(questions) > 1 else None))
        conn.commit()
        conn.close()
        return redirect(url_for('admin_page'))
    return render_template('assign_test.html')

@app.route('/delete_test/<int:test_id>', methods=['POST'])
def delete_test(test_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    conn = get_db_connection()
    conn.execute("DELETE FROM tests WHERE id = ?", (test_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_page'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO students(username, password_hash) VALUES(?, ?)",
                         (username, password_hash))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template('student_register.html', error="Username exists")
        finally:
            conn.close()
        return redirect(url_for('student_login'))
    return render_template('student_register.html')

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if session.get('student_logged_in'):
        return redirect(url_for('student_dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        student = conn.execute("SELECT * FROM students WHERE username = ?", (username,)).fetchone()
        conn.close()
        if student and check_password_hash(student['password_hash'], password):
            session['student_logged_in'] = True
            session['student_id'] = student['id']
            session['student_username'] = student['username']
            return redirect(url_for('student_dashboard'))
        else:
            return render_template('student_login.html', error="Invalid credentials")
    return render_template('student_login.html')

@app.route('/student/dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))
    conn = get_db_connection()
    tests = conn.execute("SELECT * FROM tests").fetchall()
    conn.close()
    return render_template('student_dashboard.html', student_name=session['student_username'], tests=tests)

@app.route('/student/logout')

def student_logout():
    session.clear()
    return redirect(url_for('student_login'))

# ---------------- EXAM ----------------
@app.route('/exam/<int:test_id>', methods=['GET', 'POST'])
def exam_page(test_id):
    global blink_counter
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    conn = get_db_connection()
    test = conn.execute("SELECT * FROM tests WHERE id = ?", (test_id,)).fetchone()
    conn.close()

    if not test:
        return "Test not found."

    if request.method == 'POST':
        data = dict(request.form)
        status_history_json = data.get('status_history', '[]')
        statuses = json.loads(status_history_json)
        total = len(statuses)
        active_percent = round(statuses.count("Attentive") / total * 100, 2) if total else 0
        drowsy_percent = round(statuses.count("Drowsy") / total * 100, 2) if total else 0
        inattentive_percent = round(statuses.count("Not Attentive") / total * 100, 2) if total else 0
        noface_percent = round(statuses.count("No Face") / total * 100, 2) if total else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store row data
        row = {
            "timestamp": timestamp,
            "student_name": session.get('student_username', 'Unknown'),
            "test_title": test['title'],
            "q1": data.get("q1", ""),
            "q2": data.get("q2", ""),
            "active_percent": active_percent,
            "drowsy_percent": drowsy_percent,
            "inattentive_percent": inattentive_percent,
            "noface_percent": noface_percent,
            "blink_count": blink_counter,
            
        }

        # --- Save to CSV ---
        file_exists = os.path.isfile('exam_results.csv')
        with open('exam_results.csv', mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # --- Save to SQLite ---
        conn = get_db_connection()
        conn.execute('''INSERT INTO exam_results 
                        (timestamp, student_name, test_title, q1, q2, active_percent, 
                         drowsy_percent, inattentive_percent, noface_percent, blink_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (timestamp, row["student_name"], row["test_title"], row["q1"], row["q2"],
                      active_percent, drowsy_percent, inattentive_percent, noface_percent, blink_counter))
        conn.commit()
        conn.close()

        return render_template('exam_submitted.html',
                               username=session['student_username'],
                               test_title=test['title'],
                               active_percent=active_percent,
                               drowsy_percent=drowsy_percent,
                               inattentive_percent=inattentive_percent,
                               noface_percent=noface_percent,
                               blink_count=blink_counter)

    return render_template('exam.html', test=test)

# ---------------- VIDEO FEED & STATUS ----------------
@app.route('/video_feed')
def video_feed():
    return Response(video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"status": current_status})

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS students(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS tests(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    q1 TEXT,
                    q2 TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS exam_results(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    student_name TEXT,
                    test_title TEXT,
                    q1 TEXT,
                    q2 TEXT,
                    active_percent REAL,
                    drowsy_percent REAL,
                    inattentive_percent REAL,
                    noface_percent REAL,
                    blink_count INTEGER)''')
    conn.commit()
    conn.close()
    app.run(debug=True)
