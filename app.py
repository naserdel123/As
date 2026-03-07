from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_text_from_frame(frame, text_detector):
    # كشف النصوص باستخدام EAST model
    conf_thresh = 0.5
    nms_thresh = 0.4
    input_size = (320, 320)
    blob = cv2.dnn.blobFromImage(frame, 1.0, input_size, (123.68, 116.78, 103.94), swapRB=True, crop=False)
    text_detector.setInput(blob)
    (scores, geometry) = text_detector.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])
    
    # استخراج bounding boxes
    boxes, confidences = [], []
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        scores_data = scores[0, 0, y]
        x_data0, x_data1, x_data2, x_data3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        for x in range(cols):
            if scores_data[x] < conf_thresh:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles_data[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])
    
    # تطبيق NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    
    # إنشاء mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(indices) > 0:
        for i in indices.flatten():
            (start_x, start_y, end_x, end_y) = boxes[i]
            cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)
    
    # إزالة النصوص بـ inpainting
    inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_frame

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # تحميل النموذج
            model_path = 'models/frozen_east_text_detection.pb'
            text_detector = cv2.dnn.readNet(model_path)
            
            # معالجة الفيديو
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = remove_text_from_frame(frame, text_detector)
                out.write(processed_frame)
            
            cap.release()
            out.release()
            
            return send_file(output_path, as_attachment=True)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
