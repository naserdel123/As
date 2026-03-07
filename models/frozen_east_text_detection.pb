#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EAST Text Detection using OpenCV DNN
=====================================

كود كامل لكشف النصوص باستخدام نموذج EAST (Efficient and Accurate Scene Text Detector)
Complete code for text detection using EAST model with OpenCV DNN

المؤلف: AI Assistant
التاريخ: 2024
"""

import cv2
import numpy as np
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ============================================================================
# Configuration / الإعدادات
# ============================================================================

# رابط تحميل النموذج الموثوق
MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
MODEL_DIR = "models"
MODEL_FILENAME = "frozen_east_text_detection.pb"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# أسماء طبقات الإخراج في النموذج
LAYER_NAMES = [
    "feature_fusion/Conv_7/Sigmoid",   # scores: احتمالية وجود نص
    "feature_fusion/concat_3"          # geometry: إحداثيات المربعات
]

# Default parameters / المعاملات الافتراضية
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 320
DEFAULT_CONFIDENCE = 0.5
DEFAULT_NMS_THRESHOLD = 0.4


def download_model(url: str, destination: str) -> bool:
    """
    تحميل النموذج من الإنترنت إذا لم يكن موجوداً محلياً
    Download model from internet if not exists locally
    
    Args:
        url: رابط التحميل
        destination: مسار الحفظ المحلي
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # إنشاء المجلد إذا لم يكن موجوداً
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        print(f"[INFO] جاري تحميل النموذج من: {url}")
        print(f"[INFO] سيتم الحفظ في: {destination}")
        
        # إظهار التقدم أثناء التحميل
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r[INFO] تقدم التحميل: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        # تحميل الملف
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print("\n[INFO] تم اكتمال التحميل بنجاح!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] فشل تحميل النموذج: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # حذف الملف التالف
        return False


def check_and_download_model(model_path: str = MODEL_PATH) -> str:
    """
    التحقق من وجود النموذج وتحميله إذا لزم الأمر
    Check if model exists and download if necessary
    
    Args:
        model_path: مسار النموذج المحلي
    
    Returns:
        str: مسار النموذج أو None في حالة الفشل
    """
    if os.path.exists(model_path):
        print(f"[INFO] النموذج موجود محلياً: {model_path}")
        return model_path
    
    print(f"[INFO] النموذج غير موجود. سيتم التحميل تلقائياً...")
    
    if download_model(MODEL_URL, model_path):
        # التحقق من حجم الملف (يجب أن يكون أكبر من 0)
        if os.path.getsize(model_path) > 0:
            return model_path
        else:
            print("[ERROR] الملف المحمل فارغ!")
            return None
    return None


def decode_predictions(scores: np.ndarray, geometry: np.ndarray, 
                       confidence_threshold: float) -> tuple:
    """
    فك تشفير مخرجات النموذج لاستخراج إحداثيات المربعات
    Decode model outputs to extract bounding box coordinates
    
    Args:
        scores: مصفوفة احتماليات النص [1, 1, rows, cols]
        geometry: مصفوفة الإحداثيات [1, 5, rows, cols]
        confidence_threshold: عتبة الثقة للتصفية
    
    Returns:
        tuple: (rects, confidences, angles) - المربعات والثقة والزوايا
    """
    # الحصول على أبعاد المصفوفات
    (num_rows, num_cols) = scores.shape[2:4]
    
    rects = []      # قائمة المربعات
    confidences = [] # قائمة الثقة
    angles_list = [] # قائمة الزوايا (للدعم المستقبلي للمربعات المائلة)
    
    # المرور على كل البكسلات في feature map
    for y in range(0, num_rows):
        # استخراج البيانات لكل صف
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]  # top distance
        x_data1 = geometry[0, 1, y]  # right distance  
        x_data2 = geometry[0, 2, y]  # bottom distance
        x_data3 = geometry[0, 3, y]  # left distance
        angles_data = geometry[0, 4, y]  # rotation angle
        
        for x in range(0, num_cols):
            # تجاهل البكسلات ذات الثقة المنخفضة
            if scores_data[x] < confidence_threshold:
                continue
            
            # حساب الإزاحة (الشبكة أصغر 4 مرات من الصورة الأصلية)
            offset_x = x * 4.0
            offset_y = y * 4.0
            
            # استخراج زاوية الدوران
            angle = angles_data[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # حساب الارتفاع والعرض
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            
            # حساب إحداثيات النهاية
            end_x = int(offset_x + (cos_a * x_data1[x]) + (sin_a * x_data2[x]))
            end_y = int(offset_y - (sin_a * x_data1[x]) + (cos_a * x_data2[x]))
            
            # حساب إحداثيات البداية
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            
            # إضافة إلى القوائم
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))
            angles_list.append(angle)
    
    return (rects, confidences, angles_list)


def non_max_suppression_rotated(rects: list, confidences: list, 
                                threshold: float) -> list:
    """
    تطبيق Non-Maximum Suppression لإزالة المربعات المتداخلة
    Apply NMS to remove overlapping boxes
    
    Note: This uses standard axis-aligned NMS. For rotated boxes,
    OpenCV's NMSBoxes would be better but requires cv2.dnn.NMSBoxes
    with rotated rect support.
    
    Args:
        rects: قائمة المربعات (x1, y1, x2, y2)
        confidences: قائمة الثقة
        threshold: عتبة التداخل
    
    Returns:
        list: المؤشرات المختارة بعد NMS
    """
    if len(rects) == 0:
        return []
    
    # تحويل إلى numpy arrays
    boxes = np.array(rects)
    scores = np.array(confidences)
    
    # استخدام NMSBoxes من OpenCV DNN
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.3,  # threshold for considering
        nms_threshold=threshold
    )
    
    # في إصدارات OpenCV المختلفة، قد يكون الإرجاع مختلفاً
    if len(indices) > 0:
        if isinstance(indices, (tuple, list)) and len(indices) > 0:
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = list(indices)
        elif isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
    
    return indices


def detect_text_east(image: np.ndarray, net: cv2.dnn.Net,
                     width: int = 320, height: int = 320,
                     confidence_threshold: float = 0.5,
                     nms_threshold: float = 0.4) -> tuple:
    """
    كشف النصوص في الصورة باستخدام EAST
    Detect text in image using EAST detector
    
    Args:
        image: صورة الإدخال (BGR)
        net: شبكة OpenCV DNN المحملة
        width: عرض إعادة التحجيم
        height: ارتفاع إعادة التحجيم
        confidence_threshold: عتبة الثقة
        nms_threshold: عتبة NMS
    
    Returns:
        tuple: (image_with_boxes, boxes, scores, processing_time)
    """
    # حفظ الأبعاد الأصلية
    orig_h, orig_w = image.shape[:2]
    
    # إعادة تحجيم الصورة لأبعاد مناسبة للنموذج (يجب أن تكون مضاعفات 32)
    # Resize image to model input size (must be multiple of 32)
    resized = cv2.resize(image, (width, height))
    
    # إنشاء blob للشبكة العصبية
    # Create blob: scale=1.0, mean subtraction for ImageNet stats
    blob = cv2.dnn.blobFromImage(
        resized, 
        scalefactor=1.0,
        size=(width, height),
        mean=(123.68, 116.78, 103.94),  # ImageNet mean (BGR)
        swapRB=True,  # Convert RGB to BGR
        crop=False
    )
    
    # تمرير البيانات للشبكة
    net.setInput(blob)
    
    # قياس وقت المعالجة
    start_time = cv2.getTickCount()
    outputs = net.forward(LAYER_NAMES)
    end_time = cv2.getTickCount()
    
    processing_time = (end_time - start_time) / cv2.getTickFrequency()
    
    scores = outputs[0]
    geometry = outputs[1]
    
    # فك تشفير المخرجات
    rects, confidences, angles = decode_predictions(scores, geometry, confidence_threshold)
    
    # تطبيق NMS
    indices = non_max_suppression_rotated(rects, confidences, nms_threshold)
    
    # حساب نسب التحجيم للعودة للأبعاد الأصلية
    r_w = orig_w / float(width)
    r_h = orig_h / float(height)
    
    # نسخة للرسم
    output_image = image.copy()
    final_boxes = []
    final_scores = []
    
    # رسم المربعات المختارة
    if len(indices) > 0:
        for i in indices:
            if i >= len(rects):
                continue
                
            start_x, start_y, end_x, end_y = rects[i]
            
            # إعادة تحجيم الإحداثيات للأبعاد الأصلية
            start_x = int(start_x * r_w)
            start_y = int(start_y * r_h)
            end_x = int(end_x * r_w)
            end_y = int(end_y * r_h)
            
            # التأكد من أن الإحداثيات ضمن حدود الصورة
            start_x = max(0, min(start_x, orig_w - 1))
            start_y = max(0, min(start_y, orig_h - 1))
            end_x = max(0, min(end_x, orig_w - 1))
            end_y = max(0, min(end_y, orig_h - 1))
            
            # تخطي المربعات الصغيرة جداً
            if (end_x - start_x) < 5 or (end_y - start_y) < 5:
                continue
            
            # رسم المستطيل الأخضر
            cv2.rectangle(output_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # إضافة نص الثقة
            label = f"{confidences[i]:.2f}"
            label_y = start_y - 10 if start_y - 10 > 10 else start_y + 20
            cv2.putText(output_image, label, (start_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            final_boxes.append((start_x, start_y, end_x, end_y))
            final_scores.append(confidences[i])
    
    # إضافة معلومات الأداء
    info_text = f"Time: {processing_time*1000:.1f}ms | Boxes: {len(final_boxes)}"
    cv2.putText(output_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output_image, final_boxes, final_scores, processing_time


def process_image(image_path: str, net: cv2.dnn.Net, args) -> None:
    """
    معالجة صورة واحدة
    Process a single image
    
    Args:
        image_path: مسار الصورة
        net: شبكة OpenCV DNN
        args: معاملات سطر الأوامر
    """
    # قراءة الصورة
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] لا يمكن قراءة الصورة: {image_path}")
        return
    
    print(f"[INFO] معالجة الصورة: {image_path} ({image.shape[1]}x{image.shape[0]})")
    
    # كشف النصوص
    result, boxes, scores, proc_time = detect_text_east(
        image, net,
        width=args.width,
        height=args.height,
        confidence_threshold=args.conf,
        nms_threshold=args.nms
    )
    
    print(f"[INFO] تم اكتشاف {len(boxes)} منطقة نص")
    print(f"[INFO] وقت المعالجة: {proc_time*1000:.2f} ms")
    
    # عرض النتيجة
    window_name = "EAST Text Detection"
    cv2.imshow(window_name, result)
    
    # حفظ النتيجة اختيارياً
    output_path = f"{Path(image_path).stem}_output.jpg"
    cv2.imwrite(output_path, result)
    print(f"[INFO] تم حفظ النتيجة في: {output_path}")
    
    print("[INFO] اضغط أي مفتاح للخروج...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path: str, net: cv2.dnn.Net, args) -> None:
    """
    معالجة فيديو أو كاميرا ويب
    Process video file or webcam
    
    Args:
        video_path: مسار الفيديو أو None للكاميرا
        net: شبكة OpenCV DNN
        args: معاملات سطر الأوامر
    """
    # فتح مصدر الفيديو
    if video_path is None or video_path.lower() == "0":
        print("[INFO] فتح كاميرا الويب...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"[INFO] فتح ملف الفيديو: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] لا يمكن فتح مصدر الفيديو")
        return
    
    # الحصول على خصائص الفيديو
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] دقة الفيديو: {frame_width}x{frame_height}, FPS: {fps}")
    
    # إعداد كاتب الفيديو للحفظ
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    
    frame_count = 0
    total_time = 0
    
    window_name = "EAST Text Detection - Press 'q' to quit"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] انتهاء الفيديو")
            break
        
        # كشف النصوص في الإطار
        result, boxes, scores, proc_time = detect_text_east(
            frame, net,
            width=args.width,
            height=args.height,
            confidence_threshold=args.conf,
            nms_threshold=args.nms
        )
        
        total_time += proc_time
        frame_count += 1
        
        # إعداد الكاتب في الإطار الأول
        if writer is None and video_path is not None:
            h, w = result.shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            print(f"[INFO] جاري الحفظ في: {output_path}")
        
        # كتابة الإطار
        if writer is not None:
            writer.write(result)
        
        # عرض النتيجة
        cv2.imshow(window_name, result)
        
        # التحقق من مفتاح الخروج
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("[INFO] إيقاف المستخدم")
            break
    
    # إطلاق الموارد
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] تم حفظ الفيديو في: {output_path}")
    
    cv2.destroyAllWindows()
    
    # إحصائيات الأداء
    if frame_count > 0:
        avg_time = total_time / frame_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"[INFO] إطارات معالجة: {frame_count}")
        print(f"[INFO] متوسط وقت المعالجة: {avg_time*1000:.2f} ms")
        print(f"[INFO] متوسط FPS: {avg_fps:.2f}")


def main():
    """
    الدالة الرئيسية
    Main function
    """
    # إعداد معالج المعاملات
    parser = argparse.ArgumentParser(
        description="EAST Text Detection using OpenCV DNN - كشف النصوص باستخدام OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / أمثلة:
  # معالجة صورة:
  python text_detection.py --input image.jpg
  
  # معالجة فيديو:
  python text_detection.py --input video.mp4
  
  # استخدام الكاميرا:
  python text_detection.py --input 0
  
  # تغيير أبعاد الإدخال والعتبات:
  python text_detection.py --input image.jpg --width 640 --height 640 --conf 0.7 --nms 0.3
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="مسار الصورة أو الفيديو (استخدم '0' للكاميرا) / Path to image/video (use '0' for webcam)"
    )
    
    parser.add_argument(
        "-e", "--east",
        default=MODEL_PATH,
        help=f"مسار نموذج EAST (افتراضي: {MODEL_PATH}) / Path to EAST model"
    )
    
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"عرض إعادة التحجيم (افتراضي: {DEFAULT_WIDTH}, يجب أن يكون مضاعف 32) / Resize width (multiple of 32)"
    )
    
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"ارتفاع إعادة التحجيم (افتراضي: {DEFAULT_HEIGHT}, يجب أن يكون مضاعف 32) / Resize height (multiple of 32)"
    )
    
    parser.add_argument(
        "-c", "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"عتبة الثقة (افتراضي: {DEFAULT_CONFIDENCE}) / Confidence threshold"
    )
    
    parser.add_argument(
        "-n", "--nms",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
        help=f"عتبة NMS (افتراضي: {DEFAULT_NMS_THRESHOLD}) / NMS threshold"
    )
    
    args = parser.parse_args()
    
    # التحقق من أن الأبعاد مضاعفات 32
    if args.width % 32 != 0 or args.height % 32 != 0:
        print("[WARNING] الأبعاد يجب أن تكون مضاعفات 32. جاري التقريب...")
        args.width = (args.width // 32) * 32
        args.height = (args.height // 32) * 32
        print(f"[INFO] الأبعاد المعدلة: {args.width}x{args.height}")
    
    # التحقق من وجود النموذج وتحميله إذا لزم الأمر
    model_path = args.east
    if not os.path.exists(model_path):
        print(f"[INFO] النموذج غير موجود في: {model_path}")
        model_path = check_and_download_model(MODEL_PATH)
        if model_path is None:
            print("[ERROR] فشل في الحصول على النموذج. الخروج...")
            sys.exit(1)
    
    # تحميل النموذج باستخدام OpenCV DNN
    print(f"[INFO] جاري تحميل النموذج: {model_path}")
    try:
        net = cv2.dnn.readNet(model_path)
        print("[INFO] تم تحميل النموذج بنجاح")
    except Exception as e:
        print(f"[ERROR] فشل تحميل النموذج: {e}")
        sys.exit(1)
    
    # تحديد نوع الإدخال
    input_path = args.input
    
    # التحقق إذا كان الإدخال رقم (كاميرا)
    if input_path == "0":
        process_video(None, net, args)
    elif os.path.isfile(input_path):
        # التحقق من امتداد الملف لتحديد نوعه
        ext = Path(input_path).suffix.lower()
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        if ext in image_extensions:
            process_image(input_path, net, args)
        elif ext in video_extensions:
            process_video(input_path, net, args)
        else:
            # محاولة فتح كصورة أولاً
            test_img = cv2.imread(input_path)
            if test_img is not None:
                process_image(input_path, net, args)
            else:
                # محاولة فتح كفيديو
                process_video(input_path, net, args)
    else:
        print(f"[ERROR] الملف غير موجود: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
