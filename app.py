from flask import Flask, render_template, request #A web frameworkproviding tools for handling HTTP requests, routing, templates, and more.
from werkzeug.utils import secure_filename   # A utility library for WSGI (Web Server Gateway Interface) used by Flask, providing various components for handling HTTP, such as file uploads and secure filename generation.
import os
import imageio#A library for reading and writing images and videos in various formats, facilitating image and video I/O operations in Python.
import face_recognition#A library for face detection and recognition, built on top of dlib and providing a high-level interface for facial feature analysis.
from PIL import Image, ImageDraw, ImageFont  #A Python Imaging Library (PIL) fork known as Pillow, providing image processing capabilities, including opening, manipulating, and saving various image formats.
import numpy as np
import time

app = Flask(__name__)

IMAGE_UPLOAD_FOLDER = 'uploaded images'   #Configuration setting for Flask, specifically setting a secret key used for securing session data.
VIDEO_UPLOAD_FOLDER = 'uploaded videos'
RESULT_FOLDER = 'result'

app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
app.config['VIDEO_UPLOAD_FOLDER'] = VIDEO_UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def annotate_faces(video_path, target_image_path, output_path):
    #This code uses face recognition to compare a target face with faces in each video frame, 
    #marking matches with rectangles and labels. The allowed_file function checks if a file's extension is permitted.

    target_image = face_recognition.load_image_file(target_image_path)
    target_encoding = face_recognition.face_encodings(target_image)[0]

    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data()['fps']
    output_writer = imageio.get_writer(output_path, fps=fps)

    for frame_number, frame in enumerate(video_reader):
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        annotated_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated_frame)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            matches = face_recognition.compare_faces([target_encoding], face_encodings[i], tolerance=0.5)

            if matches.count(True) / len(matches) > 0.7:
                draw.rectangle([left, top, right, bottom], outline=(0, 255, 0), width=2)

                label = "Person Found"
                font = ImageFont.load_default()
                draw.text((left + 6, bottom - 6), label, font=font, fill=(255, 255, 255))

        output_writer.append_data(np.array(annotated_frame))

    output_writer.close()

def draw_label(image, location, label, color, label_background_color):
    font_size = 12
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    #This function draws a labeled rectangle on an image at a specified location. 
    #It includes text with the given label, using specified colors and font settings. 
    #The label is enclosed in a background rectangle to enhance visibility. 
    
    text_bbox = draw.textbbox((location[0], location[1] - font_size - 3), label, font=font)

    label_size = (text_bbox[2] - text_bbox[0] + 12, text_bbox[3] - text_bbox[1] + 6)

    rectangle_location = (location[0] - 6, location[1] - label_size[1] - 3)

    draw.rectangle([rectangle_location, (rectangle_location[0] + label_size[0], rectangle_location[1] + label_size[1])],
                   fill=label_background_color)

    draw.text((location[0], location[1] - label_size[1] - 3), label, font=font, fill=color)


def draw_rectangles_and_labels(frame, face_locations):
    draw = ImageDraw.Draw(frame)  # Create ImageDraw.Draw object outside the loop
    for (top, right, bottom, left) in face_locations:
        width = right - left
        height = bottom - top

        
        size = min(width, height)
        #This function draws rectangles around detected faces on a given image frame. 
        #It adjusts the rectangle dimensions to create a square around each face, outlines them 
        # in green, and adds a label ("Person Found") below each face.

        
        new_left = left + (width - size) // 2
        new_top = top + (height - size) // 2
        new_right = new_left + size
        new_bottom = new_top + size

        draw.rectangle([new_left, new_top, new_right, new_bottom], outline=(0, 255, 0), width=2)

        font_location = (new_left + 6, new_bottom - 6)
        label = "Person Found"
        draw_label(frame, font_location, label, color=(255, 255, 255), label_background_color=(0, 255, 0))
    del draw

def annotate_and_recognize_faces(frame, target_encoding, confidence_threshold=0.7):
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for i, face_encoding in enumerate(face_encodings):
        if not face_encoding:
            continue
        matches = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.5)

        print(matches and sum(matches) / len(matches) > confidence_threshold)
        if matches and sum(matches) / len(matches) > confidence_threshold:
            draw_rectangles_and_labels(frame, [face_locations[i]])
            #This function looks for faces in a picture, compares them to a known face, 
            # and if the match is strong enough, it marks the face with a rectangle and label. 
            # The modified image is then returned.

    return frame

def process_video(video_path, target_encoding, target_image_path):
    video_reader = imageio.get_reader(video_path)
    total_frames = len(video_reader)

    result_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(video_path))
    result_video_writer = imageio.get_writer(result_path, fps=30)

    person_found = False
    frame_number = 0
     #This function goes through each frame of a video, finds and marks faces, and creates 
    # a new video with labeled faces. It returns whether a person was found in any frame and 
    # the path to the result video.

    try:
        for frame_number in range(total_frames):
            print(frame_number)
            frame = video_reader.get_data(frame_number)
            annotated_frame = annotate_and_recognize_faces(frame, target_encoding)
            print(f"Processing frame {frame_number}, shape: {frame.shape}")
            result_video_writer.append_data(annotated_frame)

            
            if not person_found and np.any(np.array(annotated_frame) != frame):
                person_found = True

    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")

    finally:
        result_video_writer.close()
    return person_found, result_path

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Check if files exist in the request
        if 'target_image' not in request.files or 'video' not in request.files:
            return render_template('index.html', error='No file part')

        target_image_file = request.files['target_image']
        video_file = request.files['video']
        #This function manages user uploads, ensuring they're valid images and videos. 
        #It processes the data, annotates faces in video frames, and generates 
        #a new video with labeled faces. The function returns the result path and 
        #displays error messages for invalid selections.

        # Check if filenames are not empty
        if target_image_file.filename == '' or video_file.filename == '':
            return render_template('index.html', error='No selected file')

        if not allowed_file(target_image_file.filename) or not allowed_file(video_file.filename):
            return render_template('index.html', error='Invalid file format')

        target_image_path = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], secure_filename(target_image_file.filename))
        target_image_file.save(target_image_path)

        video_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        output_path = os.path.join(app.config['RESULT_FOLDER'], 'output.mp4')

        annotate_faces(video_path, target_image_path, output_path)

        return render_template("index.html", result=output_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
