from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import cv2
import time
import ffmpy
from flask import Flask, render_template,request, jsonify
from werkzeug import secure_filename
app = Flask(__name__, template_folder='templates')
#file = "test2.mp4"

current_time_time = time.strftime("%Y%m%d-%H%M%S")

path = Path('data/train')

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.load('stage-2')
learn.unfreeze()

def convert(inputted_file):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    video_name = str(current_time) + ".avi"
    ff = ffmpy.FFmpeg(inputs={inputted_file : None}, outputs={video_name: ' -c:a mp3 -c:v mpeg4'})
    ff.cmd
    ff.run()
    return video_name

def getFrame(current_frame, move_frame, col_frame, cap):
    cap.set(cv2.CAP_PROP_POS_MSEC,current_frame*1000)
    hasFrames,image = cap.read()
    if hasFrames:
        pred_class = learn.predict(Image(pil2tensor(image, np.float32).div_(255)))[0]
        pred_class_string = str(pred_class)
        if pred_class_string == "moving" and move_frame == 0:
            move_frame = current_frame
            pred_class_string = "m"
        if pred_class_string == "collided" and col_frame == 0:
            col_frame = current_frame
            pred_class_string = "c"
        print(pred_class_string)
        print(current_frame)
    return hasFrames, move_frame, col_frame

def checkTime(video_name):
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frameRate = 0.01

    move_frame = 0
    col_frame = 0

    current_frame = 0
    count = 1
    success = getFrame(current_frame, move_frame, col_frame, cap)

    while success:
        count = count + 1
        current_frame = current_frame + frameRate
        success, move_frame, col_frame = getFrame(current_frame, move_frame, col_frame, cap)

    cap.release()
    cv2.destroyAllWindows()

    return move_frame, col_frame, fps

def tellTime(move_frame, col_frame, fps):
    col_frame -= 0.01
    seconds = ((col_frame - move_frame)*fps)
    col_mins = seconds // 60
    col_secs = seconds % 60
    result_message = ("{:02}:{:02}".format(col_mins, col_secs))
    print(result_message)
    return result_message

#vname = convert(file)
#move_frame, col_frame = checkTime(vname)
#time = tellTime(move_frame, col_frame)


learn.export()

defaults.device = torch.device('cpu')

@app.route('/', methods=['GET', 'POST'])
def checkFile():
   if request.method == 'GET':
        return render_template('index.html', value='hi')
   if request.method == 'POST':
       file_s = request.files['file']
       if not file_s.filename.endswith:
           return jsonify(time="You didn't submit a video")
       global current_time_time
       name_file = request.remote_addr+current_time_time+secure_filename(file_s.filename)
       file_s.save(name_file)
       vname = convert(name_file)
       move_frame, col_frame, fps = checkTime(vname)
       time = tellTime(move_frame, col_frame, fps)
       return render_template('result.html', time=time)
#       return jsonify(time=time)

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='192.168.2.242', port=port)
