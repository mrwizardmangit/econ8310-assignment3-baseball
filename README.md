![](image.png)

Due to Github's file size limits, the trained model weights are hosted here: https://drive.google.com/file/d/1OukSS3rECSYYu5RlDGgy5JOOdvMm6edg/view?usp=drive_link

Please download this file and place it in the root directory before running evaluate.py.

data_loader.py contains the logic for separating .mov files into individual frames, parsing xml files (from annotating video), and applying the logic from the xml files onto the un-annotated frames taken from the raw video.

train.py uses the class defined in data_loader.py to convert the raw video / xml into readable data and uses that data to train a fasterrcnn resnet 50 model. The output of this training is baseball_weights.pth.

evaluate.py uses the weights from baseball_weights.pth to apply the model to a video. It attempts to draw bounding boxes around everything it perceives as a baseball and displays its confidence in that identification between 0 and 1 (1 being 100% confident). When run, this file outputs an annotated mp4 of the input video.

To test: 1. Put video under test as .mov file in video_data folder 2. Open evaluate.py and update the input_video path to video under test 3. Run evaluate.py

It should process the video frame by frame and generate an annotated_pitch.mp4 file in your root directory.