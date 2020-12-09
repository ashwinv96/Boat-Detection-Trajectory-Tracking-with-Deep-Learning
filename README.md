# Boat Trajectory Tracking with Deep Learning
Tracking the trajectory of a target ship in GPS coordinates from a single video taken from a monocular camera with unknown camera intrinsic and extrinsic parameters. The first stage of the detector uses a custom trained YOLOv4 object detector. The trajectory is estimated through by learning a quadratic approximation to the cost function. 
This project was our team's entry into the <em>AI Tracks at Sea</em> competition held by the US Navy.
## Team
**Advisor**: Dr. Olugbenga Anubi <br />
**Members**: Ashwin Vadivel, Boluwatife Olabiran, Muhammad Saud Ul Hassan, Yu Zheng

## Clone repo and add TensorRT weights and sample video
Use git clone to clone the repo into your local machine. Download the model file and video sample at https://drive.google.com/drive/folders/1dE54rw5-pUVBVzzuK1zqbwRjhWruqXbx?usp=sharing. Next add the video "19.mp4" to the root of the cloned repo. Add the .trt model file into the directory "yolo/".
## Usage
To run the demo please enter the following command: 
```python
python3 main.py \
--video </path_to_video> \
-n number_of_points_to_generate \
-lat source_latitude \
-lon source_longitude
```
The algorithm will run through every frame of the video and ouput detections to the terminal. Once completed, the detected trajectory will be saved in the directory into the file: <strong><em>interpolated_data.csv</em></strong>. The number of points in the trajectory will be equal to the argument <em>-n number_of_points_to_generate</em>.
## Visualize Output Trajectory 
In order to view the output trajectory, we used the free website: <em>https://www.gpsvisualizer.com/</em>.
Ground Truth Trajectory             | Model Predicted Trajectory
:-------------------------:|:-------------------------:
<img src="https://github.com/ashwinv96/Boat-Trajectory-Tracking-without-Camera-Params/blob/master/Figures/gt.png?raw=true"/>  | <img src="https://github.com/ashwinv96/Boat-Trajectory-Tracking-without-Camera-Params/blob/master/Figures/pred.png?raw=true" />
