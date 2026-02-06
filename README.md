# Cricket-match-detection
Detecting the players in the video where each person should have unique id and bounding boxes using COCO model (YOLOV8l) to get efficient outputs
All the players movement should be tracked using Deepsort
The Bird view like from the top view to the ground every player will looks like a dot using Homography

#bird_view.ipynb
It is the file to pointing the rectangular points from the final_track video and the fround image to plot the top view detections

#player_detection.ipynb
It is the file to detect the players in the court, pointing the players with bounding boxes with unique Ids and the tracking from start to end point movement of each player

#bird_view
It is the image showing top view of a frame from the video
