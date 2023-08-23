# Project_torchReID
1. clone
2. cd Project_torchReID/torchReID
3. unzip deep-person-reid.zip
4. download weight
   https://drive.google.com/file/d/1ZjnQIggA1m69Dq-9uzXD1YrQMyoDUt7r/view?usp=drive_link
5. put it into deep_person_reid/log/resnext101/model/
6. type in terminal : "conda env create -f environment.yml" and activate
7. mkdir our_datum folder
8. put original video input in our_datum/input/
9. put yolo npy folder in our_datum/tmp/npys/
10. put preprocessed video in our_datum/tmp/videos/
11. run python main_func.py
