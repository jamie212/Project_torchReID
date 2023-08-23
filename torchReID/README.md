# Project_torchReID
1. clone
2. cd Project_torchReID/torchReID
3. unzip deep-person-reid.zip
4. download weight
   https://drive.google.com/file/d/1ZjnQIggA1m69Dq-9uzXD1YrQMyoDUt7r/view?usp=drive_link
5. put it into deep_person_reid/log/resnext101/model/
6. type in terminal : "conda env create -f environment.yml"
7. download data
   https://drive.google.com/file/d/1f_qrqdF2TD65UzepZskBxLuxiTWXSn4c/view?usp=drive_link
   (put this in same folder with main_new.py)
8. step 9~11 if you want to use your data
9. put original video input in our_datum/input/
10. put yolo npy folder in our_datum/tmp/npys/
11. put preprocessed video in our_datum/tmp/videos/
12. run python main_new.py
