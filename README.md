# Actors Face Recognition
This repository contains the code to replicate the wave app, which recognizes the faces of the actors Aaron Eckhar, Adam Brody, Bradley Cooper, and Adrien Brody

## 1. Clone this repo
```
git clone https://github.com/ksathur/Wave-App--Actors-Face-Recognition.git
cd Wave-App--Actors-Face-Recognition
```
## 2. Environment
This code runs on Python 3.7
### 2.1 Create a conda environment with python 3.7 and activate it.
```
conda create -n face_recog_wave python=3.7
conda activate face_recog_wave
```
### 2.2 Pip install `requirements.txt`
This will install all the libraries that are require to run the wave app testing.
```
pip install -r requirements.txt
```
## 3. Run
```
wave run wave_face_recognition.py
```
Go to http://localhost:10101/actors_face_detection to visualize the app. A recorded demo video is available at `./demo_video.mp4`.

## 4. Custom Training
- Upload the folder contains the images of each person into `./dataset/raw`
- Change the directory to `./utils`
```
cd utils
```
- Run `face_csv_file_writter.py` to create a `.csv` file which consists of images directories and labels.
```
python `face_csv_file_writter.py`
```
- Go back to previous working directory.
```
cd ../
```
- Run `face_classifier.py` to start the training. The model and the accuracy information will be saved into `./dumps/model` and `./dumps/accuracy` respectively.
```
python `face_classifier.py`
```
- Run `face_tester.py` to evaluate the peformance of the model on validation dataset.
```
python face_tester.py
```
- Run `plot.py` to visualize the performance (via confusion matrix) (`./dumps/accuracy/cfm.png`)
```
python plot.py
```
- To run testing on any test data download the images into `./test_sample` and run the `face_classifier_sample_test.py`
```
python face_classifier_sample_test.py
```
