from urllib import request
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
import sqlite3
import sys
from tensorflow.keras.models import load_model
import sqlite3
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from joblib import dump, load
import sheet
import pandas as pd
from datetime import datetime
import socket
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders



class MovieSplashScreen(QtWidgets.QSplashScreen):

    def __init__(self, pathToGIF):
        # self.setWindowFlag(Qt.FramelessWindowHint)
        self.movie = QtGui.QMovie(pathToGIF)
        self.movie.jumpToFrame(0)
        pixmap = QtGui.QPixmap(self.movie.frameRect().size())
        QtWidgets.QSplashScreen.__init__(self, pixmap)
        self.movie.frameChanged.connect(self.repaint)
        QtCore.QTimer.singleShot(7000, self.showWindow)

    def showWindow(self):
        splash.close()
        mainwindow = MainWindow()
        widget.addWidget(mainwindow)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def showEvent(self, event):
        self.movie.start()

    def hideEvent(self, event):
        self.movie.stop()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        QtWidgets.QMainWindow.__init__(self, None)
        loadUi("loginForm.ui", self)
        self.login.clicked.connect(self.btnClicked)

    def btnClicked(self):
        user = self.name.text()
        password = self.passw.text()
        conn = sqlite3.connect("dbtest.db")
        cur = conn.cursor()
        query = 'SELECT email FROM instructors WHERE email =\'' + user + "\'"
        cur.execute(query)
        valUser = cur.fetchone()
        if len(user) == 0 or len(password) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please input all fields.")
            msg.setWindowTitle("Error")
            msg.exec_()
        elif not valUser:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please input valid mail")
            msg.setWindowTitle("Error")
            msg.exec_()

        else:
            conn = sqlite3.connect("dbtest.db")
            cur = conn.cursor()
            query = 'SELECT password FROM instructors WHERE email =\'' + user + "\'"
            cur.execute(query)
            result_pass = cur.fetchone()[0]
            if result_pass == password:
                screen2 = Screen2()
                widget.addWidget(screen2)
                widget.setCurrentIndex(widget.currentIndex() + 1)
                fo = open('user.txt', 'w')
                fo.write(user)
                fo.close()

            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Invalid password")
                msg.setWindowTitle("Error")
                msg.exec_()


class Screen2(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self, None)
        loadUi("second screen.ui", self)
        self.off.clicked.connect(self.stop)
        self.out.clicked.connect(self.logout)
        self.start.clicked.connect(self.go)

        self.course.currentTextChanged.connect(self.onCurrentTextChanged)

        fo = open('user.txt', 'r')
        user = fo.read()
        fo.close()

        conn = sqlite3.connect("dbtest.db")
        cur = conn.cursor()
        query = 'SELECT inst_id FROM instructors WHERE email =\'' + user + "\'"
        cur.execute(query)
        r = cur.fetchone()
        conn = sqlite3.connect("dbtest.db")
        curr = conn.cursor()
        query = 'SELECT DISTINCT c_id FROM studnet_course WHERE ins_id =\'' + str(r[0]) + "\'"
        curr.execute(query)
        idCr = curr.fetchall()
        for rec in idCr:
            conn = sqlite3.connect("dbtest.db")
            cur = conn.cursor()
            query = 'SELECT DISTINCT courses.name FROM courses JOIN studnet_course ON studnet_course.c_id=courses.course_id WHERE studnet_course.c_id=\'' + str(
                rec[0]) + "\'"
            cur.execute(query)
            nameCr = cur.fetchall()
            conn.commit()
            conn.close()
            for name in nameCr:
                self.course.addItems(name)

    def onCurrentTextChanged(self, text):
        self.grade.clear()
        conn = sqlite3.connect('dbtest.db')
        c = conn.cursor()
        q = 'SELECT DISTINCT grade FROM courses WHERE name =\'' + text + "\'"
        c.execute(q)
        rec = c.fetchall()
        conn.commit()
        conn.close()
        for record in rec:
            self.grade.addItems(record)

    def go(self):
        def image_recognize(image, all_data):
            image = cv2.resize(image, (640, 480))
            (h, w) = image.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                              (104.0, 177.0, 123.0),
                                              swapRB=False, crop=False)
            # person_id =None
            detector.setInput(imageBlob)
            detections = detector.forward()
            for i in range(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.8:
                    # compute the (x, y)-coordinates of the bounding box for the face

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI

                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large

                    if fW < 30 or fH < 30:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0),
                                                     swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # perform classification to recognize the face

                    preds = recognizer.predict(vec)[0]
                    j = np.argmax(preds)

                    proba = preds[j]
                    name = le.classes_[j]

                    # print(name)
                    if proba >= 0.90:
                        text = name
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(image, text, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
                    else:

                        name = 'unrecognized'
                        text = name
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(image, text, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

                    all_data = sheet.save(name, all_data)

                    # print(all_data)
            return all_data, np.array(image)

        # load serialized face detector
        protoPath = "face_detection_model/deploy.prototxt"
        modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load serialized face embedding model
        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

        # load the actual face recognition model along with the label encoder
        recognizer = load_model('recognizer.h5')

        # le = pickle.loads(open("output/le.pickle", "rb").read())
        le = load('output/le.joblib')

        # initialize the video stream, then allow the camera sensor to warm up
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # start the FPS throughput estimator
        fps = FPS().start()
        all_data = None

        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            all_data, frame = image_recognize(frame, all_data)
            # update the FPS counter
            fps.update()

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # stop the timer and display FPS information
        fps.stop()

        folder_name = datetime.today().strftime('%Y-%m-%d')
        file_name = datetime.today().strftime('%H.%M.%S')
        file_name = str(file_name).replace(':', '.')

        '''-------------------------- check folder name is exists---------------------------------'''
        os.chdir('F:/fourth year/Graduation project/gui/result')
        try:
            # Create target Directory
            os.mkdir(folder_name)

        except FileExistsError:
            print("Directory is already exists")

        # Create target directory & all intermediate directories if don't exists
        os.chdir('F:/fourth year/Graduation project/gui/result')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        '''///////////////////////////////database//////////////////////////////'''

        try:
            conn = sqlite3.connect('F:/fourth year/Graduation project/gui/dbtest.db')

            # execute(all_data.get('person_id')[0],)q
            # print(all_data.get('person_id')[0])
            cursor = conn.cursor()

            columns_wanted = ['stu_id', 'f_name', 'l_name', 'section']
            qs = "SELECT {} FROM students WHERE  stu_id  =?".format(','.join(columns_wanted))

            for pid in all_data.get('person_id', []):
                cursor.execute(qs, (pid,))
                values = cursor.fetchall()

                if len(values) > 0:
                    values = values[0]

                print('\nValues:', values, )
                for i, v in enumerate(values):
                    all_data.setdefault(columns_wanted[i], []).append(v)


        except sqlite3.Error as error:
            print("Error while creating a sqlite table", error)

        '''///////////////////////////////remove duplicate/////////////////////////////'''

        df = pd.DataFrame(all_data, columns=columns_wanted)
        df.drop_duplicates(subset='stu_id', keep="last", inplace=True)

        '''///////////////////////////////save file/////////////////////////////'''
        excel_save_path = r'F:\fourth year\Graduation project\gui\result\{}\{}.xlsx'.format(folder_name, file_name)
        fo = open('F:/fourth year/Graduation project/gui/path.txt', 'w')
        fo.write(excel_save_path)
        fo.close()
        df['name'] = df['f_name'] + ' ' + df['l_name']
        df.drop(['f_name', 'l_name'], axis=1, inplace=True)
        columns_wanted = ['stu_id', 'name', 'section']
        df = df.reindex(columns=columns_wanted)
        df.to_excel(excel_save_path, index=False)

        # cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def stop(self):
        fo = open('F:/fourth year/Graduation project/gui/user.txt', 'r')
        user = fo.read()
        fo.close()
        course = str(self.course.currentText())
        grade = str(self.grade.currentText())

        username = 'it.unit.bfci@gmail.com'
        password = r"H123456789."
        reciepent = user
        pp = open('F:/fourth year/Graduation project/gui/path.txt', 'r')
        path = pp.read()
        pp.close()

        def send_mail(send_from, send_to, subject, text, file, username='', password='', isTls=True):
            msg = MIMEMultipart()
            msg['From'] = send_from
            msg['To'] = send_to
            msg['Date'] = formatdate(localtime=True)
            msg['Subject'] = subject
            msg.attach(MIMEText(text))

            part = MIMEBase('application', "octet-stream")
            part.set_payload(open(file, "rb").read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename=file)
            msg.attach(part)

            smtp = smtplib.SMTP('smtp.gmail.com', 587)

            if isTls:
                smtp.ehlo()
                smtp.starttls()

            smtp.login(username, password)
            smtp.sendmail(send_from, send_to, msg.as_string())
            smtp.quit()

        send_mail("it unit", reciepent, course + " attendance", "Grade: " + grade, path, username, password)

    def logout(self):
        mainwindow = MainWindow()
        widget.addWidget(mainwindow)
        widget.setCurrentIndex(widget.currentIndex() - 1)


app = QApplication(sys.argv)
app.setApplicationName("Attendance system")
pathToGIF = "qq.gif"
splash = MovieSplashScreen(pathToGIF)
widget = QtWidgets.QStackedWidget()
widget.addWidget(splash)
widget.setFixedWidth(780)
widget.setFixedHeight(600)
widget.show()
app.exec_()
