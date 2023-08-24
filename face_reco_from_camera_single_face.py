# Real-time face detection and recognition for single face
# Do detection -> recognize face, new face -> do re-recognition
# No OT here, OT will be used only for multi faces

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import logging

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # For FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # Save the features of faces in the database
        self.features_known_list = []
        # Save the name of faces in the database
        self.face_name_known_list = []

        # List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # List to save names of objects in current frame
        self.current_frame_name_list = []

        # cnt for faces in frame N-1 and N
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    # Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    # Update FPS of video stream
    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # putText on cv2 window
    def draw_note(self, img_rd):
        # Add some statements
        cv2.putText(img_rd, "Face Recognizer for single face", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # Write names under ROI
        logging.debug(self.current_frame_name_list)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(xy=self.current_frame_face_position_list[0], text=self.current_frame_name_list[0], font=self.font,
                  fill=(255, 255, 0))
        img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1. Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2. Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3. Update cnt for faces in frames
                self.last_frame_faces_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.1 If cnt not changes, 1->1 or 0->0
                if self.current_frame_face_cnt == self.last_frame_faces_cnt:
                    logging.debug("scene 1: No face cnt changes in this frame!!!")

                    if "unknown" in self.current_frame_name_list:
                        logging.debug("   >>> reclassify_interval_cnt")
                        self.reclassify_interval_cnt += 1

                    # 4.1.1 One face in this frame
                    if self.current_frame_face_cnt == 1:
                        if self.reclassify_interval_cnt == self.reclassify_interval:
                            logging.debug("  scene 1.1 Re-classify for current frame")

                            self.reclassify_interval_cnt = 0
                            self.current_frame_face_feature_list = []
                            self.current_frame_face_X_e_distance_list = []
                            self.current_frame_name_list = []

                            for i in range(len(faces)):
                                shape = predictor(img_rd, faces[i])
                                self.current_frame_face_feature_list.append(
                                    face_reco_model.compute_face_descriptor(img_rd, shape))

                            # a. Traversal all the faces in the database
                            for k in range(len(faces)):
                                self.current_frame_name_list.append("unknown")

                                # b. Positions of faces captured
                                self.current_frame_face_position_list.append(tuple(
                                    [faces[k].left(),
                                     int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                                # c. For every face detected, compare it with all the faces in the database
                                for i in range(len(self.features_known_list)):
                                    # If the data of person_X is not empty
                                    if str(self.features_known_list[i][0]) != '0.0':
                                        e_distance_tmp = self.return_euclidean_distance(
                                            self.current_frame_face_feature_list[k],
                                            self.features_known_list[i])
                                        logging.debug("    with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                        self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                    else:
                                        # For empty data
                                        self.current_frame_face_X_e_distance_list.append(999999999)

                                # d. Find the one with minimum e distance
                                similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                    min(self.current_frame_face_X_e_distance_list))

                                if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                    # Modify name if needed
                                    self.current_frame_name_list[k] = self.face_name_known_list[similar_person_num]
                                    logging.debug("    recognition result for face %d: %s", k + 1,
                                                  self.face_name_known_list[similar_person_num])
                                else:
                                    logging.debug("    recognition result for face %d: %s", k + 1, "unknown")
                        else:
                            logging.debug(
                                "  scene 1.2 No re-classification needed for current frame")
                            # Get ROI positions
                            for k, d in enumerate(faces):
                                cv2.rectangle(img_rd,
                                              tuple([d.left(), d.top()]),
                                              tuple([d.right(), d.bottom()]),
                                              (255, 255, 255), 2)

                                self.current_frame_face_position_list[k] = tuple(
                                    [faces[k].left(),
                                     int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)])

                                img_rd = self.draw_name(img_rd)

                # 4.2 If face cnt changes, 1->0 or 0->1
                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []

                    # 4.2.1 Face cnt 0->1
                    if self.current_frame_face_cnt == 1:
                        logging.debug("  scene 2.1 Get faces in this frame and do face recognition")
                        self.current_frame_name_list = []

                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))

                        # a. Traversal all the faces in the database
                        for k in range(len(faces)):
                            self.current_frame_name_list.append("unknown")

                            # b. Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # c. For every face detected, compare it with all the faces in database
                            for i in range(len(self.features_known_list)):
                                # If data of person_X is not empty
                                if str(self.features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.features_known_list[i])
                                    logging.debug("    with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # Empty data for person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # d. Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                # Modify name if needed
                                self.current_frame_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("    recognition result for face %d: %s", k + 1,
                                              self.face_name_known_list[similar_person_num])
                            else:
                                logging.debug("    recognition result for face %d: %s", k + 1, "unknown")

                        if "unknown" in self.current_frame_name_list:
                            self.reclassify_interval_cnt += 1

                    # 4.2.1 Face cnt 1->0
                    elif self.current_frame_face_cnt == 0:
                        logging.debug("  scene 2.2 No face in this frame!!!")

                        self.reclassify_interval_cnt = 0
                        self.current_frame_name_list = []
                        self.current_frame_face_feature_list = []

                # 5. Add note on cv2 window
                self.draw_note(img_rd)

                if kk == ord('q'):
                    break

                self.update_fps()

                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
