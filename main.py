import dlib
import os
import statistics
import utils

detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

DIRECTORY = "./pictures"

# Declare the sets of points corresponding to the different facial sets
CALIBRATION_SET = [0, 1, 2, 14, 15, 16]
EYEBROW_SET = range(17, 27)
NOSE_SET = range(27, 36)
EYE_SET = range(36, 48)
MOUTH_SET = range(48, 67)

counter = 1

with open("tsv.txt", "w") as file:
    # print header
    file.write("id\tactor\tmean eyebrow\tmean nose\tmean eye\tmean mouth"
               "\tmean delta eyebrow\tmean delta nose\tmean delta eye\tmean delta mouth\n")

    # Load the directory
    for f in os.listdir(DIRECTORY):
        # Initiate the positions arrays for the current fragment
        average_eyebrow_positions = []
        average_nose_positions = []
        average_eye_positions = []
        average_mouth_positions = []

        # For every frame of the video
        for f2 in os.listdir(DIRECTORY + '/' + f):
            # Load the frame
            img = dlib.load_rgb_image(DIRECTORY + '/' + f + '/' + f2)
            # Detect the face
            dets = detector(img, 1)
            # In the area of the detected face
            for k, d in enumerate(dets):
                # Detect the landmark points of the face
                landmarks = landmark_detector(img, d)

                # Create a calibration set
                calibration_set = []
                # For all calibration points
                for n in CALIBRATION_SET:
                    # Find the coordinates of the point
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # Add the point to the calibration set
                    calibration_set.append((x, y))
                # Calculate the mean of the calibration set
                cal_x = statistics.mean([a_tuple[0] for a_tuple in calibration_set])
                cal_y = statistics.mean([a_tuple[1] for a_tuple in calibration_set])
                cal = (cal_x, cal_y)

                # Create a set
                eyebrow_set = []
                # For all points in the eyebrow set
                for n in EYEBROW_SET:
                    # Find the coordinates of the point
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    eyebrow_set.append((x, y))
                # Calculate the mean
                mean_x = statistics.mean([a_tuple[0] for a_tuple in eyebrow_set])
                mean_y = statistics.mean([a_tuple[1] for a_tuple in eyebrow_set])
                mean_distance = utils.delta((mean_x, mean_y), cal)
                average_eyebrow_positions.append(mean_distance)

                # Create a set
                nose_set = []
                # For all points in the nose set
                for n in NOSE_SET:
                    # Find the coordinates of the point
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    nose_set.append((x, y))
                # Calculate the mean
                mean_x = statistics.mean([a_tuple[0] for a_tuple in nose_set])
                mean_y = statistics.mean([a_tuple[1] for a_tuple in nose_set])
                mean_distance = utils.delta((mean_x, mean_y), cal)
                average_nose_positions.append(mean_distance)

                # Create a set
                eye_set = []
                # For all points in the eye set
                for n in EYE_SET:
                    # Find the coordinates of the point
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    eye_set.append((x, y))
                # Calculate the mean
                mean_x = statistics.mean([a_tuple[0] for a_tuple in eye_set])
                mean_y = statistics.mean([a_tuple[1] for a_tuple in eye_set])
                mean_distance = utils.delta((mean_x, mean_y), cal)
                average_eye_positions.append(mean_distance)

                # Create a set
                mouth_set = []
                # For all points in the mouth set
                for n in MOUTH_SET:
                    # Find the coordinates of the point
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    mouth_set.append((x, y))
                # Calculate the mean
                mean_x = statistics.mean([a_tuple[0] for a_tuple in mouth_set])
                mean_y = statistics.mean([a_tuple[1] for a_tuple in mouth_set])
                mean_distance = utils.delta((mean_x, mean_y), cal)
                average_mouth_positions.append(mean_distance)

        mean_eyebrow = statistics.mean(average_eyebrow_positions)
        mean_nose = statistics.mean(average_nose_positions)
        mean_eye = statistics.mean(average_eye_positions)
        mean_mouth = statistics.mean(average_mouth_positions)

        delta_eyebrow = []
        for i in range(1, len(average_eyebrow_positions)):
            delta_eyebrow.append(abs(average_eyebrow_positions[i] - average_eyebrow_positions[i-1]))
        mean_delta_eyebrow = statistics.mean(delta_eyebrow)

        delta_nose = []
        for i in range(1, len(average_nose_positions)):
            delta_nose.append(abs(average_nose_positions[i] - average_nose_positions[i-1]))
        mean_delta_nose = statistics.mean(delta_nose)

        delta_eye = []
        for i in range(1, len(average_eye_positions)):
            delta_eye.append(abs(average_eye_positions[i] - average_eye_positions[i-1]))
        mean_delta_eye = statistics.mean(delta_eye)

        delta_mouth = []
        for i in range(1, len(average_mouth_positions)):
            delta_mouth.append(abs(average_mouth_positions[i] - average_mouth_positions[i-1]))
        mean_delta_mouth = statistics.mean(delta_mouth)

        file.write(f + '\t' + f[18:] + '\t' + str(mean_eyebrow) + '\t' + str(mean_nose)
                   + '\t' + str(mean_eye) + '\t' + str(mean_mouth)
                   + '\t' + str(mean_delta_eyebrow) + '\t' + str(mean_delta_nose) + '\t'
                   + str(mean_delta_eye) + '\t' + str(mean_delta_mouth))
        file.write("\n")
        file.flush()
        print("finished video %s of %s" % (counter, len(os.listdir(DIRECTORY))))
        counter += 1
