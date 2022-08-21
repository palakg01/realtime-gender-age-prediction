import cv2
import numpy as np
import face_recognition

vid_captured = cv2.VideoCapture(0)

while True:
    
    ret, curr_frame = vid_captured.read()
    curr_frame_resized = cv2.resize(curr_frame, (0,0), fx=0.25, fy=0.25)
    faces_array = face_recognition.face_locations(curr_frame_resized, number_of_times_to_upsample=2, model="hog")
    
    for curr_face in faces_array:
        top, right, bottom, left = curr_face
        top*=4
        bottom*=4
        left*=4
        right*=4
        curr_face_sliced = curr_frame[top:bottom, left:right]
        
        # mean vals
        AGE_GENDER_MEAN_VALS = (78.4263377603, 87.7689143744, 114.895847746)
        
        # convert img to blob
        curr_blob = cv2.dnn.blobFromImage(curr_face_sliced, 1, (227,227), AGE_GENDER_MEAN_VALS, swapRB = False)
        
        # 1.GENDER
        # gender array, load models, create cnn net and set input
        gender_labels = ["Male", "Female"]
        gender_protext = "./models/deploy_gender.prototxt"
        gender_caffemodel = "./models/gender_net.caffemodel" 
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(curr_blob)
        
        # predictions
        gender_predictions = gender_cov_net.forward()
        
        # get max from gender labels
        max_label_index = np.argmax(gender_predictions[0])
        gender_predicted = gender_labels[max_label_index]
        
        
        # 2. AGE
        age_labels = ['(0-2)','(3-6)','(7-12)','(13-20)','(21-35)','(36-45)','(46-60)','(61-90)']
        age_protext = "./models/deploy_age.prototxt"
        age_caffemodel = "./models/age_net.caffemodel"
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(curr_blob)
        
        age_predictions = age_cov_net.forward()
        max_label_index_age = np.argmax(age_predictions[0])
        age_predicted = age_labels[max_label_index_age]
        
        
        # display results
        curr_frame = cv2.rectangle(curr_frame,(left, top),(right, bottom), (255, 0, 0), 2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(curr_frame, "Gender:"+gender_predicted + " Age:"+age_predicted, (left, top), font, 0.7, (255,0,0), 1)
    
    cv2.imshow("frame",curr_frame)    
    
    if(cv2.waitKey(1)&0xFF == ord('q')):
        break
    
vid_captured.release()
cv2.destroyAllWindows()