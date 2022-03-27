# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:34:28 2022
@author: FrankNGUEN
#http://ainoodle.website/2020/03/09/ket-hop-object-detection-va-object-tracking-chuong-3-thu-lam-he-thong-ban-toc-do-xe-hoi-tren-cao-toc/
"""
#------------------------------------------------------------------------------
# import thu vien
#------------------------------------------------------------------------------
import cv2
import dlib
import math
import time
#------------------------------------------------------------------------------
#Load model Haarcascade
car_detect = cv2.CascadeClassifier('model/car_detect_harrcascade.xml')
#------------------------------------------------------------------------------
# Dinh nghia cac tham so dai , rong
video              = cv2.VideoCapture('test/highway.mp4')
f_width            = 1280
f_height           = 720
pixels_per_meter   = 1         # Cai dat tham so : so diem anh / 1 met, o day dang de 1 pixel = 1 met
frame_idx          = 0         # Cac tham so phuc vu tracking 
car_number         = 0
fps                = 0
carTracker         = {}
carNumbers         = {}
carStartPosition   = {}
carCurrentPosition = {}
speed              = [None] * 1000
#------------------------------------------------------------------------------
# Ham xoa cac tracker khong tot
#------------------------------------------------------------------------------
def remove_bad_tracker():
	global carTracker, carStartPosition, carCurrentPosition
	# Xoa cac car tracking khong tot
	delete_id_list = []
	# Duyet qua cac car
	for car_id in carTracker.keys():
		# Voi cac car ma conf tracking < 4 thi dua vao danh sach xoa
		if carTracker[car_id].update(image) < 4:
			delete_id_list.append(car_id)
	# Thuc hien xoa car
	for car_id in delete_id_list:
		carTracker.pop(car_id, None)
		carStartPosition.pop(car_id, None)
		carCurrentPosition.pop(car_id, None)
	return
#------------------------------------------------------------------------------
# Ham tinh toan toc do
#------------------------------------------------------------------------------
def calculate_speed(startPosition, currentPosition, fps):
	global pixels_per_meter
	distance_in_pixels          = math.sqrt(math.pow(currentPosition[0] -   	    # Tinh toan khoang cach di chuyen theo pixel
                                         startPosition[0], 2) + math.pow(currentPosition[1] - startPosition[1], 2))
	distance_in_meters          = distance_in_pixels / pixels_per_meter 	        # Tinh toan khoang cach di chuyen bang met
	speed_in_meter_per_second   = distance_in_meters * fps        	                # Tinh toc do met tren giay
	speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6   	            # Quy doi sang km/h
	return speed_in_kilometer_per_hour
#------------------------------------------------------------------------------
while True:
	start_time   = time.time()
	_, image     = video.read()
	if image is None:
		break
	image        = cv2.resize(image, (f_width, f_height))
	output_image = image.copy()
	frame_idx    += 1
	remove_bad_tracker()
	if not (frame_idx % 10): 	                                                     # Thuc hien detect moi 10 frame
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      		 # Thuc hien detect car trong hinh
		cars = car_detect.detectMultiScale(gray, 1.2, 13, 18, (24, 24))
		for (_x, _y, _w, _h) in cars:                                                # Duyet qua cac car detect duoc
			x          = int(_x)
			y          = int(_y)
			w          = int(_w)
			h          = int(_h)
			x_center   = x + 0.5 * w                                                 # Tinh tam diem cua car
			y_center   = y + 0.5 * h
			matchCarID = None
			for carID in carTracker.keys():                                          # Duyet qua cac car da track
				trackedPosition = carTracker[carID].get_position()                   # Lay vi tri cua car da track
				t_x             = int(trackedPosition.left())
				t_y             = int(trackedPosition.top())
				t_w             = int(trackedPosition.width())
				t_h             = int(trackedPosition.height())				
				t_x_center      = t_x + 0.5 * t_w                                    # Tinh tam diem cua car da track
				t_y_center      = t_y + 0.5 * t_h
				# Kiem tra xem co phai ca da track hay khong
				if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
					matchCarID = carID
			if matchCarID is None:                                                    # Neu khong phai car da track thi tao doi tuong tracking moi
				tracker                      = dlib.correlation_tracker()
				tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
				carTracker[car_number]       = tracker
				carStartPosition[car_number] = [x, y, w, h]
				car_number                   +=1
	for carID in carTracker.keys():                                                   # Thuc hien update position cac car
		trackedPosition           = carTracker[carID].get_position()
		t_x                       = int(trackedPosition.left())
		t_y                       = int(trackedPosition.top())
		t_w                       = int(trackedPosition.width())
		t_h                       = int(trackedPosition.height())
		cv2.rectangle(output_image, (t_x, t_y), (t_x + t_w, t_y + t_h), (255,0,0), 4)
		carCurrentPosition[carID] = [t_x, t_y, t_w, t_h]
	end_time = time.time()                                                             # Tinh toan frame per second
	if not (end_time == start_time):
		fps = 1.0/(end_time - start_time)
	for i in carStartPosition.keys():                                                  # Lap qua cac xe da track va tinh toan toc do 
			[x1, y1, w1, h1]    = carStartPosition[i]
			[x2, y2, w2, h2]    = carCurrentPosition[i]
			carStartPosition[i] = [x2, y2, w2, h2]
			if [x1, y1, w1, h1] != [x2, y2, w2, h2]:                                   # Neu xe co di chuyen thi
				# Neu nhu chua tinh toan toc do va toa do hien tai < 200 thi tinh toan toc do
				if (speed[i] is None or speed[i] == 0) and y2<200:
					speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2],fps)
				# Neu nhu da tinh toc do va xe da vuot qua tung do 200 thi hien thi tong do
				if speed[i] is not None and y2 >= 200:
					cv2.putText(output_image, str(int(speed[i])) + " km/h", (x2,  y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#------------------------------------------------------------------------------
	cv2.imshow('video', output_image)
	if cv2.waitKey(1) == ord('q'):                                                     # Detect phim Q
		break
cv2.destroyAllWindows()
#------------------------------------------------------------------------------