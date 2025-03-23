all:
	cd build && cmake .. && make && cd .. && ./build/det_pose rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ --device cpu
