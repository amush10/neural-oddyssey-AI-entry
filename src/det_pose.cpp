// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "mmdeploy/detector.hpp"
#include "mmdeploy/pose_detector.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Object detection model path");
DEFINE_ARG_string(pose_model, "Pose estimation model path");

DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "det_pose_output.jpg", "Output image path");
DEFINE_string(skeleton, "coco", R"(Path to skeleton data or name of predefined skeletons: "coco")");

DEFINE_int32(det_label, 0, "Detection label use for pose estimation");
DEFINE_double(det_thr, .5, "Detection score threshold");
DEFINE_double(det_min_bbox_size, -1, "Detection minimum bbox size");

DEFINE_double(pose_thr, 0, "Pose key-point threshold");

int main(int argc, char* argv[]) {
	utils::ParseArguments(argc, argv);

	cv::VideoCapture cap("/dev/video0"); 
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(cv::CAP_PROP_FRAME_COUNT, 30);

	mmdeploy::Device device{FLAGS_device};
	mmdeploy::Detector detector(mmdeploy::Model(ARGS_det_model), device);
	mmdeploy::PoseDetector pose(mmdeploy::Model(ARGS_pose_model), device);

	cv::Mat frame;

	while (true) {
		cap >> frame;

		if (frame.empty()) {
			std::cerr << "Error: Could not grab a frame!" << std::endl;
			break;
		}

		std::string filename = "frame.png";
		cv::imwrite(filename, frame);

		mmdeploy::Detector::Result dets = detector.Apply(frame);

		std::vector<mmdeploy_rect_t> bboxes;
		for (const mmdeploy_detection_t& det : dets) {
			if (det.label_id == FLAGS_det_label && det.score > FLAGS_det_thr) {
				bboxes.push_back(det.bbox);
			}
		}

		mmdeploy::PoseDetector::Result poses = pose.Apply(frame, bboxes);

		if (bboxes.size() != poses.size()) {
			continue;
		}

		utils::Visualize v;
		v.set_skeleton(utils::Skeleton::get(FLAGS_skeleton));
		auto sess = v.get_session(frame);
		for (size_t i = 0; i < bboxes.size(); ++i) {
			sess.add_bbox(bboxes[i], -1, -1);
			sess.add_pose(poses[i].point, poses[i].score, poses[i].length, FLAGS_pose_thr);
		}

		frame = sess.get();

		auto drawP = [&](int i) {
			cv::Point keypoint(poses[0].point[i].x, poses[0].point[i].y);
			cv::circle(frame, keypoint, 5, cv::Scalar(0, 255, 0), 10);
			return keypoint;
		};
		auto dist = [](cv::Point p1, cv::Point p2) {
			auto sqr = [](float x) { return x * x; };
			return std::sqrt(sqr(p2.x - p1.x) + sqr(p2.y - p1.y));
		};
		constexpr float radToDeg = 180. / 3.14159;
		cv::Point head = drawP(4);
		cv::Point shoulder = drawP(6);
		cv::Point hip = drawP(12);

		cv::Point shoulderNorth(shoulder.x, head.y);
		float distShoulderHead = dist(head, shoulder);
		float distShoulderNorth = dist(shoulder, shoulderNorth);
		float headsaPrior = distShoulderNorth / distShoulderHead;
		int neckSeverity = 0;
		if (headsaPrior > 0 && headsaPrior < 1) {
			float headShoulderAngle = std::acos(headsaPrior) * radToDeg;
			std::cout << "HEADSHOULDER ANGLE: " << headShoulderAngle << '\n';
			neckSeverity = headShoulderAngle < 30 ? 0 : headShoulderAngle < 70 ? 1 : 2;
			cv::putText(frame, "neck: " + std::to_string(headShoulderAngle), {50, 50}, 0, 1, cv::Scalar(0,
																							   neckSeverity != 2 ? 255 : 0,
																							   neckSeverity != 0 ? 255 : 0));
		}


		cv::Point hipNorth(hip.x, shoulder.y);
		float distHipShoulder = dist(shoulder, hip);
		float distHipNorth = dist(shoulder, hipNorth);
		float hipsaPrior = distHipNorth / distHipShoulder;
		float backSeverity = 0;
		if (hipsaPrior > 0 && hipsaPrior < 1) {
			float hipShoulderAngle = 90 - std::acos(hipsaPrior) * radToDeg;
			std::cout << "HIPSHOULDER ANGLE: " << hipShoulderAngle << '\n';
			backSeverity = hipShoulderAngle < 10 ? 0 : hipShoulderAngle < 40 ? 1 : 2;
			cv::putText(frame, "back: " + std::to_string(hipShoulderAngle), {50, 100}, 0, 1, cv::Scalar(0,
																							   backSeverity != 2 ? 255 : 0,
																							   backSeverity != 0 ? 255 : 0));
		}

		cv::imshow(FLAGS_output, frame);

		if (cv::waitKey(1) == 'q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}
