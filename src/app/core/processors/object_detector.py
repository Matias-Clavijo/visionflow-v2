from collections import deque
import time

import numpy as np
import os
import logging
import cv2

from src.app.core.utils.string_utils import resolve_path
from src.app.models.frame_data import FrameData

DNN_BACKENDS = {
    'opencv': cv2.dnn.DNN_BACKEND_OPENCV,
    'cuda': cv2.dnn.DNN_BACKEND_CUDA,
    'openvino': cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
}

DNN_TARGETS = {
    'cpu': cv2.dnn.DNN_TARGET_CPU,
    'cuda': cv2.dnn.DNN_TARGET_CUDA,
}

logger = logging.getLogger(__name__)



class ObjectDetector:
    def __init__(self, params):

        self.phasher = cv2.img_hash.PHash_create()

        self.name = params.get("name")
        self.model_path = resolve_path(params.get("model_path")) if params.get("model_path") else None
        self.config_path = resolve_path(params.get("config_path")) if params.get("config_path") else None
        self.classes_path = resolve_path(params.get("classes_path")) if params.get("classes_path") else None

        self.backend = params.get("dnn_backend", "opencv").lower()
        self.target = params.get("dnn_target", "cpu").lower()

        self.scale_factor = float(params.get("scale_factor", 1 / 255.0))
        self.input_size = params.get("input_size", (416, 416))
        self.mean_values = params.get("mean_values", (0, 0, 0))
        self.swap_rb = params.get("swap_rb", True)
        self.crop_image = params.get("crop_image", False)

        self.confidence_threshold = params.get("confidence_threshold", 0.5)
        self.nms_threshold = params.get("nms_threshold", 0.4)

        self.process_every_n_frames = params.get("process_every_n_frames", 1)
        self.strategy_for_skipped_frames = params.get("strategy_for_skipped_frames", "DROP")

        self.frame_count = 0

        self.has_to_process = True
        self.frames_processed = 0
        self.frames_to_cached = 100

        self.processing_times = deque(maxlen=1000)
        self.errors_count = 0

        self.preprocess_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.postprocess_times = deque(maxlen=1000)

        self.net = None
        self.classes = []

        self.output_layers = []
        self.last_detections = []

        self.last_similar_frame_with_no_events = None

        self._load_model()
        self._load_classes()

        logger.info(f"CVTagger initialized with model: {self.model_path}")
        logger.info(f"Using backend: {self.backend}, target: {self.target}")
        logger.info(f"Loaded {len(self.classes)} classes")

    def _ensure_required_data(self, params):
        if not params.get("name"):
            raise ValueError("Name is required")
        if not params.get("model_path"):
            raise ValueError("Model path is required")
        if not params.get("classes_path"):
            raise ValueError("Classes path is required")

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            if self.model_path.endswith('.weights') and self.config_path:
                # YOLO darknet format
                if not os.path.exists(self.config_path):
                    raise FileNotFoundError(f"Config file not found: {self.config_path}")

                self.net = cv2.dnn.readNet(self.model_path, self.config_path)
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            elif self.model_path.endswith('.onnx'):
                self.net = cv2.dnn.readNetFromONNX(self.model_path)

            elif self.model_path.endswith('.pb'):
                if not self.config_path or not os.path.exists(self.config_path):
                    raise FileNotFoundError(f"Config file required for TensorFlow model: {self.config_path}")
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)

            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")

            backend_id = DNN_BACKENDS[self.backend]
            target_id = DNN_TARGETS[self.target]

            self.net.setPreferableBackend(backend_id)
            self.net.setPreferableTarget(target_id)

            logger.info(f"Model loaded with backend={self.backend}, target={self.target}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_classes(self):
        if not self.classes_path or not os.path.exists(self.classes_path):
            raise FileNotFoundError(f"Classes file not found: {self.classes_path}")

        try:
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} class names")
        except Exception as e:
            logger.error(f"Error loading classes: {str(e)}")
            self.classes = [f"class_{i}" for i in range(80)]

    def frames_similar_phash(self, frame1, frame2, threshold=5):
        hash1 = self.phasher.compute(frame1)
        hash2 = self.phasher.compute(frame2)

        distance = self.phasher.compare(hash1, hash2)

        print(f"distancia: {distance}")
        return distance <= threshold

    def _preprocess_frame(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame,
            self.scale_factor,
            self.input_size,
            self.mean_values,
            self.swap_rb,
            crop=self.crop_image
        )
        return blob

    def _postprocess_detections(self, outputs, frame_shape):
        boxes = []
        confidences = []
        class_ids = []

        height, width = frame_shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Scale bounding box back to original image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        index = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        final_boxes = []
        final_confidences = []
        final_class_ids = []

        if len(index) > 0:
            for i in index.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids

    def process(self, data: FrameData | None) -> FrameData | None:
        start_time = time.perf_counter()

        should_process = (self.frame_count % self.process_every_n_frames == 0)

        if should_process:
            if not self.has_to_process:
                if self.frames_processed >= self.frames_to_cached:
                    should_process = True
                    logger.info(f"Stopping cache")
                    self.frames_processed = 0
                    self.has_to_process = True
                else:
                    should_process = False
                    self.frames_processed += 1
            elif self.last_similar_frame_with_no_events is not None:
                same_image = self.frames_similar_phash(data.frame, self.last_similar_frame_with_no_events)
                if same_image:
                    should_process = False
                    self.frames_processed += 1

        self.frame_count += 1

        frame = data.frame

        if not should_process:
            if self.strategy_for_skipped_frames == "DROP":
                return None

            if self.strategy_for_skipped_frames == "CACHE":
                end_time = time.perf_counter()
                if self.last_detections:
                    total_time = (end_time - start_time) * 1000
                    data.metadata[f"processor"] = {
                        "count": len(self.last_detections),
                        "tags": self.last_detections,
                        "event": len(self.last_detections) != 0,
                        "cached": True,
                        "frame_info": {
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "channels": frame.shape[2]
                        },
                        "model_info": {
                            "model_path": self.model_path,
                            "confidence_threshold": self.confidence_threshold,
                            "nms_threshold": self.nms_threshold
                        },
                        "performance": {
                            "processing_time_ms": {
                                "total": total_time,
                                "preprocess": total_time,
                                "inference": 0,
                                "postprocess": 0
                            }
                        }
                    }
                return data
        try:
            preprocess_start = time.perf_counter()
            blob = self._preprocess_frame(frame)
            preprocess_end = time.perf_counter()
            preprocess_ms = (preprocess_end - preprocess_start) * 1000
            self.preprocess_times.append(preprocess_ms)

            inference_start = time.perf_counter()
            self.net.setInput(blob)
            if self.output_layers:
                outputs = self.net.forward(self.output_layers)
            else:
                outputs = self.net.forward()
            inference_end = time.perf_counter()
            inference_ms = (inference_end - inference_start) * 1000
            self.inference_times.append(inference_ms)

            postprocess_start = time.perf_counter()
            boxes, confidences, class_ids = self._postprocess_detections(outputs, frame.shape)

            detections = []
            for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x, y, w, h = box
                detection = {
                    "class_id": int(class_id),
                    "class_name": self.classes[class_id] if class_id < len(self.classes) else "unknown",
                    "confidence": float(confidence),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "center_x": int(x + w // 2),
                        "center_y": int(y + h // 2)
                    }
                }
                detections.append(detection)
            postprocess_end = time.perf_counter()
            postprocess_ms = (postprocess_end - postprocess_start) * 1000
            self.postprocess_times.append(postprocess_ms)

            end_time = time.perf_counter()
            total_processing_ms = (end_time - start_time) * 1000
            self.processing_times.append(total_processing_ms)

            self.last_detections = detections.copy()

            self.has_to_process = not (len(detections) != 0)

            if len(detections) == 0:
                self.last_similar_frame_with_no_events = frame.copy()

            data.metadata[f"processor"] = {
                "count": len(detections),
                "cached": False,
                "tags": detections,
                "event": len(detections) != 0,
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2]
                },
                "model_info": {
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold
                },
                "performance": {
                    "processing_time_ms": {
                        "total": round(total_processing_ms, 2),
                        "preprocess": round(preprocess_ms, 2),
                        "inference": round(inference_ms, 2),
                        "postprocess": round(postprocess_ms, 2)
                    }
                }
            }

            logger.info(
                f"Processed frame {self.frame_count}: {len(detections)} detections in {total_processing_ms:.2f}ms")

        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error processing frame {self.frame_count}: {str(e)}")
            data.metadata[f"{self.name}"] = {"error": str(e)}

        return data

    @property
    def average_processing_time_ms(self):
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    @property
    def max_processing_time_ms(self):
        if not self.processing_times:
            return 0.0
        return max(self.processing_times)

    @property
    def min_processing_time_ms(self):
        if not self.processing_times:
            return 0.0
        return min(self.processing_times)
