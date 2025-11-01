import uuid

import cv2
import time
import threading
import logging

from src.app.models.frame_data import FrameData

class RtspCapturer:

    def __init__(self, params):
        self.logger = logging.getLogger(self.__class__.__name__)

        self._ensure_required_data(params)

        self.name = params.get("name")
        self.device_name = params.get("device_name")
        self.rtsp_url = params.get("rtsp_url")
        self.username = params.get("username", "")
        self.password = params.get("password", "")

        self.timeout = params.get("timeout", 10)
        self.buffer_size = params.get("buffer_size", 10)

        self.cap = None
        self.running = False
        self.output_queue = None

        self.reconnect_attempts = 0
        self.max_reconnect_attempts = params.get("max_reconnect_attempts", 5)
        self.reconnect_delay = params.get("reconnect_delay", 1.0)

        self.thread = None

    def _ensure_required_data(self, params):
        if not params.get("name"):
            raise ValueError("Name is required")
        if not params.get("rtsp_url"):
            raise ValueError("RTSP URL is required")

        rtsp_url = params.get("rtsp_url")
        if not rtsp_url.startswith(("rtsp://", "rtmp://")):
            raise ValueError("Invalid RTSP URL format. Must start with 'rtsp://' or 'rtmp://'")

    def register_output_queue(self, output_queue):
        self.output_queue = output_queue
        self.logger.info(f"Registered external output queue for {self.name}")

    def _configure_capture(self):
        if not self.cap:
            return
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except Exception as e:
            self.logger.warning(f"Error configuring capture properties: {str(e)}")

    def get_stream_info(self):
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            return {
                'rtsp_url': self.rtsp_url,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'width': width,
                'height': height
            }
        except Exception as e:
            self.logger.error(f"Error getting stream info: {str(e)}")
            return None

    def _reconnect(self):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None

            time.sleep(self.reconnect_delay)

            self.logger.info(f"Attempting to reconnect to RTSP stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self._configure_capture()

        except Exception as e:
            self.logger.error(f"Error during RTSP reconnection: {str(e)}")

    def _capture_loop(self):
        self.logger.info(f"🎬 Starting capture loop for {self.rtsp_url}")
        while self.running:
            if not self.cap or not self.output_queue:
                self.logger.warning("⚠️ Missing cap or output_queue, waiting...")
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.reconnect_attempts += 1
                    self.logger.warning(
                        f"Failed to read frame from RTSP stream (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")

                    if self.reconnect_attempts > self.max_reconnect_attempts:
                        self.logger.error("Maximum reconnection attempts reached for RTSP stream")
                        self.running = False
                        break

                    self._reconnect()
                    continue

                if self.reconnect_attempts > 0:
                    self.logger.info("RTSP stream connection restored")
                    self.reconnect_attempts = 0

                timestamp = time.time()
                frame_id = str(uuid.uuid4())

                data = FrameData(
                    frame_id=f"{frame_id}",
                    frame=frame,
                    timestamp=timestamp,
                    metadata={
                        "frame_id": frame_id,
                        "device": self.device_name,
                        "timestamp": timestamp
                    }
                )

                if self.output_queue.full():
                    self.logger.warning(f"Skipped frame {data.frame_id} due to full queue")
                    continue

                try:
                    self.output_queue.put_nowait(data)
                except Exception as e:
                    self.logger.error(f"❌ Failed to queue frame {data.frame_id}: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error capturing frame from RTSP: {str(e)}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts <= self.max_reconnect_attempts:
                    self._reconnect()
                else:
                    self.running = False
                    break

    def start(self):
        if not self.output_queue:
            raise RuntimeError("Output queue not registered. Call register_output_queue() first.")
            
        try:
            self.logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open RTSP stream: {self.rtsp_url}")

            self._configure_capture()

            stream_info = self.get_stream_info()
            if stream_info:
                self.logger.info(f"RTSP Stream Info: \n "
                                 f"URL: {stream_info['rtsp_url']}\n"
                                 f"Resolution: {stream_info['resolution']}\n"
                                 f"FPS: {stream_info['fps']}")

            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            self.logger.info(f"Started capturing from RTSP stream: {self.rtsp_url}")

        except Exception as e:
            self.logger.error(f"Failed to start RTSP capture: {str(e)}")
            raise RuntimeError(f"Could not start RTSP capture: {str(e)}")

    def _clear_queue(self):
        if not self.output_queue:
            return
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break

    def stop(self):
        self.logger.info(f"Stopping RTSP capture from: {self.rtsp_url}")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.logger.debug("Waiting for RTSP capture thread to stop...")
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.logger.warning("RTSP capture thread did not stop gracefully")

        if self.cap:
            try:
                self.cap.release()
                self.logger.debug("RTSP stream released successfully")
            except Exception as e:
                self.logger.error(f"Error releasing RTSP stream: {e}")
            finally:
                self.cap = None

        if self.output_queue:
            try:
                self._clear_queue()
                self.logger.debug("Output queue cleared")
            except Exception as e:
                self.logger.error(f"Error clearing queue: {e}")
