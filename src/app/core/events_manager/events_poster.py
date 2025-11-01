import logging

import cv2
import numpy as np
import os
import time
from collections import deque
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from src.app.core.utils.string_utils import resolve_path
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from src.app.models.frame_data import FrameData
from src.app.models.frames_queue_manager import FrameDescriptor
from src.app.models.shared_frame import SharedFramePool


class EventPoster:
    def __init__(self, params):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.output_dir = resolve_path(params.get("output_dir", "output/video_clips"))
        self.name = params.get("name")
        self.clip_duration = params.get("clip_duration", 10.0)
        self.fps = params.get("fps", 30.0)
        self.codec = params.get("codec", "avc1")
        self.container = params.get("container", "mp4")

        self.max_resolution = params.get("max_resolution", None)
        self.buffer_size = params.get("buffer_size", 1000)

        self.use_cloud_storage = params.get("use_cloud_storage", False)
        self.b2_app_key_id = params.get("b2_app_key_id", "005a7351082aa2d0000000001")
        self.b2_app_key = params.get("b2_app_key", "K005HOQbGe1cEaos7n3PSkB9KvdIhao")
        self.b2_bucket_name = params.get("b2_bucket_name", "visionflow-v1")
        self.b2_folder_path = params.get("b2_folder_path", "")
        self.keep_local_copy = params.get("keep_local_copy", True)

        self.use_mongodb = params.get("use_mongodb", False)
        self.mongo_uri = params.get("mongo_uri", None)
        self.mongo_host = params.get("mongo_host", "localhost")
        self.mongo_port = params.get("mongo_port", 27017)
        self.mongo_database = params.get("mongo_database", "visionflow")
        self.mongo_collection = params.get("mongo_collection", "video_clips")
        self.mongo_username = params.get("mongo_username", None)
        self.mongo_password = params.get("mongo_password", None)

        self.b2_api = None
        self.b2_bucket = None
        if self.use_cloud_storage:
            self._initialize_b2_api()
        elif self.use_cloud_storage:
            self.logger.error("B2SDK not available but cloud storage is enabled. Install b2sdk: pip install b2sdk")
            self.use_cloud_storage = False

        # Initialize MongoDB connection
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection_obj = None
        if self.use_mongodb:
            self._initialize_mongodb()

        self.last_trigger_time = time.time()
        self.frames_per_clip = int(self.clip_duration * self.fps)

        self.max_workers = params.get("max_workers", 1)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize running state
        self._running = True

        self.previous_frame_context_count = 100
        self.after_frame_context_count = 100
        self.actual_frame_context_count = 0
        self.event_detected = False
        self.event_descriptor: FrameDescriptor | None = None
        self.frames_data_pool: SharedFramePool | None = None
        self.frames_descriptors: deque = deque(maxlen=self.previous_frame_context_count + self.after_frame_context_count + 1)

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"VideoClipGenerator initialized:")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Clip duration: {self.clip_duration}s")
        self.logger.info(f"  FPS: {self.fps}")
        self.logger.info(f"  Frames per clip: {self.frames_per_clip}")
        self.logger.info(f"  Buffer size: {self.buffer_size}")
        self.logger.info(f"  Cloud storage enabled: {self.use_cloud_storage}")
        if self.use_cloud_storage:
            self.logger.info(f"  B2 bucket: {self.b2_bucket_name}")
            if self.b2_folder_path:
                self.logger.info(f"  B2 folder: {self.b2_folder_path}")
            self.logger.info(f"  Keep local copy: {self.keep_local_copy}")
        self.logger.info(f"  MongoDB enabled: {self.use_mongodb}")
        if self.use_mongodb:
            if self.mongo_uri:
                self.logger.info(
                    f"  MongoDB URI: {self.mongo_uri[:50]}...")  # Solo mostrar primeros 50 chars por seguridad
            else:
                self.logger.info(f"  MongoDB host: {self.mongo_host}:{self.mongo_port}")
            self.logger.info(f"  MongoDB database: {self.mongo_database}")
            self.logger.info(f"  MongoDB collection: {self.mongo_collection}")

    def register_pool(self, pool: SharedFramePool):
        self.frames_data_pool = pool

    def _initialize_b2_api(self):
        try:
            info = InMemoryAccountInfo()
            self.b2_api = B2Api(info)
            self.b2_api.authorize_account("production", self.b2_app_key_id, self.b2_app_key)
            self.b2_bucket = self.b2_api.get_bucket_by_name(self.b2_bucket_name)
            self.logger.info(f"Successfully connected to B2 bucket: {self.b2_bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize B2 API: {e}")
            self.use_cloud_storage = False

    def _initialize_mongodb(self):
        try:
            if self.mongo_uri:
                connection_string = self.mongo_uri
                self.logger.info(f"Using MongoDB URI connection")
            else:
                if self.mongo_username and self.mongo_password:
                    connection_string = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}/"
                else:
                    connection_string = f"mongodb://{self.mongo_host}:{self.mongo_port}/"
                self.logger.info(f"Using MongoDB host: {self.mongo_host}:{self.mongo_port}")

            self.mongo_client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)

            self.mongo_client.admin.command('ping')

            self.mongo_db = self.mongo_client[self.mongo_database]
            self.mongo_collection_obj = self.mongo_db[self.mongo_collection]

            self.logger.info(f"Successfully connected to MongoDB: {self.mongo_database}.{self.mongo_collection}")

        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            self.use_mongodb = False
        except Exception as e:
            self.logger.error(f"Error initializing MongoDB: {e}")
            self.use_mongodb = False

    def _upload_to_b2(self, local_filepath, filename):
        def upload_file():
            try:
                if not self.b2_bucket:
                    self.logger.error("B2 bucket not initialized")
                    return False

                bucket_filename = filename
                if self.b2_folder_path:
                    folder_path = self.b2_folder_path.strip('/')
                    if folder_path:
                        bucket_filename = f"{folder_path}/{filename}"

                self.b2_bucket.upload_local_file(
                    local_file=local_filepath,
                    file_name=bucket_filename
                )

                self.logger.info(f"Successfully uploaded {bucket_filename} to B2 bucket")

                if not self.keep_local_copy:
                    try:
                        os.remove(local_filepath)
                        self.logger.info(f"Deleted local file: {filename}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete local file {filename}: {e}")

                return True

            except Exception as e:
                self.logger.error(f"Failed to upload {filename} to B2: {e}")
                return False

        if self.use_cloud_storage and self.b2_bucket:
            self.executor.submit(upload_file)
        else:
            self.logger.warning("Cloud storage not available for upload")

    def _save_metadata_to_mongodb(self, video_metadata):
        def save_metadata():
            try:
                if self.mongo_collection_obj is None:
                    self.logger.error("MongoDB collection not initialized")
                    return False

                if 'created_at' not in video_metadata:
                    video_metadata['created_at'] = datetime.now(timezone.utc)

                result = self.mongo_collection_obj.insert_one(video_metadata)

                self.logger.info(f"Successfully saved metadata to MongoDB with ID: {result.inserted_id}")
                return True

            except PyMongoError as e:
                self.logger.error(f"Failed to save metadata to MongoDB: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error saving metadata: {e}")
                return False

        if self.use_mongodb is not None and self.mongo_collection_obj is not None:
            self.executor.submit(save_metadata)
        else:
            self.logger.warning("MongoDB not available for metadata storage")

    def _resize_frame_if_needed(self, frame):
        if self.max_resolution is None:
            return frame

        height, width = frame.shape[:2]
        max_width, max_height = self.max_resolution

        width_scale = max_width / width if width > max_width else 1.0
        height_scale = max_height / height if height > max_height else 1.0
        scale = min(width_scale, height_scale)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return frame

    def _should_trigger_clip(self, descriptor: FrameDescriptor) -> bool:
        if not self.event_detected:
            self.event_detected = descriptor.metadata.get("processor", {}).get("event", False)
            if self.event_detected:
                self.event_descriptor = descriptor
        else:
            self.actual_frame_context_count += 1
        return self.actual_frame_context_count == (self.after_frame_context_count + self.previous_frame_context_count) + 1

    def _get_clip_frames(self, descriptor: FrameDescriptor):
        result = []

        for _ in range(len(self.frames_descriptors)):
            descriptor = self.frames_descriptors.pop()
            frame = self.frames_data_pool.to_numpy(descriptor.shm_idx)

            data = FrameData(
                frame_id=descriptor.frame_id,
                frame=frame,
                timestamp=None,
                metadata=descriptor.metadata,
                descriptor = descriptor
            )

            result.insert(0, data)

        return result

    def _write_video_clip_async(self, frames_data, filename):
        def write_video():
            writer = None
            try:
                filepath = os.path.join(self.output_dir, filename)

                if not frames_data:
                    self.logger.warning(f"No frames to write for clip: {filename}")
                    return

                sample_frame = self._resize_frame_if_needed(frames_data[0].frame)
                height, width = sample_frame.shape[:2]

                self.logger.info(f"Creating video writer: {filename}")

                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))

                frames_written = 0
                for frame_data in frames_data:
                    try:
                        # frame_data is a FrameData object, not a dictionary
                        frame = self._resize_frame_if_needed(frame_data.frame)
                        
                        # Additional validation before writing
                        if frame is None or frame.size == 0:
                            self.logger.error(f"Invalid frame for writing: {frame_data.frame_id}")
                            continue
                            
                        # Check frame properties
                        if len(frame.shape) != 3 or frame.shape[2] != 3:
                            self.logger.error(f"Invalid frame format for video writing: {frame.shape}")
                            continue
                            
                        if frame.dtype != np.uint8:
                            self.logger.warning(f"Converting frame dtype from {frame.dtype} to uint8")
                            frame = frame.astype(np.uint8)

                        if frames_written == 0:
                            self.logger.info(f"First frame properties - Shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
                            
                            # Save first frame as image for debugging
                            debug_img_path = os.path.join(self.output_dir, f"debug_frame_{frame_data.frame_id}.jpg")
                            cv2.imwrite(debug_img_path, frame)
                            self.logger.info(f"Saved debug frame: {debug_img_path}")
                        
                        writer.write(frame)
                        frames_written += 1
                        self.frames_data_pool.release(frame_data.descriptor.shm_idx)


                    except Exception as e:
                        self.logger.error(f"Error writing frame to video: {e}")
                        continue

                writer.release()

                if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:  # At least 1KB
                    self.logger.info(f"Successfully created video clip: {filename} ({frames_written} frames)")

                    if self.use_cloud_storage:
                        self._upload_to_b2(filepath, filename)
                else:
                    self.logger.error(f"Video file seems corrupted or too small: {filename}")

            except Exception as e:
                self.logger.error(f"Error creating video clip {filename}: {e}")
            finally:
                if writer is not None and writer.isOpened():
                    writer.release()

        self.executor.submit(write_video)
        self.actual_frame_context_count = 0
        self.event_detected = False



    def _generate_clip(self, descriptor: FrameDescriptor):
        try:
            clip_frames = self._get_clip_frames(descriptor)
            print(f"son {len(clip_frames)} frames")

            if len(clip_frames) < 5:
                self.logger.warning(f"Not enough frames for clip generation: {len(clip_frames)}")
                return

            # Use frame_id as base filename but add proper extension
            base_filename = self.event_descriptor.frame_id
            filename = f"{base_filename}.{self.container}"

            self._write_video_clip_async(clip_frames, filename)
            self._save_metadata_to_mongodb(self.event_descriptor.metadata)

            self.last_trigger_time = time.time()

        except Exception as e:
            self.logger.error(f"Error generating clip: {e}")

    def _add_descriptor(self, descriptor: FrameDescriptor):
        try:
            if len(self.frames_descriptors) == self.frames_descriptors.maxlen:
                if self.frames_data_pool:
                    evicted = self.frames_descriptors.pop()
                    self.frames_data_pool.release(evicted.shm_idx)
            # Always append the incoming descriptor (do not overwrite it with evicted)
            self.frames_descriptors.append(descriptor)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def process(self, descriptor: FrameDescriptor):
        start_time = time.perf_counter()
        try:
            self._add_descriptor(descriptor)
            if self._should_trigger_clip(descriptor):
                print("generando clip")
                self._generate_clip(descriptor)

            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error processing frame in VideoClipGenerator: {e}")
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000

    def stop(self):
        self.logger.info("Stopping VideoClipGenerator...")
        self._running = False

        try:
            # Generate final clip from remaining frames
            if len(self.frame_buffer) > 5:
                self.logger.info(f"Generating final clip from remaining {len(self.frame_buffer)} frames...")
                self._generate_clip()
            else:
                self.logger.debug(f"Skipping final clip generation - only {len(self.frame_buffer)} frames in buffer")

            # Wait for any pending uploads and metadata saves to complete
            if self.use_cloud_storage or self.use_mongodb:
                self.logger.info("Waiting for pending uploads and metadata saves to complete...")
                try:
                    self.executor.shutdown(wait=True)
                    self.logger.debug("All background tasks completed successfully")
                except Exception as e:
                    self.logger.error(f"Error waiting for background tasks: {e}")

            # Clear frame buffer
            if hasattr(self, 'frame_buffer'):
                self.frame_buffer.clear()
                self.logger.debug("Frame buffer cleared")

        except Exception as e:
            self.logger.error(f"Error during VideoClipGenerator shutdown: {e}")
