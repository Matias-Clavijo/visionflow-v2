#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

from src.app.core.events_manager.events_poster import EventPoster

sys.path.append(str(Path(__file__).parent / "src"))

from src.app.core.orchestrators.DirectOrchestrator import DirectOrchestrator
from src.app.core.capturers.rtsp_capturer import RtspCapturer
from src.app.core.processors.object_detector import ObjectDetector


def setup_logging(config):
    log_config = config.get('logging', {})
    level_str = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', 'visionflow.log')

    # Convert string level to logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

def create_default_config():
    config = {
        'rtsp_capturer': {
            "name": "IPCamera01",
            "device_name": "Iphone",
            "rtsp_url": "rtsp://192.168.68.54:8554/preview",
            "max_reconnect_attempts": 5,
            "reconnect_delay": 2.0,
            "max_queue_size": 10000,
            "timeout": 15,
            # Quality reduction settings for better performance
            "target_width": 640,        # Reduce to 640px width (or use quality_factor instead)
            "target_height": 480,       # Reduce to 480px height
            # "quality_factor": 0.5,    # Alternative: scale to 50% of original size
            "jpeg_quality": 80,         # JPEG compression quality (0-100, lower = more compression)
            "frame_skip": 2             # Process every 2nd frame (1 = all frames, 2 = every other, etc.)
        },
        'web_cam_capturer': {
            "name": "WebCam",
            "device_name": "WebCam",
            "connection": 0,
            "max_reconnect_attempts": 5,
            "reconnect_delay": 2.0,
            "max_queue_size": 10000,
            "timeout": 15
        },
        'object_detector': {
            "name": "object_tagger",
            "model_path": "models/yolo/yolov4.weights",
            "classes_path": "models/yolo/coco.names",
            "config_path": "models/yolo/yolov4.cfg",
            "process_every_n_frames": 1,
            "strategy_for_skipped_frames": "CACHE"
        },
        'video_clip_generator': {
            "name": "video_clip_generator",
            "output_dir": "output",
            "use_cloud_storage": True,
            "b2_folder_path": "videos",
            "clip_duration": 30,
            "fps": 25.0,
            "codec": "mp4v",
            "container": "mp4",
            "quality": 100,
            "max_resolution": [1280, 720],
            "buffer_size": 1000,
            "trigger_mode": "time",
            "trigger_interval": 30,
            "max_workers": 3,
            "use_mongodb": True,
            "mongo_uri": "mongodb+srv://tesis:ucu2025tesis@visionflow.92xlyhu.mongodb.net/?retryWrites=true&w=majority&appName=visionflow",
            "mongo_database": "visionflow",
            "mongo_collection": "events"
        },
        'orchestrator': {
            'pool_shape': (1920, 1080, 3),
            'pool_buffers': 300,
            'capturer_queue_size': 10000,
            'processor_queue_size': 50000
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'visionflow.log'
        }
    }
    return config


def initialize_components(config, logger):
    logger.info("Initializing VisionFlow2 components...")

    logger.info("Initializing RTSP Capturer...")
    try:
        rtsp_capturer = RtspCapturer(config['rtsp_capturer'])
        logger.info(f"✓ RTSP Capturer '{rtsp_capturer.name}' initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize RTSP Capturer: {e}")
        return None, None, None

    logger.info("Initializing Object Detector...")
    try:
        object_detector = ObjectDetector(config['object_detector'])
        logger.info(f"✓ Object Detector '{object_detector.name}' initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Object Detector: {e}")
        logger.warning("Continuing without Object Detector - check model files exist")
        object_detector = None

    return rtsp_capturer, object_detector


def setup_orchestrator(components, config, logger):
    rtsp_capturer, object_detector = components

    logger.info("Setting up DirectOrchestrator...")

    orchestrator = DirectOrchestrator(
        pool_shape=config['orchestrator']['pool_shape'],
        pool_buffers=config['orchestrator']['pool_buffers']
    )

    logger.info(f"Registering capturer: {rtsp_capturer.name}")
    orchestrator.register_capturer(
        rtsp_capturer,
        config['orchestrator']['capturer_queue_size']
    )

    if object_detector:
        logger.info(f"Registering processor: {object_detector.name}")
        orchestrator.register_processor(
            object_detector,
            config['orchestrator']['processor_queue_size']
        )
    else:
        logger.warning("Skipping processor registration - Object Detector not available")


    logger.info(f"Registering events manager")
    orchestrator.register_events_manager(EventPoster, config['video_clip_generator'])

    logger.info("Building orchestrator pipeline...")
    try:
        orchestrator.build()
        logger.info("✓ Orchestrator pipeline built successfully")
    except Exception as e:
        logger.error(f"✗ Failed to build orchestrator pipeline: {e}")
        return None

    return orchestrator


def main():
    try:
        config = create_default_config()

        logger = setup_logging(config)
        logger.info("=" * 60)
        logger.info("Starting VisionFlow2 Application")
        logger.info("=" * 60)
        logger.info("Configuration loaded successfully")

        components = initialize_components(config, logger)
        if not any(components):
            logger.error("Failed to initialize required components. Exiting.")
            sys.exit(1)

        orchestrator = setup_orchestrator(components, config, logger)
        if not orchestrator:
            logger.error("Failed to setup orchestrator. Exiting.")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Starting VisionFlow2 Pipeline")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the application")
        # Run the main pipeline
        orchestrator.run()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Shutdown requested by user")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Unexpected error in main application: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)
    finally:
        logger.info("VisionFlow2 application stopped")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()