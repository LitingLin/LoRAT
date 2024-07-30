def opencv_no_multithreading():
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass
