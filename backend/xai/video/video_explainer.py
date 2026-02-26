import cv2
from xai.utils.preprocessing import preprocess_image


class VideoExplainer:

    def __init__(self, grad_cam):
        """
        grad_cam → instance of GradCAM class
        """
        self.grad_cam = grad_cam

    def explain(self, video_path, max_frames=5):

        cap = cv2.VideoCapture(video_path)

        results = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            input_tensor = preprocess_image(frame)

            heatmap = self.grad_cam.generate(input_tensor, frame)
            results.append(heatmap)

            frame_count += 1

        cap.release()

        return results