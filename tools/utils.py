def detect_face(self, target, model, resize=(1280, 720)) -> dict:
        """
        Detects faces on the image and returns the boundary boxes.
        :param target: the path to the image.
        :param model: the path to the AI model.
        :param resize: dimensions of the resized image.
        :return: boundary boxes of the detected faces.
        """
        boxes = dict()
        i = 0

        model = YOLO(model)
        if model.ckpt is None:
            print("Failed to load model.")
            return boxes

        image = cv2.imread(target)
        if image is not None:
            image = cv2.resize(image, resize)
            face_result = model.predict(image, conf=0.40)
            parameters = face_result[0].boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                boxes[str(i)] = [x1, y1, h, w]
                i += 1
        else:
            print("Failed to load image.")

        return boxes