from retinaface import RetinaFace
import cv2
import os



input_path_to_video = "/content/drive/MyDrive/face_detect_data/test_video.mp4"
frame_path = "/content/drive/MyDrive/face_detect_data/frames"
detected_faces_folder = "/content/drive/MyDrive/face_detect_data/detected_faces"
video_name = "/content/drive/MyDrive/face_detect_data/final_output.avi"



os.makedirs(
    "/content/drive/MyDrive/face_detect_data/frames",
    exist_ok=True
)

os.makedirs(
    "/content/drive/MyDrive/face_detect_data/detected_faces",
    exist_ok=True
)

vid = cv2.VideoCapture(input_path_to_video)

count, success = 0, True
while success:
    success, image = vid.read() # Read frame
    if success:
        cv2.imwrite(f"/content/drive/MyDrive/face_detect_data/frames/frame{count}.jpg", image) # Save frame
        count += 1

vid.release()

print("Total frames extracted:", count)


for path in os.listdir(frame_path):

    image_path = os.path.join(frame_path, path)

    resp = RetinaFace.detect_faces(image_path)

    print(image_path, resp)

    # Read original image
    image = cv2.imread(image_path)

    # Draw bounding boxes
    if isinstance(resp, dict):

        for key, face in resp.items():

            x1, y1, x2, y2 = face["facial_area"]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # Save detected image
    cv2.imwrite(
        os.path.join(detected_faces_folder, path),
        image
    )

print("Face detection completed!")





def generate_video():
    

    # Get all image files
    images = [
        img for img in os.listdir(detected_faces_folder)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Sort images numerically
    images.sort(key=lambda x: int(x.split("frame")[1].split(".")[0]))

    print("Total frames:", len(images))

    # Read first frame
    frame = cv2.imread(os.path.join(detected_faces_folder, images[0]))

    height, width, layers = frame.shape

    # Set original video FPS
    fps = 30

    # Initialize video writer
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    # Append frames
    for image in images:

        frame = cv2.imread(os.path.join(detected_faces_folder, image))

        video.write(frame)

    # Release resources
    video.release()

    print("Video generated successfully!")
    print("Output:", video_name)


generate_video()