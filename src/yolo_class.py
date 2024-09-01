import torch 
import torchvision.transforms as transforms
from torchvision import models
import cv2
from tqdm import tqdm
#import albumentations as A
from ultralytics import YOLO 
import pickle
import pandas as pd
import uuid
import os 
import time
import numpy as np

class YOLO_detector():
    """
    A class for detecting keypoints and objects using a YOLO-based model.

    The class provides functionality to predict keypoints in images, draw keypoints on images and video frames, 
    and perform object detection in videos. It uses a pre-trained model for keypoint detection and allows for 
    further customisation and application to different media types.
    """
    def __init__(self, model, model_path, pretrained=True):
        """
        Initialises the YOLO_detector class.

        Args:
            model (str): The model name or architecture to use.
            model_path (str): Path to the pre-trained model.
            pretrained (bool, optional): Whether to use a pre-trained model. Defaults to True.
        """
        self.pretrained=pretrained
        self.model_path = model_path
        self.model = models.resnet50(pretrained=self.pretrained)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4,0.4,0.4], std=[0.2,0.2,0.2])
        ])

    def predict(self, image):
        """
        Predicts keypoints for a given image using the model.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: Predicted keypoints.
        """
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_tensor = self.transform(im_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(im_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        orig_h, orig_w = image.shape[:2]
        keypoints[::2] *= orig_w / 224.0
        keypoints[1::2] *= orig_h / 224.0

        return keypoints
    
    def draw_keyP(self, image, keypoints):
        """
        Draws keypoints on the given image.

        Args:
            image (numpy.ndarray): The image on which to draw keypoints.
            keypoints (numpy.ndarray): The keypoints to be drawn.

        Returns:
            numpy.ndarray: Image with keypoints drawn.
        """
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
        return image
    
    def draw_keyP_video(self, video_frames, keypoints):
        """
        Draws keypoints on a series of video frames.

        Args:
            video_frames (list): List of video frames.
            keypoints (numpy.ndarray): The keypoints to be drawn on each frame.

        Returns:
            list: List of video frames with keypoints drawn.
        """
        out_vid_frames = []
        for frame in video_frames:
            frame = self.draw_keyP(frame, keypoints)
            out_vid_frames.append(frame)
        return out_vid_frames


class Object_tracker():
    """
    A class for tracking objects in video sequences using a YOLO model.

    This class allows for detection and tracking of objects across video frames, selection and filtering of 
    objects based on keypoints, and interpolation of object positions. It also provides functionality to 
    draw bounding boxes around detected objects in video frames and identify frames where objects are hit.
    """
    def __init__(self, model_path):
        """
        Initialises the Object_tracker class.

        Args:
            model_path (str): Path to the YOLO model used for object tracking.
        """
        self.model = YOLO(model_path)
        self.columns = ['x1', 'x2', 'y1', 'y2']
        self.UtilsObj = Utils()

    def choose_and_filter(self, keypoints, obj_detections):
        """
        Chooses and filters object detections based on keypoints.

        Args:
            keypoints (numpy.ndarray): Array of keypoints.
            obj_detections (list): List of object detections.

        Returns:
            list: Filtered object detections.
        """
        obj_detections_frame_one = obj_detections[0]
        chosen_obj = self.choose_obj(keypoints, obj_detections_frame_one)
        filtered_obj_detections = []
        for obj_dict in obj_detections:
            filtered_obj_dict = {track_id: bbox for track_id, bbox in obj_dict.items() if track_id in chosen_obj}
            filtered_obj_detections.append(filtered_obj_dict)
        return filtered_obj_detections
    
    def choose_obj(self, keypoints, obj_dict):
        """
        Chooses the object closest to the keypoints from the first frame's detections.

        Args:
            keypoints (numpy.ndarray): Array of keypoints.
            obj_dict (dict): Dictionary of object detections for the first frame.

        Returns:
            list: List of chosen object IDs.
        """
        distances=[]
        for track_id, bbox in obj_dict.items():
            obj_centre = self.UtilsObj.get_box_centre(bbox)

            distance_min = float('inf')
            for i in range(0, len(keypoints), 2):
                keypoints = (keypoints[i], keypoints[i+1])
                distance = self.UtilsObj.calc_distance(obj_centre, keypoints)
                if distance < distance_min:
                    distance_min = distance
            distances.append(track_id, distance_min)

        distances.sort(key = lambda x: x[1])
        chosen_obj = [distances[0][0], distances[1][0]]
        return chosen_obj
                    

    def interp_obj_pos(self, obj_pos):
        """
        Interpolates object positions between frames to fill in any gaps.

        Args:
            obj_pos (list): List of object positions across frames.

        Returns:
            list: Interpolated object positions.
        """
        obj_pos = [x.get(1,[]) for x in obj_pos]
        df_obj_pos = pd.DataFrame(obj_pos, columns=self.columns)
        df_obj_pos = df_obj_pos.interpolate()
        df_obj_pos = df_obj_pos.bfill()

        obj_pos = [{1:x} for x in df_obj_pos.to_numpy().tolist()]

        return obj_pos
    
    def get_obj_shot_frames(self, obj_pos):
        """
        Identifies the frames where an object appears to be hit based on position changes.

        Args:
            obj_pos (list): List of object positions across frames.

        Returns:
            list: Frame numbers where the object is hit.
        """
        obj_pos = [x.get(1, []) for x in obj_pos]
        df_obj_pos = pd.DataFrame(obj_pos, columns=self.columns)
        df_obj_pos['obj_hit'] = 0
        df_obj_pos['mid_y'] = (df_obj_pos['y1'] + df_obj_pos['y2']) / 2
        df_obj_pos['mid_y_roll_mean'] = df_obj_pos['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_obj_pos['del_y'] = df_obj_pos['mid_y_roll_mean'].diff()

        min_frame_for_change = 30
        for i in range(1, len(df_obj_pos)- int(min_frame_for_change * 1.2)):
            negative_pos_change = df_obj_pos['del_y'].iloc[i] > 0 and df_obj_pos['del_y'].iloc[i+1] < 0
            positive_pos_change = df_obj_pos['del_y'].iloc[i] < 0 and df_obj_pos['del_y'].iloc[i+1] > 0
            if negative_pos_change or positive_pos_change:
                change_counter = 0
                for change_frame in range(i+1, i + int(min_frame_for_change * 1.2)+1):
                    negative_pos_change_next_frame = df_obj_pos['del_y'].iloc[i] > 0 and df_obj_pos['del_y'].iloc[change_frame] < 0
                    positive_pos_change_next_frame = df_obj_pos['del_y'].iloc[i] < 0 and df_obj_pos['del_y'].iloc[change_frame] > 0
                    if negative_pos_change and negative_pos_change_next_frame:
                        change_counter += 1
                    elif positive_pos_change and positive_pos_change_next_frame:
                        change_counter += 1

                    if change_counter > min_frame_for_change-1:
                        df_obj_pos['obj_hit'].iloc[i] = 1

        frame_nums_w_hit = df_obj_pos[df_obj_pos['obj_hit']==1].index.tolist()

        return frame_nums_w_hit
    
    def frames_detect(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects objects across multiple video frames.

        Args:
            frames (list): List of video frames.
            read_from_stub (bool, optional): If True, reads detections from a stub file. Defaults to False.
            stub_path (str, optional): Path to the stub file. Defaults to None.

        Returns:
            list: List of detected objects across frames.
        """
        obj_detections= []
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                obj_detections = pickle.load(f)
            return obj_detections
        for frame in frames:
            player_dict = self.frame_detect(frame)
            obj_detections = obj_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(obj_detections, f)
        
        return obj_detections


    def frame_detect(self, frame):
        """
        Detects objects in a single video frame.

        Args:
            frame (numpy.ndarray): The video frame to analyse.

        Returns:
            dict: Dictionary of detected objects with bounding box coordinates.
        """
        res = self.model.predict(frame, conf=0.15[0])
        obj_dict = {}
        for box in res.boxes:
            res = box.xyxy.tolist()[0]
            obj_dict[1] = res

        return obj_dict
    
    def draw_bbox(self, video_frames, obj_detection):
        """
        Draws bounding boxes around detected objects on video frames.

        Args:
            video_frames (list): List of video frames.
            obj_detection (list): List of detected objects with bounding boxes.

        Returns:
            list: Video frames with bounding boxes drawn.
        """
        out_vid_frames = []
        for frame, obj_dict in zip(video_frames, obj_detection):
            for track_id, bbox in obj_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame,f"Object ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            out_vid_frames.append(frame)
        
        return out_vid_frames


class Utils():
    """
    A utility class for handling video operations and performing basic image processing tasks.

    The class includes methods to read and save video files, calculate the centre of a bounding box, and 
    compute the distance between two points. It serves as a support class for video and image processing 
    tasks, facilitating operations such as video frame extraction and bounding box manipulation.
    """
    def __init__(self, video_path=None, out_vid_frames=None, out_vid_path=None):
        """
        Initialises the Utils class.

        Args:
            video_path (str, optional): Path to the input video. Defaults to None.
            out_vid_frames (list, optional): List of output video frames. Defaults to None.
            out_vid_path (str, optional): Path to save the output video. Defaults to None.
        """
        self.video_path = video_path
        self.out_vid_frames = out_vid_frames
        self.out_vid_path = out_vid_path

    def read_video(self):
        """
        Reads video frames from the specified video path.

        No arguments.

        Returns:
            list: List of video frames.
        """
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return frames

    def save_video(self):
        """
        Saves video frames to a specified output path.

        No arguments.

        Returns:
            None.
        """
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_vid_path, fourcc, 24, (self.out_vid_frames[0].shape[1], self.out_vid_frames[0].shape[0]))
        for frame in self.out_vid_frames:
            out.write(frame)
        out.release()

    def get_box_centre(self, bbox):
        """
        Calculates the centre of a bounding box.

        Args:
            bbox (tuple): Coordinates of the bounding box (x1, y1, x2, y2).

        Returns:
            tuple: Coordinates of the centre of the bounding box (centre_x, centre_y).
        """
        x1, y1, x2, y2 = bbox
        centre_x = int((x1 + x2) / 2)
        centre_y = int((y1 + y2) / 2)

        return centre_x, centre_y
    
    def calc_distance(self, p1, p2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            p1 (tuple): Coordinates of the first point.
            p2 (tuple): Coordinates of the second point.

        Returns:
            float: Euclidean distance between the two points.
        """
        return  ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

class YOLO_main():
    """
    A class for managing the YOLO model's training and application to video streams.

    This class provides methods for training a YOLO model using a dataset, capturing video from a camera 
    for real-time object detection, and processing video streams to apply YOLO-based tracking. It is designed 
    to serve as the main interface for YOLO operations, integrating video processing with object detection 
    and tracking functionalities.
    """
    def __init__(self, data_path, image_path, image_number, model, video_path=None, video_capture=False):
        """
        Initialises the YOLO_main class.

        Args:
            data_path (str): Path to the data directory.
            image_path (str): Path to the image directory.
            image_number (int): Number of images to process.
            model (str): The YOLO model to use.
            video_path (str, optional): Path to the input video. Defaults to None.
            video_capture (bool, optional): Whether to capture video from a live source. Defaults to False.
        """
        self.data_path = data_path
        self.image_path = image_path
        self.image_number = image_number
        self.video_capture = video_capture
        self.model = model
        self.video_path = video_path
    
    def train(self):
        """
        Trains the YOLO model using the specified dataset.

        No arguments.

        Returns:
            None.
        """
        IMAGES_PATH = os.path.join(str(self.data_path), str(self.image_path)) #/data/images
        labels = ['threat', 'non-threat']
        number_imgs = self.image_number

        if self.video_capture == True:
            cap = cv2.VideoCapture(0)
            # Loop through labels
            for label in labels:
                print('Collecting images for {}'.format(label))
                time.sleep(5)
                
                # Loop through the range of number of images
                for img_num in range(number_imgs):
                    print('Collecting images for {}, image number {}'.format(label, img_num))

                    ret, frame = cap.read() # Camera feed
                    imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
                    cv2.imwrite(imgname, frame)
                    
                    # Render to the screen
                    cv2.imshow('Image Collection', frame)
                    
                    # 2 second delay between captures
                    time.sleep(2)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()

            for label in labels:
                print('Collecting images for {}'.format(label))
                for img_num in range(number_imgs):
                    print('Collecting images for {}, image number {}'.format(label, img_num))
                    imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
                    print(imgname)   

            os.system("cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2")

    def cam_capture(self):
        """
        Captures video from a camera and performs real-time detection using the YOLO model.

        No arguments.

        Returns:
            None.
        """
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections 
            results = self.model(frame)
            
            cv2.imshow('YOLO', np.squeeze(results.render()))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def video_stream(self):
        """
        Processes a video stream and performs object detection using the YOLO model.

        No arguments.

        Returns:
            None.
        """
        if self.video_path is not None:
            model = self.model
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    exit()
                tracks = model.track(frame, show=False, persist=True)
         
                
    def main(self):
        """
        Main method for processing a video, detecting objects, and saving the output video with annotations.

        No arguments.

        Returns:
            None.
        """
        input_video = "inputVideo.mp4"
        UtilObj = Utils(input_video)
        
        video_frames = UtilObj.read_video()
        ObjDetector = Object_tracker(model_path="path_to_model")

        Obj_detections = ObjDetector.frames_detect(video_frames, read_from_stub=True, stub_path="pathToModel")
        obj_shot_frames = ObjDetector.get_obj_shot_frames(Obj_detections)


        for obj_ind in range(len(obj_shot_frames) - 1):
            start_frame = obj_shot_frames[obj_ind]
            end_frame = obj_shot_frames[obj_ind + 1]
            obj_shot_time_seconds = (end_frame - start_frame) / 24

        output_video_frames = ObjDetector.draw_bbox(video_frames, Obj_detections)

        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        UtilObj.save_video(output_video_frames, "output_videos/output_video.avi")



class DataSynth():
    """
    A class for generating synthetic datasets, including images and YOLO annotations.

    This class handles the creation of composite images with multiple objects, the addition of noise to 
    backgrounds, and the generation of corresponding YOLO annotations. It is used to create training datasets 
    for object detection models, supporting various transformations and ensuring that the generated datasets 
    meet specified criteria for object placement and overlap.
    """
    def __init__(self, object_dict, path_main, img_path, mask_path, x, y, idx, folder, image_number) -> None:
        """
        Initialises the DataSynth class.

        Args:
            object_dict (dict): Dictionary containing object data.
            path_main (str): Main path to the dataset.
            img_path (str): Path to the image.
            mask_path (str): Path to the mask image.
            x (int): X-coordinate for placing the object.
            y (int): Y-coordinate for placing the object.
            idx (int): Index of the object.
            folder (str): Folder where the data is stored.
            image_number (int): Number of images to generate.
        """
        self.object_dict = object_dict
        self.path_main = path_main
        self.img_path = img_path
        self.mask_path = mask_path
        self.x = x
        self.y = y
        self.idx = idx
        self.folder = folder
        self.image_number = image_number

    def file_handle(self):
        """
        Handles file operations for the object dictionary.

        No arguments.

        Returns:
            tuple: Folder name, image paths, mask paths, background image paths, noise image paths, noise mask paths.
        """
        for key, _ in self.object_dict.item():
            folder_name = self.object_dict[key]['folder']

            file_ims = sorted(os.listdir(os.path.join(self.path_main, folder_name, 'images')))
            file_ims = [os.path.join(self.path_main, folder_name, 'images', f) for f in file_ims]

            file_masks = sorted(os.listdir(os.path.join(self.path_main, folder_name, 'masks')))
            file_masks = [os.path.join(self.path_main, folder_name, 'masks', f) for f in file_masks]

            self.object_dict[key]['images'] = file_ims
            self.object_dict[key]['masks'] = file_masks

            files_bg_imgs = sorted(os.listdir(os.path.join(self.path_main, 'bg')))
            files_bg_imgs = [os.path.join(self.path_main, 'bg', f) for f in files_bg_imgs]

            files_bg_noise_imgs = sorted(os.listdir(os.path.join(self.path_main, "bg_noise", "images")))
            files_bg_noise_imgs = [os.path.join(self.path_main, "bg_noise", "images", f) for f in files_bg_noise_imgs]
            files_bg_noise_masks = sorted(os.listdir(os.path.join(self.path_main, "bg_noise", "masks")))
            files_bg_noise_masks = [os.path.join(self.path_main, "bg_noise", "masks", f) for f in files_bg_noise_masks]


            return folder_name, file_ims, file_masks, files_bg_imgs, files_bg_noise_imgs, files_bg_noise_masks
        
    def get_im_and_mask(self):
        """
        Reads and processes an image and its corresponding mask.

        No arguments.

        Returns:
            tuple: Processed image and mask.
        """
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

        mask = cv2.imread(self.mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BAYER_BG2RGB)

        mask_bool = mask[:,:,0] == 0
        mask = mask_bool.astype(np.uint8)

        return img, mask

    def resize_img(self, img, desired_max, desired_min=None):
        """
        Resizes an image to the desired dimensions.

        Args:
            img (numpy.ndarray): Image to be resized.
            desired_max (int): Desired maximum dimension.
            desired_min (int, optional): Desired minimum dimension. Defaults to None.

        Returns:
            numpy.ndarray: Resized image.
        """
        h, w = img.shape[0], img.shape[1]
        
        longest, shortest = max(h, w), min(h, w)
        longest_new = desired_max
        if desired_min:
            shortest_new = desired_min
        else:
            shortest_new = int(shortest * (longest_new / longest))
        
        if h > w:
            h_new, w_new = longest_new, shortest_new
        else:
            h_new, w_new = shortest_new, longest_new
            
        transform_resize = A.Compose([
            A.Sequential([
            A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
            ], p=1)
        ])

        transformed = transform_resize(image=img)
        img_r = transformed["image"]
            
        return img_r

    def resize_transform_obj(self, img, mask, longest_min, longest_max, transforms=False):
        """
        Resizes and applies transformations to an object image and mask.

        Args:
            img (numpy.ndarray): Object image to be resized.
            mask (numpy.ndarray): Object mask to be resized.
            longest_min (int): Minimum dimension for resizing.
            longest_max (int): Maximum dimension for resizing.
            transforms (albumentations.Compose, optional): Additional transformations to apply. Defaults to False.

        Returns:
            tuple: Transformed image and mask.
        """
        h, w = mask.shape[0], mask.shape[1]
        
        longest, shortest = max(h, w), min(h, w)
        longest_new = np.random.randint(longest_min, longest_max)
        shortest_new = int(shortest * (longest_new / longest))
        
        if h > w:
            h_new, w_new = longest_new, shortest_new
        else:
            h_new, w_new = shortest_new, longest_new
            
        transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

        transformed_resized = transform_resize(image=img, mask=mask)
        img_t = transformed_resized["image"]
        mask_t = transformed_resized["mask"]
            
        if transforms:
            transformed = transforms(image=img_t, mask=mask_t)
            img_t = transformed["image"]
            mask_t = transformed["mask"]
            
        return img_t, mask_t

    def transforms(self):
        """
        Defines the transformations to apply to background and object images.

        No arguments.

        Returns:
            tuple: Transformations for objects and background objects.
        """
        transforms_bg_obj = A.Compose([
            A.RandomRotate90(p=1),
            A.ColorJitter(brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.07,
                        always_apply=False,
                        p=1),
            A.Blur(blur_limit=(3,15),
                always_apply=False,
                p=0.5)
        ])

        transforms_obj = A.Compose([
            A.RandomRotate90(p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                                    contrast_limit=0.1,
                                    brightness_by_max=True,
                                    always_apply=False,
                                    p=1)
        ])

        return transforms_obj, transforms_bg_obj

    def add_obj(self, img_comp, mask_comp, img, mask, x, y, idx):
        """
        Adds an object to a composite image and updates the mask.

        Args:
            img_comp (numpy.ndarray): Composite image.
            mask_comp (numpy.ndarray): Composite mask.
            img (numpy.ndarray): Object image.
            mask (numpy.ndarray): Object mask.
            x (int): X-coordinate for placing the object.
            y (int): Y-coordinate for placing the object.
            idx (int): Object index.

        Returns:
            tuple: Updated composite image, composite mask, and added object mask.
        """
        h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
        
        h, w = img.shape[0], img.shape[1]
        
        x = x - int(w/2)
        y = y - int(h/2)
        
        mask_bool = mask == 1
        mask_rgb_b = np.stack([mask_bool, mask_bool, mask_bool], axis=2)
        
        if x >= 0 and y >= 0:
        
            h_part = h - max(0, y+h-h_comp) # h_part - part of the image which gets into the frame of img_comp along y-axis
            w_part = w - max(0, x+w-w_comp) # w_part - part of the image which gets into the frame of img_comp along x-axis

            img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * ~mask_rgb_b[0:h_part, 0:w_part, :] + (img * mask_rgb_b)[0:h_part, 0:w_part, :]
            mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * ~mask_bool[0:h_part, 0:w_part] + (idx * mask_bool)[0:h_part, 0:w_part]
            mask_added = mask[0:h_part, 0:w_part]
            
        elif x < 0 and y < 0:
            
            h_part = h + y
            w_part = w + x
            
            img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
            mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * ~mask_bool[h-h_part:h, w-w_part:w] + (idx * mask_bool)[h-h_part:h, w-w_part:w]
            mask_added = mask[h-h_part:h, w-w_part:w]
            
        elif x < 0 and y >= 0:
            
            h_part = h - max(0, y+h-h_comp)
            w_part = w + x
            
            img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_b[0:h_part, w-w_part:w, :] + (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
            mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * ~mask_bool[0:h_part, w-w_part:w] + (idx * mask_bool)[0:h_part, w-w_part:w]
            mask_added = mask[0:h_part, w-w_part:w]
            
        elif x >= 0 and y < 0:
            
            h_part = h + y
            w_part = w - max(0, x+w-w_comp)
            
            img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * ~mask_rgb_b[h-h_part:h, 0:w_part, :] + (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
            mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * ~mask_bool[h-h_part:h, 0:w_part] + (idx * mask_bool)[h-h_part:h, 0:w_part]
            mask_added = mask[h-h_part:h, 0:w_part]
        
        return img_comp, mask_comp, mask_added
    
    def create_bg_with_noise(self, 
                         bg_max=1920,
                         bg_min=1080,
                         max_objs_to_add=60,
                         longest_bg_noise_max=1000,
                         longest_bg_noise_min=200,
                         blank_bg=False):
        """
        Creates a background image with added noise objects.

        Args:
            bg_max (int, optional): Maximum dimension of the background. Defaults to 1920.
            bg_min (int, optional): Minimum dimension of the background. Defaults to 1080.
            max_objs_to_add (int, optional): Maximum number of objects to add. Defaults to 60.
            longest_bg_noise_max (int, optional): Maximum size of noise objects. Defaults to 1000.
            longest_bg_noise_min (int, optional): Minimum size of noise objects. Defaults to 200.
            blank_bg (bool, optional): Whether to use a blank background. Defaults to False.

        Returns:
            numpy.ndarray: Background image with noise objects added.
        """
    
        _, _, _, files_bg_imgs, files_bg_noise_imgs, files_bg_noise_masks = self.file_handle()

        if blank_bg:
            img_comp_bg = np.ones((bg_min, bg_max,3), dtype=np.uint8) * 255
            mask_comp_bg = np.zeros((bg_min, bg_max), dtype=np.uint8)
        else:    
            idx = np.random.randint(len(files_bg_imgs))
            img_bg = cv2.imread(files_bg_imgs[idx])
            img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
            img_comp_bg = self.resize_img(img_bg, bg_max, bg_min)
            mask_comp_bg = np.zeros((img_comp_bg.shape[0], img_comp_bg.shape[1]), dtype=np.uint8)

        for i in range(1, np.random.randint(max_objs_to_add) + 2):

            transforms_obj, transforms_bg_obj = self.transforms()

            idx = np.random.randint(len(files_bg_noise_imgs))
            img, mask = self.get_im_and_mask(files_bg_noise_imgs[idx], files_bg_noise_masks[idx])
            x, y = np.random.randint(img_comp_bg.shape[1]), np.random.randint(img_comp_bg.shape[0])
            img_t, mask_t = self.resize_transform_obj(img, mask, longest_bg_noise_min, longest_bg_noise_max, transforms=transforms_bg_obj)
            img_comp_bg, _, _ = self.add_obj(img_comp_bg, mask_comp_bg, img_t, mask_t, x, y, i)
            
        return img_comp_bg
    
    def check_areas(mask_comp, obj_areas, overlap_degree=0.3):
        """
        Checks if the object areas in the mask meet the overlap criteria.

        Args:
            mask_comp (numpy.ndarray): Composite mask.
            obj_areas (list): List of object areas.
            overlap_degree (float, optional): Allowed overlap degree. Defaults to 0.3.

        Returns:
            bool: True if areas are acceptable, False otherwise.
        """
        obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
        masks = mask_comp == obj_ids[:, None, None]
        
        ok = True
        
        if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
            ok = False
            return ok
        
        for idx, mask in enumerate(masks):
            if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
                ok = False
                break
                
        return ok   

    def create_composition(self, img_comp_bg,
                        max_objs=15,
                        overlap_degree=0.2,
                        max_attempts_per_obj=10):
        """
        Creates a composition of multiple objects on a background.

        Args:
            img_comp_bg (numpy.ndarray): Background image.
            max_objs (int, optional): Maximum number of objects to add. Defaults to 15.
            overlap_degree (float, optional): Allowed overlap degree. Defaults to 0.2.
            max_attempts_per_obj (int, optional): Maximum attempts to place each object. Defaults to 10.

        Returns:
            tuple: Final composite image, composite mask, labels, and object areas.
        """

        img_comp = img_comp_bg.copy()
        h, w = img_comp.shape[0], img_comp.shape[1]
        mask_comp = np.zeros((h,w), dtype=np.uint8)
        
        obj_areas = []
        labels_comp = []
        num_objs = np.random.randint(max_objs) + 2
        
        i = 1
        
        for _ in range(1, num_objs):

            obj_idx = np.random.randint(len(self.obj_dict)) + 1
            
            for _ in range(max_attempts_per_obj):

                imgs_number = len(self.obj_dict[obj_idx]['images'])
                idx = np.random.randint(imgs_number)
                img_path = self.obj_dict[obj_idx]['images'][idx]
                mask_path = self.obj_dict[obj_idx]['masks'][idx]
                img, mask = self.get_im_and_mask(img_path, mask_path)

                x, y = np.random.randint(w), np.random.randint(h)
                longest_min = self.obj_dict[obj_idx]['longest_min']
                longest_max = self.obj_dict[obj_idx]['longest_max']
                img, mask = self.resize_transform_obj(img,
                                                mask,
                                                longest_min,
                                                longest_max,
                                                transforms=self.transforms_obj)

                if i == 1:
                    img_comp, mask_comp, mask_added = self.add_obj(img_comp,
                                                            mask_comp,
                                                            img,
                                                            mask,
                                                            x,
                                                            y,
                                                            i)
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(obj_idx)
                    i += 1
                    break
                else:        
                    img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                    img_comp, mask_comp, mask_added = self.add_obj(img_comp,
                                                            mask_comp,
                                                            img,
                                                            mask,
                                                            x,
                                                            y,
                                                            i)
                    ok = self.check_areas(mask_comp, obj_areas, overlap_degree)
                    if ok:
                        obj_areas.append(np.count_nonzero(mask_added))
                        labels_comp.append(obj_idx)
                        i += 1
                        break
                    else:
                        img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()        
            
        return img_comp, mask_comp, labels_comp, obj_areas
    
    def create_yolo_annotations(self, mask_comp, labels_comp):
        """
        Creates YOLO annotations for the objects in the composite image.

        Args:
            mask_comp (numpy.ndarray): Composite mask.
            labels_comp (list): List of labels for the objects.

        Returns:
            list: YOLO annotations for the objects.
        """
        comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]
        
        obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
        masks = mask_comp == obj_ids[:, None, None]

        annotations_yolo = []
        for i in range(len(labels_comp)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            annotations_yolo.append([labels_comp[i] - 1,
                                    round(xc/comp_w, 5),
                                    round(yc/comp_h, 5),
                                    round(w/comp_w, 5),
                                    round(h/comp_h, 5)])

        return annotations_yolo
    
    def generate_dataset(self, folder, split='train'):
        """
        Generates a synthetic dataset with images and YOLO annotations.

        Args:
            folder (str): Folder to save the dataset.
            split (str, optional): Dataset split ('train', 'test', etc.). Defaults to 'train'.

        Returns:
            None.
        """
        time_start = time.time()

        for j in tqdm(range(self.image_number)):
            img_comp_bg = self.create_bg_with_noise(max_objs_to_add=60)

            img_comp, mask_comp, labels_comp, _ = self.create_composition(img_comp_bg,
                                                                 max_objs=15,
                                                                 overlap_degree=0.2,
                                                                 max_attempts_per_obj=10)

            img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(folder, split, 'images/{}.jpg').format(j), img_comp)

            annotations_yolo = self.create_yolo_annotations(mask_comp, labels_comp)
            for i in range(len(annotations_yolo)):
                with open(os.path.join(folder, split, 'labels/{}.txt').format(j), "a") as f:
                    f.write(' '.join(str(el) for el in annotations_yolo[i]) + '\n')
                    
        time_end = time.time()
        time_total = round(time_end - time_start)
        time_per_img = round((time_end - time_start) / self.imgs_number, 1)
        
        print("Generation of {} synthetic images is completed. It took {} seconds, or {} seconds per image".format(self.imgs_number, time_total, time_per_img))
        print("Images are stored in '{}'".format(os.path.join(folder, split, 'images')))
        print("Annotations are stored in '{}'".format(os.path.join(folder, split, 'labels')))
