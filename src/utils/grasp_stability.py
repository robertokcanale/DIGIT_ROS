import torch, time, cv2, os
import numpy as np
from models import *
from digit_interface import Digit
from utils.contact_area_functions import *
from utils.sensor_functions import *
import torch.backends.cudnn as cudnn
from MessagePopup import Popup

# Device Used
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraspStability:

    def __init__(self, digit_sn1, digit_sn2, lstm, weights):
            self.last_active = time.time()
            self.SENSOR_SERIAL_NUMBER_LIST = [digit_sn1 if isinstance(digit_sn1, str) else str(digit_sn1), digit_sn2 if isinstance(digit_sn2, str) else str(digit_sn2)]
            self.SENSOR_NAME_LIST = ["left", "right"]
            self.STAGE = ["grasp", "lift"]
            self.FPS_CONFIG = Digit.STREAMS["QVGA"]["fps"]["60fps"]
            self.DIGIT_INTENSITY = 10
            self.fps = 5
            self.numImg = 8
            self.weights = os.getcwd() + weights
            self.running = True
            print(os.getcwd())
            if lstm == 1:
                self.net = self.build_model_lstm()
            else:
                self.net, self.featureNet = self.build_model()

            #DIGIT
            self.sensors = {}
            self.sensors = initialize_sensors(self.sensors, self.SENSOR_SERIAL_NUMBER_LIST, self.SENSOR_NAME_LIST)

            #Making set of images into a baseline`
            for sn, sensor in self.sensors.items():
                sensor = compute_baseline(sensor)
                sensor["data_sequence"] = []
                sensor["diff_sequence"] = []
            print("==> Fnished initialization")

    def build_model(self):
        print('==> Building model..')
        state = torch.load(self.weights)
        net = DigitResNet(num_classes=32)
        net = net.to(device)
        
        featureNet = create_FeatureCombinationNet(input_feature_size=512, num_classes=2)
        featureNet = featureNet.to(device)
        
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            featureNet = torch.nn.DataParallel(featureNet)
            cudnn.benchmark = True
            
            net.load_state_dict(state["net"])
            featureNet.load_state_dict(state["featurenet"])
        print('==> Model Built Correctly')
        return net, featureNet

    def build_model_lstm(self):
        print('==> Building model..')
        state = torch.load(self.weights)        
        net = create_resnet18rnn(num_classes=2)
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            
            net.load_state_dict(state["net"])
        print('==> Model Built Correctly')
        return net

    def run(self):
        for sn, sensor in self.sensors.items():
            sensor = recompute_baseline(sensor)
            sensor["baseline"] = cv2.cvtColor(sensor["baseline"], cv2.COLOR_BGR2LAB)
            _,_,base_b = cv2.split(sensor["baseline"])
            sensor["baseline"] = base_b
        while self.running:                    
            start_time = time.time()
            while time.time() - start_time < 2:
                #Write the original images
                timestamp = time.time() * 1000.

                #Write the difference images
                for sn, sensor in self.sensors.items():
                    sensor["current_frame"] = cv2.GaussianBlur(sensor["object"].get_frame(),(11,11),5)
                    
                    current_frame = cv2.cvtColor(sensor["current_frame"].copy(), cv2.COLOR_BGR2LAB)
                    _,_,curr_b = cv2.split(current_frame)
                    
                    image_diff, _ = contact_area(target=curr_b.copy(),base=sensor["baseline"])
                    
                    image_b = cv2.resize(curr_b, (0,0), fx=0.25, fy=0.25)
                    image_diff = cv2.resize(image_diff, (0,0), fx=0.25, fy=0.25)
                    
                    image_b = np.stack([image_b, image_b, image_b], axis=2)
                    image_diff = np.stack([image_diff, image_diff, image_diff], axis=2)
                    sensor["diff_sequence"].append((image_diff, timestamp))
                    sensor["data_sequence"].append((image_b, timestamp))

            image_vis_list = []
            for sn, sensor in self.sensors.items():
                sensor["data_sequence"] = sorted(sensor["data_sequence"], key=lambda x: x[1], reverse=True)
                sensor["diff_sequence"] = sorted(sensor["diff_sequence"], key=lambda x: x[1], reverse=True)
                rgb_imgs_list = []
                diff_imgs_list = []
                prev_timestamp = None
                #Extracting the images from Last to First
                for (image_rgb, ts1), (image_diff_lab, ts2) in zip(sensor["data_sequence"], sensor["diff_sequence"]):
                    timestamp = ts1
                    #Sanity Check for the path
                    if prev_timestamp == None:
                        prev_timestamp = timestamp
                        
                        cv2.imwrite("image_rgb_"+str(ts1)+".png", image_rgb)
                        cv2.imwrite("image_diff_lab_"+str(ts1)+".png", image_diff_lab)
                        rgb_imgs_list.append(image_rgb)
                        diff_imgs_list.append(image_diff_lab)
                    else:
                        #Convert Time to Seconds
                        time_diff = (prev_timestamp - timestamp) / 1000.
                        if time_diff >= (1. / self.fps):
                            prev_timestamp = timestamp
                            
                            cv2.imwrite("image_rgb_"+str(ts1)+".png", image_rgb)
                            cv2.imwrite("image_diff_lab_"+str(ts1)+".png", image_diff_lab)
                            
                            rgb_imgs_list.append(image_rgb)
                            diff_imgs_list.append(image_diff_lab)
                #Sanity Check to have the same number of images
                while len(rgb_imgs_list) > self.numImg:
                    del rgb_imgs_list[-1]
                    del diff_imgs_list[-1]
                    
                diff_imgs_list_vis = cv2.hconcat(diff_imgs_list)
                
                diff_imgs_list_vis = diff_imgs_list_vis * 8 * 255
                diff_imgs_list_vis = np.clip(diff_imgs_list_vis, 0., 255.)

                image_vis_list.append(cv2.vconcat([cv2.hconcat(rgb_imgs_list), diff_imgs_list_vis.astype(np.uint8)]))

                sensor["processed_rgb_list"] = rgb_imgs_list
                sensor["processed_diff_list"] = diff_imgs_list

            self.image_vis = cv2.vconcat(image_vis_list)
            #TODO process the data for inferencing here WILL NEED TO UPDATE SOON
            filtered_rgb_imgs_LEFT = self.sensors["left"]["processed_rgb_list"]
            filtered_diff_imgs_LEFT = self.sensors["left"]["processed_diff_list"]
            filtered_rgb_imgs_RIGHT = self.sensors["right"]["processed_rgb_list"]
            filtered_diff_imgs_RIGHT = self.sensors["right"]["processed_diff_list"]
            with torch.no_grad():
                diff_features_LEFT_list = []
                diff_features_RIGHT_list = []
                for rgb_img_LEFT, diff_img_LEFT, rgb_img_RIGHT, diff_img_RIGHT in zip(filtered_rgb_imgs_LEFT, filtered_diff_imgs_LEFT, filtered_rgb_imgs_RIGHT, filtered_diff_imgs_RIGHT):
                    rgb_img_LEFT, diff_img_LEFT, rgb_img_RIGHT, diff_img_RIGHT = torch.tensor(rgb_img_LEFT).float().to(device), torch.tensor(diff_img_LEFT).float().to(device), torch.tensor(rgb_img_RIGHT).float().to(device), torch.tensor(diff_img_RIGHT).float().to(device)
                    rgb_img_LEFT, diff_img_LEFT, rgb_img_RIGHT, diff_img_RIGHT = torch.unsqueeze(rgb_img_LEFT, 0), torch.unsqueeze(diff_img_LEFT, 0), torch.unsqueeze(rgb_img_RIGHT, 0), torch.unsqueeze(diff_img_RIGHT, 0)
                    rgb_img_LEFT, diff_img_LEFT, rgb_img_RIGHT, diff_img_RIGHT = torch.permute(rgb_img_LEFT, (0, 3, 1, 2)), torch.permute(diff_img_LEFT, (0, 3, 1, 2)), torch.permute(rgb_img_RIGHT, (0, 3, 1, 2)), torch.permute(diff_img_RIGHT, (0, 3, 1, 2))

                    features_diff_LEFT = self.net(diff_img_LEFT.contiguous())
                    features_diff_RIGHT = self.net(diff_img_RIGHT.contiguous())
                    
                    # features_rgb (Batch_size, features_size)
                    diff_features_LEFT_list.append(features_diff_LEFT)
                    diff_features_RIGHT_list.append(features_diff_RIGHT)

                diff_features_LEFT = torch.cat(diff_features_LEFT_list, dim=1)
                diff_features_RIGHT = torch.cat(diff_features_RIGHT_list, dim=1)
                final_features = torch.cat([diff_features_LEFT, diff_features_RIGHT], dim=1)
                outputs = self.featureNet(final_features)
                outputs = outputs.cpu().numpy()
                outputs = np.squeeze(outputs, axis=0)
                pred_class = np.argmax(outputs)
                print(outputs)
                print(pred_class)


            #DUMMY DATA
            stability = pred_class
            message = "({:.1f}, 0)".format(stability)
            print("==> Inference Result...")
            print(message)
            Popup(pred_class)
            return(pred_class)


    def run_lstm(self):
        for sn, sensor in self.sensors.items():
            sensor = recompute_baseline(sensor)
            sensor["baseline"] = cv2.cvtColor(sensor["baseline"], cv2.COLOR_BGR2LAB)
            _,_,base_b = cv2.split(sensor["baseline"])
            sensor["baseline"] = base_b
        while self.running:
            self.last_active = time.time()
            start_time = time.time()
            while time.time() - start_time < 2:
                #Write the original images
                timestamp = time.time() * 1000.
                for sn, sensor in self.sensors.items():
                    sensor["current_frame"] = cv2.GaussianBlur(sensor["object"].get_frame(),(11,11),0)
                    sensor["baseline"] = cv2.cvtColor(sensor["baseline"], cv2.COLOR_BGR2LAB)
                    _,_,base_b = cv2.split(sensor["baseline"])
                    image_baseline = cv2.resize(base_b, (0,0), fx=0.25, fy=0.25)
                    image_baseline_normalized = (image_baseline - image_baseline.mean()) / image_baseline.std()
                    sensor["baseline_norm"] = image_baseline_normalized
                    
                    current_frame = cv2.cvtColor(sensor["current_frame"].copy(), cv2.COLOR_BGR2LAB)
                    _,_,curr_b = cv2.split(current_frame)
                    image_rgb = cv2.resize(curr_b, (0,0), fx=0.25, fy=0.25)
                    image_rgb_normalized = (image_rgb - image_rgb.mean()) / image_rgb.std()
                    image_diff, _ = contact_area(target=image_rgb_normalized.copy(),base=image_baseline_normalized)
                    image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=2)
                    image_diff = np.stack([image_diff, image_diff, image_diff], axis=2)
                    sensor["diff_sequence"].append((image_diff, timestamp))
                    sensor["data_sequence"].append((image_rgb, timestamp))

            image_vis_list = []
            for sn, sensor in self.sensors.items():
                sensor["data_sequence"] = sorted(sensor["data_sequence"], key=lambda x: x[1], reverse=True)
                sensor["diff_sequence"] = sorted(sensor["diff_sequence"], key=lambda x: x[1], reverse=True)
                rgb_imgs_list = []
                diff_imgs_list = []
                prev_timestamp = None
                #Extracting the images from Last to First
                # image_baseline = sensor["baseline"]
                image_baseline = sensor["baseline_norm"]
                for (image_rgb, ts1), (image_diff, ts2) in zip(sensor["data_sequence"], sensor["diff_sequence"]):
                    timestamp = ts1
                    #Sanity Check for the path
                    if prev_timestamp == None:
                        prev_timestamp = timestamp
                        
                        rgb_imgs_list.append(image_rgb)
                        diff_imgs_list.append(image_diff)
                    else:
                        #Convert Time to Seconds
                        time_diff = (prev_timestamp - timestamp) / 1000.
                        if time_diff >= (1. / self.fps):
                            prev_timestamp = timestamp
                            
                            rgb_imgs_list.append(image_rgb)
                            diff_imgs_list.append(image_diff)
                #Sanity Check to have the same number of images
                while len(rgb_imgs_list) > self.numImg:
                    del rgb_imgs_list[-1]
                    del diff_imgs_list[-1]

                diff_imgs_list_vis = cv2.hconcat(diff_imgs_list)
                diff_imgs_list_vis = (diff_imgs_list_vis - diff_imgs_list_vis.min()) / (diff_imgs_list_vis.max() - diff_imgs_list_vis.min()) * 255

                image_vis_list.append(cv2.vconcat([cv2.hconcat(rgb_imgs_list), diff_imgs_list_vis.astype(np.uint8)]))

                sensor["processed_rgb_list"] = rgb_imgs_list
                sensor["processed_diff_list"] = diff_imgs_list
                sensor["data_sequence"] = []
                sensor["diff_sequence"] = []

            self.image_vis = cv2.vconcat(image_vis_list)
            # TODO process the data for inferencing here WILL NEED TO UPDATE SOON
            filtered_rgb_imgs_LEFT = self.sensors["left"]["processed_rgb_list"]
            filtered_diff_imgs_LEFT = self.sensors["left"]["processed_diff_list"]
            filtered_rgb_imgs_RIGHT = self.sensors["right"]["processed_rgb_list"]
            filtered_diff_imgs_RIGHT = self.sensors["right"]["processed_diff_list"]
            filtered_diff_imgs_LEFT = np.array([filtered_diff_imgs_LEFT])
            filtered_diff_imgs_RIGHT = np.array([filtered_diff_imgs_RIGHT])
            filtered_diff_imgs_LEFT = np.transpose(filtered_diff_imgs_LEFT, (0, 1, 4, 2, 3))
            filtered_diff_imgs_RIGHT = np.transpose(filtered_diff_imgs_RIGHT, (0, 1, 4, 2, 3))
            filtered_diff_imgs_LEFT = torch.tensor(filtered_diff_imgs_LEFT)
            filtered_diff_imgs_RIGHT = torch.tensor(filtered_diff_imgs_RIGHT)
            with torch.no_grad():
                filtered_diff_imgs_LEFT, filtered_diff_imgs_RIGHT = filtered_diff_imgs_LEFT.float().to(device), filtered_diff_imgs_RIGHT.float().to(device)
                outputs = self.net(filtered_diff_imgs_LEFT, filtered_diff_imgs_RIGHT)

                outputs = outputs.cpu().numpy()
                outputs = np.squeeze(outputs, axis=0)
                pred_class = np.argmax(outputs)
                print(outputs)
                print(pred_class)
            
            #DUMMY DATA
            stability = pred_class
            message = "({:.1f}, 0)".format(stability)
            print("==> Inference Result...")
            print(message)
            Popup(pred_class)
            return(pred_class)




