import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
from tqdm import tqdm
import csaps

p = Parameters()

## Training

def Testing():
    print('Testing')
    

    ## Get dataset
   
    print("Get dataset")
    loader = Generator()

 
    ## Get agent and model

    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(804, "tensor(0.5786)")
	


    ## testing

    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # check model with test data 
        for _, _, _, test_image in loader.Generate():
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0) 

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("/Users/minootaghavi/Desktop/GA/Capstone-Project-1/test/IMG_1398.mp4")
        writer = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, (1280,800)) 
        while(cap.isOpened()):
            ret, frame = cap.read()
            #torch.cuda.synchronize()
            prevTime = time.time()
            frame = cv2.resize(frame, (512,256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            ti[0] = cv2.resize(ti[0], (1280,800))
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            writer.write(ti[0])
        
            
        cap.release()
        cv2.destroyAllWindows()


    elif p.mode == 2: # check model with a picture
        test_image = cv2.imread("/Users/minootaghavi/Desktop/GA/tusimple-trained model/minoo/Deep Neural Networks/data/test_set/clips/0530/1492626047222176976_0/20.img")
        test_image = cv2.resize(test_image, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test(lane_agent, np.array([test_image])) 
        cv2.imwrite('/Users/minootaghavi/Desktop/GA/tusimple-trained model/minoo/Deep Neural Networks/save_test/image2_result.png',ti[0])
        cv2.imshow("test", ti[0])
        cv2.waitKey(0) 

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation(loader, lane_agent)

############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, index= -1, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    progressbar = tqdm(range(loader.size_test//4))
    for test_image, target_h, ratio_w, ratio_h, testset_index, gt in loader.Generate_Test():
        x, y, _ = test(lane_agent, test_image, thresh, index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)

        x_, y_ = fitting(x_, y_, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x_, y_, testset_index)


        progressbar.update(1)
    progressbar.close()
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)


############################################################################
## write result
############################################################################
def write_result_json(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        for i in x[index]:
            result_data[batch_idx]['lanes'].append(i)
            result_data[batch_idx]['run_time'] = 1
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

    result = lane_agent.predict_lanes_test(test_images)
    #torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y

if __name__ == '__main__':
    Testing()
