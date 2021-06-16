import cv2
import numpy as np
import json
from data.config import cfg
import os
import math
import shutil
import json

def draw_text(img, point, text, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.5
    thickness = 5
    text_thickness = 1
    bg_color = (255, 0, 0)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 255, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, '%d' % (text), point, fontFace, 0.5, (255, 0, 0))
    return img
 
 
def draw_text_line(img, point, text_line: str, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.5
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img

def visualize(keypoint_num, img, joints, score=None):
        pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],[18,19]]
        # 鼻子1，左眼2，右眼3，左耳4，右耳5，左肩6，右肩7，左肘8，右肘9，左腕10，
        # 右腕11，左臀12，右臀13，左膝14，右膝15，左踝16，右踝17，杆头18，杆尾19

        color = np.random.randint(0, 256, (keypoint_num, 3)).tolist()
        joints_array = np.ones((keypoint_num, 2), dtype=np.float32)
        for i in range(keypoint_num):
            joints_array[i, 0] = joints[i * 3]
            joints_array[i, 1] = joints[i * 3 + 1]
            # joints_array[i, 2] = joints[i * 3 + 2]

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints_array[pair[0] - 1],
                      joints_array[pair[1] - 1])

        for i in range(keypoint_num):
            if joints_array[i, 0] > 0 and joints_array[i, 1] > 0:
                cv2.circle(img, tuple(
                    joints_array[i, :2]), 2, (0,255,0), 2)

        return img

def draw_angle(img,joints,angle_type):
    def draw_line(img, p1, p2, color=(0, 0, 255)):
        c = color
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv2.line(img, tuple(p1), tuple(p2), c, 2)
    if angle_type == "shoulder" or angle_type == "club":
        joints = np.array(joints).astype(np.int64)
        # 球杆和肩膀角度画法，就是多画一条水平线
        joints_hori = [joints[0,0]+100,joints[0,1]]
        draw_line(img,joints[0],joints_hori,(0,0,0))
        joints_hori = [joints[0,0]-100,joints[0,1]]
        draw_line(img,joints[0],joints_hori,(0,0,0))
        if angle_type == "shoulder":
            # 侧面肩膀的线太短了
            length = 100
            shoulder_len = ((joints[1,1]-joints[0,1])**2+(joints[1,0]-joints[0,0])**2)**0.5
            point1_x = length/shoulder_len*(joints[1,0]-joints[0,0])+joints[0,0]
            point1_y = length/shoulder_len*(joints[1,1]-joints[0,1])+joints[0,1]
            joints[1,0] = point1_x
            joints[1,1] = point1_y
        draw_line(img,joints[0],joints[1])
        cv2.circle(img, tuple(
                    joints[0, :2]), 2, (0,255,0), 2)
    if angle_type == "elbow":
        # 肘
        # 就是延长一下两条线的长度,注意传入的必需是肩膀，肘，手腕这样的顺序
        joints = np.array(joints)
        length = 100
        shoulder_elbow_len = ((joints[1,1]-joints[0,1])**2+(joints[1,0]-joints[0,0])**2)**0.5
        point1_x = length/shoulder_elbow_len*(joints[0,0]-joints[1,0])+joints[1,0]
        point1_y = length/shoulder_elbow_len*(joints[0,1]-joints[1,1])+joints[1,1]
        joints[0,0] = point1_x
        joints[0,1] = point1_y

        elbow_wrist_len = ((joints[1,1]-joints[2,1])**2+(joints[1,0]-joints[2,0])**2)**0.5
        point2_x = length/elbow_wrist_len*(joints[2,0]-joints[1,0])+joints[1,0]
        point2_y = length/elbow_wrist_len*(joints[2,1]-joints[1,1])+joints[1,1]
        joints[2,0] = point2_x
        joints[2,1] = point2_y
        joints = joints.astype(np.int64)
        
        draw_line(img,joints[0],joints[1])
        draw_line(img,joints[1],joints[2])
        cv2.circle(img, tuple(
                    joints[1, :2]), 2, (0,255,0), 2)
    if angle_type == "club_elbow":
        # 四个点求交点
        joints = np.array(joints)
        x0 = joints[0,0]
        y0 = joints[0,1]
        x1 = joints[1,0]
        y1 = joints[1,1]
        x2 = joints[2,0]
        y2 = joints[2,1]
        x3 = joints[3,0]
        y3 = joints[3,1]
        
        y = ((y0-y1)*(y3-y2)*x0+(y3-y2)*(x1-x0)*y0+(y1-y0)*(y3-y2)*x2+(x2-x3)*(y1-y0)*y2)/((x1-x0)*(y3-y2)+(y0-y1)*(x3-x2))
        x = x2+(x3-x2)*(y-y2)/(y3-y2);
        # 画球杆延长线
        length = 100
        point_tail_len = ((joints[2,1]-y)**2+(joints[2,0]-x)**2)**0.5
        point1_x = length/point_tail_len*(joints[2,0]-x)+x
        point1_y = length/point_tail_len*(joints[2,1]-y)+y
        line1 = [[point1_x,point1_y],[x,y]]
        line1 = np.array(line1).astype(np.int64)
        # 画小臂延长线
        point_arm_len = ((joints[1,1]-y)**2+(joints[1,0]-x)**2)**0.5
        point2_x = length/point_arm_len*(joints[1,0]-x)+x
        point2_y = length/point_arm_len*(joints[1,1]-y)+y
        line2 = [[point2_x,point2_y],[x,y]]
        line2 = np.array(line2).astype(np.int64)
        
        draw_line(img,line1[0],line1[1])
        draw_line(img,line2[0],line2[1])
        cv2.circle(img, tuple(
                    line1[1, :2]), 2, (0,255,0), 2)

    return img    

if __name__ == '__main__':
    for dir_name in os.listdir(cfg.TEST_RESULT_PATH):
        img_only_keypoints = cfg.TEST_IMAGE_ONLY_KEYPOINTS
        img_only_keypoints = os.path.join(img_only_keypoints,dir_name)
        img_with_keypoints = os.path.join(cfg.TEST_RESULT_WITH_KEYPONTS,dir_name)
        judge_dir = os.path.join(cfg.TEST_IMAGE_JUDGE,dir_name)
        angle_dir = os.path.join(cfg.TEST_IMAGE_JUDGE,dir_name)
        json_file_path = os.path.join(cfg.TEST_IMAGE_JUDGE,dir_name,"judge.json")
        json_map = {}
        if os.path.exists(judge_dir):
            shutil.rmtree(judge_dir)
        os.makedirs(judge_dir)
        if os.path.exists(img_only_keypoints):
            shutil.rmtree(img_only_keypoints)
        os.makedirs(img_only_keypoints)
        if os.path.exists(img_with_keypoints):
            shutil.rmtree(img_with_keypoints)
        os.makedirs(img_with_keypoints)
        for img_name in os.listdir(os.path.join(cfg.TEST_RESULT_PATH,dir_name)):
            img_idx = int(img_name.split('_')[0])
            just_angle_dir = os.path.join(angle_dir,"angle",str(img_idx))
            if os.path.exists(just_angle_dir):
                shutil.rmtree(just_angle_dir)
            os.makedirs(just_angle_dir)
            
            ori_img_name = img_name
            img_name = int(img_name.split('_')[1].split('.')[0])
            img_name = "{:0>4d}".format(img_name)+".jpg"
            img_name = dir_name + '/' + img_name
            club_keypoints = []
            all_keypoints = []
            with open(os.path.join(cfg.TEST_CLUB_KEYPOINTS_PATH,dir_name,"club_keypoints.json"),'r') as f:
                club_json_str = f.read()
                club_json_data = json.loads(club_json_str)
            with open(os.path.join(cfg.TEST_KEYPOINTS_PATH,"keypoints_result",dir_name,"results.json"),'r') as f:
                human_json_str = f.read()
                human_json_data = json.loads(human_json_str)
                
            for item in club_json_data:
                if item['image_id']== img_name:
                    for i in item['keypoints']:
                        club_keypoints.extend(i)
            for item in human_json_data:
                if item['image_id']== img_name:
                    all_keypoints = item['keypoints']
            all_keypoints.extend(club_keypoints)
            data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
            img = visualize(19,data_numpy,all_keypoints)
            cv2.imwrite(os.path.join(img_with_keypoints, ori_img_name), img)
            # 老王还要只有关键点的图片
            img_backgroud = visualize(19,np.zeros(img.shape,np.uint8),all_keypoints)
            cv2.imwrite(os.path.join(img_only_keypoints, ori_img_name), img_backgroud)
        # 评价功能
            # 由于关键点检测的受限，如果导致一些计算的异常，说明，关键点检测不正确，则不进行相应角度的计算，相应的角度值赋为err
            # 以图片左上角作为源点,x轴向右,y轴向左
            # 后肘弯角（大小臂形成的角度）
            delta = 10e-4

            left_elbow_x = all_keypoints[7*3]
            left_elbow_y = all_keypoints[7*3+1]
            left_shoulder_x = all_keypoints[5*3]
            left_shoulder_y = all_keypoints[5*3+1]
            left_wrist_x = all_keypoints[9*3]
            left_wrist_y = all_keypoints[9*3+1]
            
            right_elbow_x = all_keypoints[8*3]
            right_elbow_y = all_keypoints[8*3+1]
            right_shoulder_x = all_keypoints[6*3]
            right_shoulder_y = all_keypoints[6*3+1]
            right_wrist_x = all_keypoints[10*3]
            right_wrist_y = all_keypoints[10*3+1]

            left_elbow_shoulder_x = left_shoulder_x-left_elbow_x 
            left_elbow_shoulder_y = left_shoulder_y-left_elbow_y 
            left_elbow_wrist_x = left_wrist_x-left_elbow_x 
            left_elbow_wrist_y = left_wrist_y-left_elbow_y 

            right_elbow_shoulder_x = right_shoulder_x-right_elbow_x 
            right_elbow_shoulder_y = right_shoulder_y-right_elbow_y 
            right_elbow_wrist_x = right_wrist_x-right_elbow_x 
            right_elbow_wrist_y = right_wrist_y-right_elbow_y 


            left_elbow_shoulder_len = (left_elbow_shoulder_x**2+left_elbow_shoulder_y**2)**0.5
            left_elbow_wrist_len = (left_elbow_wrist_x**2+left_elbow_wrist_y**2)**0.5

            right_elbow_shoulder_len = (right_elbow_shoulder_x**2+right_elbow_shoulder_y**2)**0.5
            right_elbow_wrist_len = (right_elbow_wrist_x**2+right_elbow_wrist_y**2)**0.5

            left_top = left_elbow_shoulder_x*left_elbow_wrist_x+left_elbow_shoulder_y*left_elbow_wrist_y
            left_bottom = left_elbow_shoulder_len*left_elbow_wrist_len
            
            if abs(left_bottom) <= delta:
                left_angle = "err"                
            else:
                left_res = left_top/left_bottom
                if abs(left_res) > 1:
                    left_angle = "err"                
                else:    
                    left_angle = math.degrees(math.acos(left_res))
            
            right_top = right_elbow_shoulder_x*right_elbow_wrist_x+right_elbow_shoulder_y*right_elbow_wrist_y
            right_bottom = right_elbow_shoulder_len*right_elbow_wrist_len
            
            if abs(right_bottom) <= delta:
                right_angle = "err"                
            else:
                right_res = right_top/right_bottom
                if abs(right_res) > 1:
                    right_angle = "err"                
                else:    
                    right_angle = math.degrees(math.acos(right_res))
            

            # 双肩角度
            left_right_shoulder_x = left_shoulder_x - right_shoulder_x
            left_right_shoulder_y = left_shoulder_y - right_shoulder_y
            if abs(left_right_shoulder_x) < delta:
                left_right_shoulder_angle = "err"
            else:    
                shoulder_res = abs(left_right_shoulder_y/left_right_shoulder_x)
                left_right_shoulder_angle = math.degrees(math.atan(shoulder_res))
            
            # 球杆角度
            club_tail_x = all_keypoints[18*3]
            club_tail_y = all_keypoints[18*3+1]
            club_head_x = all_keypoints[17*3]
            club_head_y = all_keypoints[17*3+1]
            club_x = club_head_x - club_tail_x
            club_y = club_head_y - club_tail_y
            if abs(club_x) < delta:
                club_angle = "err"
            else:
                club_res = abs(club_y/club_x)
                club_len = (club_x**2+club_y**2)**0.5
                if club_len < 5.0:
                    # 球杆检测不准，如果球杆很短，说明检测错误，这个取值只是一个经验值
                    club_angle = "err"
                else:
                    club_angle = math.degrees(math.atan(club_res))

            # 球杆和手臂的角度
            # 球杆和左臂的角度
            if club_angle == "err":
                left_club_arm_angle = "err"
                right_club_arm_angle = "err"
            else:
                left_club_arm_top = club_x*(left_elbow_wrist_x)+club_y*(left_elbow_wrist_y)
                left_club_arm_bottom = club_len*left_elbow_wrist_len
                if abs(left_club_arm_bottom) < delta:
                    left_club_arm_angle = "err"
                else:     
                    left_club_arm_res = left_club_arm_top/left_club_arm_bottom
                    if abs(left_club_arm_res) > 1:
                        left_club_arm_angle = "err"
                    else:
                        left_club_arm_angle = 180-abs(math.degrees(math.acos(left_club_arm_res)))
                # 球杆和右臂的角度
                right_club_arm_top = club_x*right_elbow_wrist_x+club_y*right_elbow_wrist_y
                right_club_arm_bottom = club_len*right_elbow_wrist_len
                if abs(right_club_arm_bottom) < delta:
                    right_club_arm_angle = "err"
                else: 
                    right_club_arm_res = right_club_arm_top/right_club_arm_bottom
                    if abs(right_club_arm_res) > 1:
                        right_club_arm_res = "err"
                    else:
                        right_club_arm_angle = 180-abs(math.degrees(math.acos(right_club_arm_res)))
                
            
            out_text = "left_elbow_angle:{}\nright_elbow_angle:{}\nshoulder_angle:{}\nclub_angle:{}\nclub_left_arm_angle:{}\nclub_right_arm_angle:{}\n".format(\
                left_angle,right_angle,left_right_shoulder_angle,club_angle,left_club_arm_angle,right_club_arm_angle)
            print(os.path.join(judge_dir, ori_img_name))
            print(out_text)            
            draw_text_line(img, (5,5), out_text)
            cv2.imwrite(os.path.join(judge_dir, ori_img_name), img)
            json_map[img_name] = out_text
            # 还需要单独角度的，13*6
            # 双肩
            if left_right_shoulder_angle != "err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                shoulder_angle_img_name = os.path.join(just_angle_dir,"shoulder.jpg")
                shoulder_points = [[left_shoulder_x,left_shoulder_y],[right_shoulder_x,right_shoulder_y]]
                shoulder_img = draw_angle(data_numpy,shoulder_points,"shoulder")
                out_text = "andgle:{}".format(left_right_shoulder_angle)
                draw_text_line(shoulder_img, (5,5), out_text)
                cv2.imwrite(shoulder_angle_img_name, shoulder_img)
                
            # 球杆
            if club_angle !="err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                club_angle_img_name = os.path.join(just_angle_dir,"club.jpg")
                club_points = [[club_tail_x,club_tail_y],[club_head_x,club_head_y]]
                club_img = draw_angle(data_numpy,club_points,"club")
                out_text = "andgle:{}".format(club_angle)
                draw_text_line(club_img, (5,5), out_text)
                cv2.imwrite(club_angle_img_name, club_img)
            # 后肘
            if left_angle !="err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                left_angle_img_name = os.path.join(just_angle_dir,"left_elbow.jpg")
                left_elbow_points = [[left_shoulder_x,left_shoulder_y],[left_elbow_x,left_elbow_y],[left_wrist_x,left_wrist_y]]
                left_elbow_img = draw_angle(data_numpy,left_elbow_points,"elbow")
                out_text = "andgle:{}".format(left_angle)
                draw_text_line(left_elbow_img, (5,5), out_text)
                cv2.imwrite(left_angle_img_name, left_elbow_img)
            if right_angle !="err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                right_angle_img_name = os.path.join(just_angle_dir,"right_elbow.jpg")
                right_elbow_points = [[right_shoulder_x,right_shoulder_y],[right_elbow_x,right_elbow_y],[right_wrist_x,right_wrist_y]]
                right_elbow_img = draw_angle(data_numpy,right_elbow_points,"elbow")
                out_text = "andgle:{}".format(right_angle)
                draw_text_line(right_elbow_img, (5,5), out_text)
                cv2.imwrite(right_angle_img_name, right_elbow_img)
            # 球杆和手臂
            if left_club_arm_angle !="err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                left_club_arm_img_name = os.path.join(just_angle_dir,"left_club_arm.jpg")
                left_club_arm_points = [[left_elbow_x,left_elbow_y],[left_wrist_x,left_wrist_y],[club_tail_x,club_tail_y],[club_head_x,club_head_y]]
                left_club_arm_img = draw_angle(data_numpy,left_club_arm_points,"club_elbow")
                out_text = "andgle:{}".format(left_club_arm_angle)
                draw_text_line(left_club_arm_img, (5,5), out_text)
                cv2.imwrite(left_club_arm_img_name, left_club_arm_img)
            if right_club_arm_angle !="err":
                data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
                right_club_arm_img_name = os.path.join(just_angle_dir,"right_club_arm.jpg")
                right_club_arm_points = [[right_elbow_x,right_elbow_y],[right_wrist_x,right_wrist_y],[club_tail_x,club_tail_y],[club_head_x,club_head_y]]
                right_club_arm_img = draw_angle(data_numpy,right_club_arm_points,"club_elbow")
                out_text = "andgle:{}".format(right_club_arm_angle)
                draw_text_line(right_club_arm_img, (5,5), out_text)
                cv2.imwrite(right_club_arm_img_name, right_club_arm_img)
                

                
    with open(json_file_path,'w') as f:
        json.dump(json_map,f) 
        
        

            
                        
             