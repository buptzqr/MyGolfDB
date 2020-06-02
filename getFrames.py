import os
import cv2
if __name__=="__main__":
    # use to get frames
    src_dir="/home/zqr/codes/GolfDB/data/videos_160"
    top_dst_dir="/home/zqr/codes/data/Videos2Frames_160"
    if not os.path.exists(top_dst_dir):
            os.mkdir(top_dst_dir)
    index=0
    fileNames = os.walk(src_dir)
    for filename in fileNames[2]:
        process_file_dir = filename.split(".")[0]
        dst_dir=os.path.join(top_dst_dir,str(process_file_dir))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        video = cv2.VideoCapture()
        if not video.open(os.path.join(src_dir,filename)):
            print("can not open the video")
            exit(1)
        pic_index = 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            save_path = "{}/{:>04d}.jpg".format(dst_dir, pic_index)
            cv2.imwrite(save_path, frame)
            pic_index += 1 
        index += 1
        video.release()
        print("Totally process {:d} video".format(index))