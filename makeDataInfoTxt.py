import os
import sys
if __name__=="__main__":
    folderPath=sys.argv[1]
    dataInfoTxtPath=os.path.join(sys.argv[2],"processedGolfVideos.txt")
    f=open(dataInfoTxtPath,'a')
    print(folderPath)
    print(dataInfoTxtPath)
    for videos in os.walk(folderPath):
        for videoName in videos[2]:
            f.write(str(videoName))
            f.write("\n")
    f.flush()
    f.close()