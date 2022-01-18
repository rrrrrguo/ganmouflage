import cv2
import os
import glob
import numpy as np
objs=sorted(os.listdir("test_mutiobj/"))
files=glob.glob(os.path.join("test_mutiobj",objs[0],"sample_0",'4views',"render*"))

images=sorted([x for x in files if 'mask' not in x])

n_images=len(images)
#print(images,masks)


for i in range(n_images):
    all_images=[]
    all_masks=[]
    for j in range(3):
        mask_path=glob.glob(os.path.join("test_mutiobj",objs[j],"sample_0",'4views',f"render_{i}*mask.png"))[0]
        image_path=mask_path.replace("_mask.png",".png")
        all_images.append(cv2.imread(image_path))
        all_masks.append(cv2.imread(mask_path))
    
    new_mask=np.max(np.stack(all_masks,0),0)
    base_image=all_images[0].copy()
    for i in range(1,3):
        base_image[all_masks[i]>0]=all_images[i][all_masks[i]>0]
    
    cv2.imwrite(f"test_multi/{image_path.split('/')[-1]}",base_image)
    cv2.imwrite(f"test_multi/{mask_path.split('/')[-1]}",new_mask)