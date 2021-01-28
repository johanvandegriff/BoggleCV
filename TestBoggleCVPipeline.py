#!/usr/bin/python3
import time
times = []
times.append(time.time())
import BoggleCVPipeline
times.append(time.time())

# letters5x5gridLabelsStr = "IZEIMFLTYTOSEINHETNORRISU" #00162
# letters5x5gridLabelsStr = "DONLIEEIIESAPYWTAAKLTINRE" #00164
# letters5x5gridLabelsStr = "RAOCODSEERGAPEWORXRHELWNT" #00165
# letters5x5gridLabelsStr = "RONLICTSSDNMPPNUAEQIHINRM" #00170
letters5x5gridLabelsStr = "VANUIRAUHESKEITTDPRCGOUCA" #00173, 00171, 00160, 00159
# letters5x5gridLabelsStr = "IXESMFLEYTOSEENOETNRRRIWM" #00172, 00161

image_dir = '/home/johanv/nextcloud/projects/boggle2.0/cascademan/categories/5x5/images/'

print()
print()
#process 2 images
for image_file in ("00173.jpg", "00171.jpg", "00160.jpg", "00159.jpg", "00176.jpg", "not a file"):
    print("]===[", image_file, "]===[")
    try:
        lettersGuessed, confidence = BoggleCVPipeline.processImage(image_dir + image_file)

        # letters5x5gridLabels = [class_names.index(letter) for letter in letters5x5gridLabelsStr]
        right = "".join([str(int(a == b)) for a,b in zip(lettersGuessed, letters5x5gridLabelsStr)])

        # confidence_right = []
        # confidence_wrong = []

        # for i, c in enumerate(confidence):
        #     if right[i] == "1":
        #         confidence_right.append(c)
        #     else:
        #         confidence_wrong.append(c)

        print("guess: " + lettersGuessed)
        print("real:  " + letters5x5gridLabelsStr)
        print("right: " + right)
        # print("confidence:", confidence)
        # print("confidence_right:", confidence_right)
        # print("confidence_wrong:", confidence_wrong)
    except BoggleCVPipeline.BoggleError:
        print("failed to find boggle board in image")
    except Exception:
        print("unknown error")
    print()
    print()
    print()
    times.append(time.time())

time_diff = [] 
for x, y in zip(times[0::], times[1::]): 
    time_diff.append(y-x) 
      
print("time_diff:", time_diff)
