# BoggleCV
Using OpenCV and Tensorflow to identify a Boggle board in an image and decode what letters are on the board.

## Try it!
BoggleCV is deployed on my online boggle site, [try it here](https://games.johanv.xyz/boggle?username=guest&page=upload)! You can take a picture of a boggle board with your phone, or upload an image.

## Demo Video
[bogglecv-demo.mp4](bogglecv-demo.mp4)

## Explanation
Here are the various stages of the computer vision and machine learning pipeline, with a sample image.

1. An HSV threshold is used to find all the blue pixels, and create a mask.
2. A contour finding algorithm from OpenCV finds the outline of large blue blob in the image.


| Image with Box Drawn | Mask to Find Blue Pixels |
| :--- | :--- |
| ![debug1](images/debug1.png) | ![debug2](images/debug2.png) |

3. The scipy signal processing library is used on the points in the outline to find corners
4. The box is cut out from the original image and warped to become a square.


| Corner Finding | Warped Image |
| :--- | :--- |
| ![plot1](images/plot1.png) | ![debug3](images/debug3.png) |

5. Next, the individual letters must be found. A sum of all the pixels in each row is taken, and the dips in brightness are found.
6. The same is repeated for the columns.
7. The positions of the rows and columns are used to cut out each letter from the image, and the letters are resized to 30x30 for easier use in machine learning.


| Row/Column Sum | Letters Cut Out |
| :--- | :--- |
| ![plot2](images/plot2.png) | ![plot3](images/plot3.png) |

8. This entire process is repeated over many images (and videos) to create a dataset of thousands of boggle letter images.
9. The images are labelled by hand. For videos, this is much easier since you only have to label it once for the whole video.
10. A Tensorflow model is trained on the dataset and exported to a file. It uses a convolutional neural network (CNN) which is optimal for images and other 2-dimensional data. After the CNN, a few 1-dimensional layers are used to get the output to an easier format.
```
#convolutional (2-dimensional) layers
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))

#data is flattened from 2 dimensions to 1
model.add(layers.Flatten())

#a fully-connected layer is added for more flexibility in combining features
model.add(layers.Dense(100, activation='relu'))

#the final layer has 1 output for each letter of the alphabet, and they are expressed as percentages that it thinks it might be of each letter. usually, one letter is close to 100% and the rest are close to 0%, which means it is very sure of its decision. But if it is less sure of a result, the number would be lower, e.g. 80%, and it might have 1 or 2 other options that were 15% and 5% for example. This information can be used, possibly to ask the user for a better quality image, or a confirmation that the letter is correct.
model.add(layers.Dense(26, activation="softmax"))
```


Note: steps 8-10 only have to be done one time to set up the pipeline.


11. Finally, the trained model is used to identify the letters in a new image. (See the output below.)


Right now, the model has about 88% accuracy on images it has never seen before. This is a pretty good start, but I am planning to improve it by deploying this to my site where people can play boggle, and create a system for uploading images and correcting any mistakes that the model makes. Then I will take this new data and re-train the model to achieve higher accuracy.

![output](images/output.png)
