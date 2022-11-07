# affine-SfM
### This project is implementation of "Shape and Motion from Image Streams under Orthography: a Factorization Method" by Tomasi and Kanade   

* An image sequence of 50 frames of a hotel is used to construct the shape and motion.  
* Features are detected in the first frame of the sequence.  
* These features are tracked throughout the sequence by implementing Lucas - Kanade optical flow (feature-tracking) algorithm.  
* This repository contains only the implementation recovering the structure and motion of the tracked features. The feature detecting and tracking implementation is done in another repository which can be found in my profile by the name '[feature-detection-tracking](https://github.com/Madhunc5229/feature-detection-tracking)'. 

## Input:  
Feature detection             |  Feature Tracking | 
:-------------------------:|:-------------------------:|
<img src="/data/feature_detection.png" width="275" alt="Alt text" title=""> | <img src="/data/Full_sequence.png" width="275" alt="Alt text" title=""> |

## Output:  
Shape recovered             |  
:-------------------------:|
<img src="/results/shape.png" width="550" alt="Alt text" title=""> |

Camera motion            |  
:-------------------------:|
<img src="/results/motion.png" width="550" alt="Alt text" title=""> |





