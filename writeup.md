# Advanced Lane Finding Project
- Prabath Peiris
- peiris.prabath@gmail.com
- Self-Driving Car NanoDegree
- Term 1
- Udacity

# Camera Calibration

## Distortion

When a camera takes a 3D objects from the real world and transform in to 2D images, there are some distortion get introduce to the image. This is a form of optical aberration. Distortion can be irregular and can follow many patterns. The most commonly enounctred distortions are mostly radiealy symmetric due to smmetry of photographics lenses. These Radial distortions can use classified into three main categories.


- **Barrel Distortion** - image magnification decreases with distance from the optical axis.
![image1](output_images/barrel_distortion.jpg "Undistorted")
- **Pincushion distortion** - Image magnification increases with the distance from the optical axis
![image1](output_images/pincushion_distortion.jpg "Undistorted")
- **Mustache distortion** - Stars our as a barrel distortion close to the image center and gradually turns in to pincushion distortion.
![image1](output_images/mustache_distortion.jpg "Undistorted")


