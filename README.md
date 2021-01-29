# Versions
## -- v1 --
This version used fancy OpenCV function to isolate the circles (balls) and only used one filter

accuracy: 95.4%

speed: 66.7 fps

ball accuracy: 64.1%

## -- v2 --
This version used 4 different filters (manually tuned) and essentially nothing else

accuracy: 93.0%

speed: 86.0 fps

ball accuracy: 56.4%

## -- v3 --
This version uses 4 filters all tuned by a neural network

accuracy: 96.3%

speed: 92.7 fps

ball accuracy: 74.8%

## -- v4 -- 
This version uses 4 filters all tuned by a neural network, but with a wider dataset. Even though it looks better on paper is actually may be worse than v3, this tends to also calls pieces of the wheel of fortune powercells.

accuracy: 95.0%

speed: 97.2 fps

ball accuracy: 78.0%

## -- v5 -- 
This is v4, but with a little manual tuning on the hue values

accuracy: 97.2%

speed: 97.2 fps

ball accuracy: 79.5%

## -- v6 --
Turns objects into convex hulls before trying to circle detect, and tuned the bounds from v5.
As an added note, I also fixed the SGD, there was a bug in the gradient descent with setting the old loss and there was a bug in the predictor, it wasn't setting the center of the circle correctly.

accuracy: 97.2%

speed: 92.1 fps

ball accuracy: 83.1%