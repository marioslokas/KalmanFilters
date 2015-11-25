This is a small explanation for the unscented kalman filter code provided.

1) Theoretical stuff
Kalman filter is a versatile algorithm with many applications. The logic behind it is (roughly) it "learns" the system covariance
and produces estimates for each new state based on the previous. The Unsented Kalman Filter improves on the basic filter when your system is non-linear.
In order to make this clearer I will explain how i used Kalman Filters:

Let's say you have a system (a function) that each time is called provides you with a 3D position of an object (x, y, z). This object could very well be
the position of a thrown or kicked ball after launching. The problem here is that this function is not good. It will sometimes provide
wrong x,y,z of where the ball should have been. This is where the very convenient Kalman filter is used. Your second function, the Kalman Filter's Update,
will get fed with the x,y,z the first function outputs and provide an estimate of where the ball actually is. This estimate tends to be better in systems with 
a lot of noise.

A kalman filter needs to know some stuff about your system. According to physics, a ball's position like the one described depends on velocity and acceleration.
You need to make the filter understand the position-velocity-acceleration function by translating it into the transition matrix. The default transition matrix in
the code essentially means that each next position depends on the previous position plus the velocity, which is the standard function for an object moving with
no acceleration. When you translate the filter for your system you must adapt the system variables in the same way and somehow translate them to a transition matrix
in the same way.

2) Variables and tunables
This Unscented Kalman filter was adapted using logic from the EmguCV Kalman filter code (http://www.emgu.com/wiki/index.php/Kalman_Filter) and from Yi Cao's Matlab
implementation (http://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter/content/ukf.m). You need to consider the following:

-How many measurement variables you have
-How many prediction variables you have
-Process and measurement noise
-Variables c and lambda
-Transition matrix

Process and measurement noise variables are usually tunned according to the system noise, so this is mostly left to each one's judgement. Refer to the EmguCv Kalman filter code
for some more info on these. C and lambda are variables that I can' really explain, it is my understanding that they can have various effects to your filter output, but with
default values I could produce normal results so tuned them at your own risk.

3) Examples
Create an MyUnscentedKalman object and use it's update function.

MyUnscentedKalman unscentedKalmanFilter = new MyUnscentedKalman (6,3); // 3 measurements (xyz of object), 6 states (xyz position, xyz velocity)

Vector3 output = unscentedKalmanFilter.update (new Vector3 (1,1,1));

output = unscentedKalmanFilter.update (new Vector3 (2,2,2));

4) Notes for extensions and use
First of all you should modify the return type of the update function. See that I pass a Vector3 argument (from UnityEngine Vector3 class) and use the xyz parameters.
You can pass anything you want instead of that but note that you need to change all dimensions of matrices in the code to reflect the number of measurements and predictions you have.
(wherever you see 6 or 3). This of course will be written better whenever I have the time.
Additionally you need the Cholesky dll provided with the code, which is a simple C++ library providing Cholesky decomposition for the code. For now this code uses EmguCV
matrices, so include the standard EmguCV libraries in your code.


IMPORTANT NOTE: When I compared the outputs of this filter to EmguCV Kalman filter code I got exactly the same outputs with identical inputs for both filters. I understand
this might be because the transition matrix does not reflect a non-linear system. If you have a non-linear system you will probably get different results.

