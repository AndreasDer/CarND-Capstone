from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass,fuel_capacity,brake_deadband,decel_limit,accel_limit,wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1,max_lat_accel,max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.
        mx = 0.2
        self.throttle_controller = PID(kp,ki,kd,mn,mx)
        
        tau = 0.5 # 1/(2pi*tau) = cutoff frquency
        ts = 0.2 # Sample time
        
        self.vel_low_pass = LowPassFilter(tau,ts)
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enable, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enable:
            self.throttle_controller.reset()
            return 0.,0.,0.
        
        current_vel = self.vel_low_pass.filt(current_vel)
        
        steering = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)
        #rospy.logwarn("Steering value: {0}".format(steering))
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error,sample_time)
        brake = 0
        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 # Using 700 istead of 400 to prevent Carla from moving
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error,self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius
            
            
        return throttle, brake, steering
