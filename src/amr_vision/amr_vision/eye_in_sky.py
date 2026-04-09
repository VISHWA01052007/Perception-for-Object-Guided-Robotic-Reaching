import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # New: To publish the 3D coordinates
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectLocalizer(Node):
    def __init__(self):
        super().__init__('object_localizer_node')
        
        # 1. Subscribers
        self.img_sub = self.create_subscription(Image, '/camera/top_down/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/top_down/depth/image_raw', self.depth_callback, 10)
        
        # 2. Publisher: Crucial for Project 5
        # This publishes the (X, Y, Z) so the robotic arm can reach it.
        self.pose_pub = self.create_publisher(Point, '/target_object_pose', 10)
        
        self.bridge = CvBridge()
        self.latest_depth_frame = None
        
        # Camera Intrinsics
        self.fx, self.fy = 381.0, 381.0
        self.cx, self.cy = 320.0, 240.0
        
        self.get_logger().info("🎯 Project 5: Eye-in-the-Sky Localizer Started!")

    def depth_callback(self, msg):
        self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg):
        if self.latest_depth_frame is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Thresholding for Red Target
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_output = cv_image.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > 200: # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                u, v = x + w//2, y + h//2 

                # 3D Transformation
                z_depth = self.latest_depth_frame[v, u]
                if np.isnan(z_depth) or np.isinf(z_depth): continue

                x_3d = (u - self.cx) * z_depth / self.fx
                y_3d = (v - self.cy) * z_depth / self.fy

                # --- PROJECT 5 REQUIREMENT: PUBLISH DATA ---
                point_msg = Point()
                point_msg.x = float(x_3d)
                point_msg.y = float(y_3d)
                point_msg.z = float(z_depth)
                self.pose_pub.publish(point_msg)
                # -------------------------------------------

                # Annotations
                cv2.rectangle(final_output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"X:{x_3d:.2f} Y:{y_3d:.2f} Z:{z_depth:.2f}m"
                cv2.putText(final_output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.imshow("Project 5: Live Localization", final_output)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()