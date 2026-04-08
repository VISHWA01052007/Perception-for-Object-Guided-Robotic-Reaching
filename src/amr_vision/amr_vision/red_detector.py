import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedDetector(Node):
    def __init__(self):
        super().__init__('red_detector_node')
        
        # Subscriber 1: RGB Image (for color detection)
        self.img_sub = self.create_subscription(
            Image, 
            '/camera/top_down/image_raw', 
            self.image_callback, 
            10)
        
        # Subscriber 2: Depth Image (for real-time Z value)
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/top_down/depth/image_raw', 
            self.depth_callback, 
            10)
        
        self.bridge = CvBridge()
        self.latest_depth_frame = None
        
        # Camera Intrinsics (Based on 640x480 resolution and 1.4 Horizontal FOV)
        self.fx = 381.0  # Focal length x
        self.fy = 381.0  # Focal length y
        self.cx = 320.0  # Principal point x
        self.cy = 240.0  # Principal point y
        
        self.get_logger().info("🚀 3D Perception System Active: Analyzing RGB-D Stream...")

    def depth_callback(self, msg):
        # Store depth map as 32-bit float (values are in meters)
        self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg):
        if self.latest_depth_frame is None:
            self.get_logger().warn("Waiting for depth data...")
            return

        # 1. Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. Pipeline: Convert to HSV for robust color thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 3. Pipeline: Thresholding (Red Color Mask)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

        # 4. Pipeline: Contour Detection (Finding the object boundary)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Copy image for final annotation to keep the original feed clean
        final_output = cv_image.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                u, v = x + w//2, y + h//2  # Object center in pixels

                # --- 3D COORDINATE TRANSFORMATION LOGIC ---
                # A. Measure Z (Depth) from the sensor at the object's center pixel
                z_depth = self.latest_depth_frame[v, u]

                # Filter out invalid depth readings (NaN or Inf)
                if np.isnan(z_depth) or np.isinf(z_depth):
                    continue

                # B. Transform 2D pixels (u,v) to 3D Camera Coordinates (X,Y,Z)
                x_3d = (u - self.cx) * z_depth / self.fx
                y_3d = (v - self.cy) * z_depth / self.fy

                # 5. Annotation: Bounding Box and 3D Coordinates
                cv2.rectangle(final_output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"X:{x_3d:.2f} Y:{y_3d:.2f} Z:{z_depth:.2f}m"
                cv2.putText(final_output, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.get_logger().info(f"Target Detected: {label}")

        # 6. Requirement 3: Visualization of the Processing Pipeline
        cv2.imshow("1_Original_Feed", cv_image)
        cv2.imshow("2_Red_Mask", mask)
        cv2.imshow("3_Final_Detection", final_output)
        
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RedDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()