import numpy as np
import cv2
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List
import threading

class CameraCalibration:
    """Camera calibration using checkerboard pattern"""
    
    def __init__(self, 
                 checkerboard_size=(6, 8),  # (columns, rows) of internal corners
                 square_size=1.0,  # Size of square in real units (cm)
                 num_images=15,
                 save_dir="camera_calibration"):
        
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.num_images = num_images
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Calibration data
        self.objpoints = []  # 3D points in real world
        self.imgpoints = []  # 2D points in image plane
        self.captured_images = []
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
        # State
        self.is_calibrating = False
        self.countdown_active = False
        self.current_count = 0
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    def start_calibration(self):
        """Start calibration process"""
        self.is_calibrating = True
        self.objpoints.clear()
        self.imgpoints.clear()
        self.captured_images.clear()
        print(f"\n{'='*60}")
        print(f"CAMERA CALIBRATION STARTED")
        print(f"{'='*60}")
        print(f"Target images: {self.num_images}")
        print(f"Checkerboard size: {self.checkerboard_size[0]}x{self.checkerboard_size[1]}")
        print(f"Ready to capture!")
        print(f"{'='*60}\n")
    
    def capture_image(self, frame: np.ndarray) -> bool:
        """
        Capture and process calibration image
        
        Args:
            frame: Current camera frame (BGR)
            
        Returns:
            bool: True if checkerboard found and captured
        """
        if not self.is_calibrating:
            return False
        
        if len(self.captured_images) >= self.num_images:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size, 
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(
                gray, 
                corners, 
                (11, 11), 
                (-1, -1), 
                self.criteria
            )
            
            # Store points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            self.captured_images.append(frame.copy())
            
            print(f"✅ Image {len(self.captured_images)}/{self.num_images} captured!")
            
            # Save image with detected corners
            img_with_corners = frame.copy()
            cv2.drawChessboardCorners(
                img_with_corners, 
                self.checkerboard_size, 
                corners_refined, 
                ret
            )
            img_path = self.save_dir / f"calib_{len(self.captured_images):02d}.jpg"
            cv2.imwrite(str(img_path), img_with_corners)
            
            return True
        else:
            print("❌ Checkerboard not found!")
            return False
    
    def calibrate(self, image_shape: Tuple[int, int]) -> bool:
        """
        Perform camera calibration
        
        Args:
            image_shape: (width, height) of images
            
        Returns:
            bool: True if calibration successful
        """
        if len(self.objpoints) < 3:
            print(f"❌ Not enough images for calibration (need at least 3, got {len(self.objpoints)})")
            return False
        
        print(f"\n{'='*60}")
        print(f"CALIBRATING CAMERA...")
        print(f"{'='*60}")
        print(f"Using {len(self.objpoints)} images")
        
        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            image_shape,
            None,
            None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], 
                    rvecs[i], 
                    tvecs[i], 
                    mtx, 
                    dist
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            self.calibration_error = total_error / len(self.objpoints)
            
            print(f"✅ Calibration successful!")
            print(f"Reprojection error: {self.calibration_error:.4f} pixels")
            print(f"{'='*60}\n")
            
            # Save results
            self.save_calibration()
            
            return True
        else:
            print("❌ Calibration failed!")
            return False
    
    def save_calibration(self):
        """Save calibration results to file"""
        if self.camera_matrix is None:
            return
        
        # Save as numpy arrays
        np.savez(
            str(self.save_dir / "camera_calibration.npz"),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            calibration_error=self.calibration_error
        )
        
        # Save as text file (human readable)
        txt_path = self.save_dir / "camera_calibration.txt"
        with open(txt_path, 'w') as f:
            f.write("Camera Calibration Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Camera Matrix (Intrinsic Parameters):\n")
            f.write(str(self.camera_matrix) + "\n\n")
            
            f.write("Distortion Coefficients:\n")
            f.write(str(self.dist_coeffs) + "\n\n")
            
            f.write(f"Reprojection Error: {self.calibration_error:.4f} pixels\n\n")
            
            f.write("Focal Length:\n")
            f.write(f"  fx = {self.camera_matrix[0, 0]:.2f}\n")
            f.write(f"  fy = {self.camera_matrix[1, 1]:.2f}\n\n")
            
            f.write("Principal Point:\n")
            f.write(f"  cx = {self.camera_matrix[0, 2]:.2f}\n")
            f.write(f"  cy = {self.camera_matrix[1, 2]:.2f}\n")
        
        print(f"💾 Calibration saved to: {self.save_dir}")
    
    def load_calibration(self, filepath: Optional[str] = None) -> bool:
        """
        Load calibration from file
        
        Args:
            filepath: Path to .npz file (if None, uses default)
            
        Returns:
            bool: True if loaded successfully
        """
        if filepath is None:
            filepath = self.save_dir / "camera_calibration.npz"
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ Calibration file not found: {filepath}")
            return False
        
        try:
            data = np.load(str(filepath))
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibration_error = float(data['calibration_error'])
            
            print(f"✅ Calibration loaded from: {filepath}")
            print(f"Reprojection error: {self.calibration_error:.4f} pixels")
            return True
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get current calibration status"""
        return {
            'is_calibrating': self.is_calibrating,
            'images_captured': len(self.captured_images),
            'target_images': self.num_images,
            'is_calibrated': self.camera_matrix is not None,
            'calibration_error': self.calibration_error
        }
    
    def reset(self):
        """Reset calibration"""
        self.is_calibrating = False
        self.objpoints.clear()
        self.imgpoints.clear()
        self.captured_images.clear()
        print("🔄 Calibration reset")