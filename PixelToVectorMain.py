import os
import cv2
import numpy as np
import svgwrite
from glob import glob
from typing import List, Tuple, Dict

class PixelArtVectorizer:
    
    def __init__(self, approx_epsilon_factor: float = 0.002):

        self.epsilon_factor = approx_epsilon_factor

    def process_image(self, image_path: str, output_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Cant Read {image_path}")
            return

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        height, width = img.shape[:2]
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')
        
        pixels = img.reshape(-1, 4)
        unique_colors = np.unique(pixels, axis=0)
        

        for color in unique_colors:
            b, g, r, a = color
            if a == 0: continue 

            lower = np.array([b, g, r, a], dtype="uint8")
            upper = np.array([b, g, r, a], dtype="uint8")
            mask = cv2.inRange(img, lower, upper)


            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            if hierarchy is None: continue

            path_data = []

            for i, cnt in enumerate(contours):
                perimeter = cv2.arcLength(cnt, True)
                epsilon = self.epsilon_factor * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3:
                    points = cnt
                else:
                    points = approx

                pts = points.reshape(-1, 2)
                
                d = f"M {pts[0][0]},{pts[0][1]} "
                for p in pts[1:]:
                    d += f"L {p[0]},{p[1]} "
                d += "Z"
                path_data.append(d)

            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            opacity = a / 255.0
            

            dwg.add(dwg.path(d=" ".join(path_data),
                             fill=hex_color,
                             opacity=opacity,
                             stroke=hex_color,
                             stroke_width=0.6,
                             stroke_linejoin="miter",  
                             stroke_miterlimit=4,      
                             shape_rendering="geometricPrecision"))

        dwg.save()
        print(f"Converted: {image_path} -> {output_path}")

def main():
    input_dir = "sample"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory '{input_dir}'.")
        return

    vectorizer = PixelArtVectorizer(approx_epsilon_factor=0.002)
    
    png_files = glob(os.path.join(input_dir, "*.png"))
    
    if not png_files:
        print(f"No PNG files found.")
        return

    print(f"Found {len(png_files)} images. Processing with V3.0 Logic...")
    
    for png_path in png_files:
        filename = os.path.splitext(os.path.basename(png_path))[0]
        output_svg = os.path.join(input_dir, f"{filename}.svg")
        vectorizer.process_image(png_path, output_svg)
        
    print("Done. Check the edges!")

if __name__ == "__main__":
    main()