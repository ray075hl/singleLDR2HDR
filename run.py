from src.hdr import *


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:  python run.py [image_path]")
        sys.exit()
    # Read image from file
    image = cv2.imread(sys.argv[1], -1)

    # Add hdr filter
    # True:  Using weighted fusion;  False: Averge fusion.
    HDR_Filer = FakeHDR(True) 
    output_image = HDR_Filer.process(image) 
    
    # Save and show the final result 
    cv2.imwrite('result.jpg', 255*output_image)
    Show_origin_and_output(image, output_image)

