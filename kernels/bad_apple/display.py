import time
import os
import struct
from PIL import Image
from io import BytesIO
import subprocess
import numpy as np
import base64
import cv2

use_fpga = False
buffer_width = 16
buffer_depth = 1350
frame_width = 480
frame_height = 360
upscale = 3

# buffer_width = 16
# buffer_depth = 0x152
# frame_width = 240
# frame_height = 180
# upscale = 6

buffer_size = buffer_depth * buffer_width
truncate = 300 if use_fpga else 0

def follow(filename):
    frame_count = 0
    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.001)
                continue
            if truncate and (" 0151 " in line):
                frame_count += 1
                if frame_count == truncate:
                    with open(filename, "w") as f:
                        f.truncate(0)
                    frame_count = 0
            yield line

def process_frame(frame_data):
    bits = np.unpackbits(np.frombuffer(frame_data, dtype=np.uint8))
    bits = bits[:frame_width * frame_height]
    image_array = bits.reshape((frame_height, frame_width))
    # image_array = np.flipud(np.fliplr(image_array))
    image_array = (image_array * 255).astype(np.uint8)
    filtered_image_array = cv2.fastNlMeansDenoising(image_array, None, h=72, templateWindowSize=8, searchWindowSize=8)
    # filtered_image_array = cv2.GaussianBlur(image_array, (3, 3), 0)
    # filtered_image_array = image_array

    filtered_image_array = np.kron(filtered_image_array, np.ones((upscale, upscale), dtype=np.uint8))

    image = Image.fromarray(filtered_image_array, mode='L')

    return image

def display_image(img):
    with BytesIO() as output:
        # img = img.resize((frame_width * upscale, frame_height * upscale), Image.NEAREST)
        img.save(output, format='PNG')
        
        output.seek(0)
        # subprocess.run(["/home/eecs/yrh/.iterm2/imgcat"], input=output.read())
        image_data = output.getvalue()
        b64_image_data = base64.b64encode(image_data).decode('utf-8')

        print("\033]", end='')
        print(f"1337;File=inline=1", end='')
        print(f";size={len(image_data)}", end='')
        print(f";name={base64.b64encode('tmp.png'.encode()).decode('utf-8')}", end='')
        # print(f";width={frame_width * upscale * 4};height={frame_height * upscale * 4}", end='')
        print(f":{b64_image_data}", end='')
        print("\a", end='')
        print('\n')

def main():
    if not use_fpga:
        filename = "/scratch/yrh/chipyard/sims/vcs/output/chipyard.harness.TestHarness.RadianceClusterConfig/kernel.radiance.out"
    else:
        filename = "/scratch/yrh/chipyard/sims/firesim/sim/generated-src/xilinx_alveo_u250/xilinx_alveo_u250-firesim-FireSim-FireSimRadianceClusterSynConfig-WithPrintfSynthesis_BaseXilinxAlveoU250Config/synthesized-prints.out0"
    # frame_data = {}
    frame_data0 = bytearray(buffer_size)
    for line in follow(filename):
        if not "fb0" in line:
            continue
        tokens = line.split()
        if not len(tokens) == (5 if use_fpga else 3):
            continue
        offset, data = tokens[3 if use_fpga else 1:]
        offset0 = int(offset, 16)

        frame_data0[offset0 * buffer_width : (offset0 + 1) * buffer_width] = bytes.fromhex(data)[::-1]

        if offset0 == buffer_depth - 1:
            img = process_frame(frame_data0)
            display_image(img)
            frame_data0 = bytearray(buffer_size)

if __name__ == "__main__":
    main()

