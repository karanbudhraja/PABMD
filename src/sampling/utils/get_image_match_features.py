from image_match.goldberg import ImageSignature
import sys
import numpy as np

# input file
imageName = sys.argv[1]

# generate signature
gis = ImageSignature()
features = gis.generate_signature(imageName)

# save signature
np.save("image_match_features", features)
