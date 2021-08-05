import tensorflow as tf
import argparse
import os


parser = argparse.ArgumentParser(
    usage="%(prog)s [OPTION] -I Input file",
    description="Sobel Edge Detection",
)
parser.add_argument(
    "-I", help="Input image", type=str
)

parser.add_argument(
    "-o", help="Output filename", type=str, default=None
)

args = parser.parse_args()

img = tf.io.read_file(args.I)
img = tf.image.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)

x, y, c = tf.shape(img)

img = tf.transpose(img, perm=(2, 0, 1))
img = tf.reshape(img, (c, x * y))
img = tf.transpose(img)

img -= tf.reduce_mean(img, axis=0)

_, eigen_vectors = tf.linalg.eigh(tf.matmul(img, img, transpose_a=True))

img = tf.matmul(eigen_vectors, img, transpose_a=True, transpose_b=True)

img = tf.reshape(img, (c, x, y))

img = tf.transpose(img, perm=(1, 2, 0))

if args.o is None:
    outfile = ".".join(os.path.basename(args.I).split(".")[:-1]) + ".jpg"
else:
    outfile = args.o

print(f'writing file {outfile}')

img = tf.image.convert_image_dtype(img, tf.uint8)

tf.io.write_file(outfile, tf.image.encode_jpeg( img ))
