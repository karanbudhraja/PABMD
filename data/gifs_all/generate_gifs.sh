# convert demonstrations
convert -delay 50 -loop 0 ../demonstrations_all/flocking/simulated/1/*.png demonstration_1.gif
convert -delay 50 -loop 0 ../demonstrations_all/flocking/simulated/2/*.png demonstration_2.gif
convert -delay 50 -loop 0 ../demonstrations_all/flocking/simulated/3/*.png demonstration_3.gif

convert -delay 50 -loop 0 ../demonstrations_/flocking/1/*.png demonstration__1.gif
convert -delay 50 -loop 0 ../demonstrations_/flocking/2/*.png demonstration__2.gif
convert -delay 50 -loop 0 ../demonstrations_/flocking/3/*.png demonstration__3.gif

# convert rotated images
convert -rotate 90 demonstration_1.gif demonstration_1_rotation.gif

# convert translated images
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_200.png -background black -flatten 1108_200_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_201.png -background black -flatten 1108_201_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_202.png -background black -flatten 1108_202_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_203.png -background black -flatten 1108_203_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_204.png -background black -flatten 1108_204_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_205.png -background black -flatten 1108_205_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_206.png -background black -flatten 1108_206_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_207.png -background black -flatten 1108_207_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_208.png -background black -flatten 1108_208_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_209.png -background black -flatten 1108_209_translation.png
convert -page +100+0 ../demonstrations_all/flocking/simulated/1/1108_210.png -background black -flatten 1108_210_translation.png
convert +repage 1108_200_translation.png 1108_200_translation.png
convert +repage 1108_201_translation.png 1108_201_translation.png
convert +repage 1108_202_translation.png 1108_202_translation.png
convert +repage 1108_203_translation.png 1108_203_translation.png
convert +repage 1108_204_translation.png 1108_204_translation.png
convert +repage 1108_205_translation.png 1108_205_translation.png
convert +repage 1108_206_translation.png 1108_206_translation.png
convert +repage 1108_207_translation.png 1108_207_translation.png
convert +repage 1108_208_translation.png 1108_208_translation.png
convert +repage 1108_209_translation.png 1108_209_translation.png
convert +repage 1108_210_translation.png 1108_210_translation.png
convert -delay 50 -loop 0 *_translation.png demonstration_1_translation.gif
rm *_translation.png

# convert point selection methods
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_error_weighted/flocking/1/*.png point\ selection/error_weighted_1.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_error_weighted/flocking/2/*.png point\ selection/error_weighted_2.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_error_weighted/flocking/3/*.png point\ selection/error_weighted_3.gif

convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_kmeans/flocking/1/*.png point\ selection/kmeans_1.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_kmeans/flocking/2/*.png point\ selection/kmeans_2.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_kmeans/flocking/3/*.png point\ selection/kmeans_3.gif

convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_mean/flocking/1/*.png point\ selection/mean_1.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_mean/flocking/2/*.png point\ selection/mean_2.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_mean/flocking/3/*.png point\ selection/mean_3.gif

convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_min_error/flocking/1/*.png point\ selection/min_error_1.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_min_error/flocking/2/*.png point\ selection/min_error_2.gif
convert -delay 50 -loop 0 ../predictions_all/point\ selection/predictions_min_error/flocking/3/*.png point\ selection/min_error_3.gif

# convert feature construction methods
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_contour/flocking/1/*.png feature\ construction/contour_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_contour/flocking/2/*.png feature\ construction/contour_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_contour/flocking/3/*.png feature\ construction/contour_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_mnist/flocking/1/*.png feature\ construction/mnist_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_mnist/flocking/2/*.png feature\ construction/mnist_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_mnist/flocking/3/*.png feature\ construction/mnist_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vae/flocking/1/*.png feature\ construction/vae_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vae/flocking/2/*.png feature\ construction/vae_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vae/flocking/3/*.png feature\ construction/vae_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vgg16/flocking/1/*.png feature\ construction/vgg16_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vgg16/flocking/2/*.png feature\ construction/vgg16_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_vgg16/flocking/3/*.png feature\ construction/vgg16_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_resnet50/flocking/1/*.png feature\ construction/resnet50_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_resnet50/flocking/2/*.png feature\ construction/resnet50_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_resnet50/flocking/3/*.png feature\ construction/resnet50_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lstm/flocking/1/*.png feature\ construction/lstm_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lstm/flocking/2/*.png feature\ construction/lstm_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lstm/flocking/3/*.png feature\ construction/lstm_3.gif

convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lbp/flocking/1/*.png feature\ construction/lbp_1.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lbp/flocking/2/*.png feature\ construction/lbp_2.gif
convert -delay 50 -loop 0 ../predictions_all/feature\ construction/predictions_lbp/flocking/3/*.png feature\ construction/lbp_3.gif

# convert dataset analysis methods
convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/1/unguided/predictions_best/flocking/1/*.png dataset\ analysis/unguided_1.gif
convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/2/unguided/predictions_best/flocking/2/*.png dataset\ analysis/unguided_2.gif
convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/3/unguided/predictions_best/flocking/3/*.png dataset\ analysis/unguided_3.gif

convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/1/guided/predictions_best/flocking/1/*.png dataset\ analysis/guided_1.gif
convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/2/guided/predictions_best/flocking/2/*.png dataset\ analysis/guided_2.gif
convert -delay 50 -loop 0 ../predictions_all/dataset\ analysis/error_weighted/3/guided/predictions_best/flocking/3/*.png dataset\ analysis/guided_3.gif

# convert filtering method
convert -delay 50 -loop 0 ../predictions_all/pruning\ od/flocking/1/*.png pruning\ od/pruning_od_1.gif
convert -delay 50 -loop 0 ../predictions_all/pruning\ od/flocking/2/*.png pruning\ od/pruning_od_2.gif
convert -delay 50 -loop 0 ../predictions_all/pruning\ od/flocking/3/*.png pruning\ od/pruning_od_3.gif
