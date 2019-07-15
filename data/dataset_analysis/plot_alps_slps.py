# snippet from lfd.py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

with open("1/predictions_best/alp_dataset_1488308839.txt", "r") as alpDatasetFile:
    alpDataset = alpDatasetFile.readlines()
    alpPoints = [eval(x) for x in alpDataset]

    m = "o"
    c = "r"
    
    # plot ALPs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotName = "alp_points_best" + ".png"
    alpPoints = zip(*alpPoints)
    [xs, ys, zs] = alpPoints
    ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel("Max-align-turn")
    ax.set_ylabel("Max-cohere-turn")
    ax.set_zlabel("Max-separate-turn")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim1_best" + ".png"
    plt.hist(xs, color=c)
    plt.xlabel("Max-align-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim2_best" + ".png"
    plt.hist(ys, color=c)
    plt.xlabel("Max-cohere-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim3_best" + ".png"
    plt.hist(zs, color=c)
    plt.xlabel("Max-separate-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

with open("1/predictions_worst/alp_dataset_1488319453.txt", "r") as alpDatasetFile:
    alpDataset = alpDatasetFile.readlines()
    alpPoints = [eval(x) for x in alpDataset]

    m = "o"
    c = "r"
    
    # plot ALPs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotName = "alp_points_worst" + ".png"
    alpPoints = zip(*alpPoints)
    [xs, ys, zs] = alpPoints
    ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel("Max-align-turn")
    ax.set_ylabel("Max-cohere-turn")
    ax.set_zlabel("Max-separate-turn")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim1_worst" + ".png"
    plt.hist(xs, color=c)
    plt.xlabel("Max-align-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim2_worst" + ".png"
    plt.hist(ys, color=c)
    plt.xlabel("Max-cohere-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "alp_points_dim3_worst" + ".png"
    plt.hist(zs, color=c)
    plt.xlabel("Max-separate-turn")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

with open("1/predictions_best/slp_dataset_1488308839.txt", "r") as slpDatasetFile:
    slpDataset = slpDatasetFile.readlines()
    slpPoints = [eval(x) for x in slpDataset]

    m = "o"
    c = "r"
    
    # plot SLPs
    fig = plt.figure()
    plotName = "slp_points_best" + ".png"
    slpPoints = zip(*slpPoints)
    [xs, ys] = slpPoints
    plt.plot(xs, ys, c+m)
    plt.xlabel("Maximum Area")
    plt.ylabel("Maximum Perimeter")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "slp_points_dim1_best" + ".png"
    plt.hist(xs, color=c)
    plt.xlabel("Maximum Area")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)
    
    fig = plt.figure()
    plotName = "slp_points_dim2_best" + ".png"
    plt.hist(ys, color=c)
    plt.xlabel("Maximum Perimeter")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

with open("1/predictions_worst/slp_dataset_1488319453.txt", "r") as slpDatasetFile:
    slpDataset = slpDatasetFile.readlines()
    slpPoints = [eval(x) for x in slpDataset]

    m = "o"
    c = "r"
    
    # plot SLPs
    fig = plt.figure()
    plotName = "slp_points_worst" + ".png"
    slpPoints = zip(*slpPoints)
    [xs, ys] = slpPoints
    plt.plot(xs, ys, c+m)
    plt.xlabel("Maximum Area")
    plt.ylabel("Maximum Perimeter")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "slp_points_dim1_worst" + ".png"
    plt.hist(xs, color=c)
    plt.xlabel("Maximum Area")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)
    
    fig = plt.figure()
    plotName = "slp_points_dim2_worst" + ".png"
    plt.hist(ys, color=c)
    plt.xlabel("Maximum Perimeter")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)

with open("slp_dataset_1488996573.txt", "r") as slpDatasetFile:
    slpDataset = slpDatasetFile.readlines()
    slpPoints = [eval(x) for x in slpDataset]

    m = "o"
    c = "r"
    
    # plot SLPs
    fig = plt.figure()
    plotName = "slp_points_guided" + ".png"
    slpPoints = zip(*slpPoints)
    [xs, ys] = slpPoints
    plt.plot(xs, ys, c+m)
    plt.xlabel("Maximum Area")
    plt.ylabel("Maximum Perimeter")
    plt.savefig(plotName)
    plt.close(fig)

    fig = plt.figure()
    plotName = "slp_points_dim1_guided" + ".png"
    plt.hist(xs, color=c)
    plt.xlabel("Maximum Area")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)
    
    fig = plt.figure()
    plotName = "slp_points_dim2_guided" + ".png"
    plt.hist(ys, color=c)
    plt.xlabel("Maximum Perimeter")
    plt.ylabel("Number of points in each value range")
    plt.savefig(plotName)
    plt.close(fig)
