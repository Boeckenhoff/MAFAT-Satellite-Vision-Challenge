import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib as mpl
import seaborn as sns
from decimal import *
import os
import math


def read_annotations(path_to_labelTxt):
    """
    Read the annotation files(txt) and create a data frame and a list that include the frames' names.

    Parameterss
    ----------
    path_to_labelTxt : str
        The files location of the annotations.

    Returns
    -------
    dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.
    list    
        A list that includes the frames' names.
    """
    list_annotations_files = []
    list_frame_names = []
    columns_names = ['x1','y1','x2','y2','x3','y3','x4','y4' ,'category_id']

    # iterate over files in the directory
    labelTxt_files = os.listdir(path_to_labelTxt)
    for filename in labelTxt_files:

        df = pd.read_csv(os.path.join(path_to_labelTxt, filename), sep=" ", header=None, names=columns_names)
        frame_name = filename.split(".")[0]
        df["Frame"] = frame_name
        list_annotations_files.append(df)
        list_frame_names.append(frame_name)

    ann_train = pd.concat(list_annotations_files)
    ann_train.reset_index(drop = True, inplace = True)

    return ann_train,list_frame_names


def bb_out_of_frame(annotations, start_range_frame=0, stop_range_frame=1280):
    """
    Count the annotations (bounding boxes) that cross the frame size.

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.
    start_range_frame : int
        The lower limit range of square frame size.
    stop_range_frame = int
        The upper limit range of square frame size.

    Returns
    -------
    int
        The number of annotations (bounding boxes) that cross the frame size.
    """
    out_of_frame = annotations.loc[(annotations.x1<start_range_frame) | (annotations.x1>stop_range_frame) | (annotations.x2<start_range_frame) | (annotations.x2>stop_range_frame)| (annotations.x3<start_range_frame) | (annotations.x3>stop_range_frame) | (annotations.x4<start_range_frame) | (annotations.x4>stop_range_frame) | 
                    (annotations.y1<start_range_frame) | (annotations.y1>stop_range_frame) | (annotations.y2<start_range_frame) | (annotations.y2>stop_range_frame)| (annotations.y3<start_range_frame) | (annotations.y3>stop_range_frame) | (annotations.y4<start_range_frame) | (annotations.y4>stop_range_frame)]
    return out_of_frame


def anno_vis_bar(annotations):
    """
    Visualization of total annotations per class

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name..

    Returns
    -------
    -
    """
    ax = sns.countplot(data=annotations, y='category_id', order = annotations['category_id'].value_counts().index, palette = color_class)
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{x:.0f}' for x in container.datavalues])
    ax.set_ylabel('category_id')
    ax.set_xlabel('annotation count')
    plt.title('Total annotations per class')
    plt.show()

def anno_bb_out_of_frame_vis_bar(annotations):
    """
    Visualization of total annotations per class that cross the frame shape.

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name..

    Returns
    -------
    -
    """

    out_of_frame = bb_out_of_frame(annotations)
    sns.countplot(out_of_frame.category_id, order = out_of_frame['category_id'].value_counts().index, palette = color_class)
    plt.title('BB partial out of frame per class')
    plt.xticks(rotation=70)


def class_per_img_hist(annotations, class_obj):
    """
    Visualization of number of frames per number of objects per class.

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name..

    Returns
    -------
    -
    """
    #GroupBy df by Frame and classes for counting each category in each frame.
    ann_classes = annotations.groupby(['Frame','category_id'])['category_id'].count()
    ann_classes = ann_classes.to_frame()
    ann_classes = ann_classes.rename(columns ={'category_id':'count'}).reset_index()

    plt.subplots_adjust(hspace=1.8, wspace= 0.5)
    plt.suptitle('Number of frames per number of objects per class')#, fontsize=20)

    #Create dataframes of classes counting in all the frames for histogram visualization.
    for i,c in enumerate(class_obj):
        ax = plt.subplot(3, 5, i + 1)
        obj_df = ann_classes.loc[ann_classes.category_id == c]
        ax.set_title(c)
        ax.hist(obj_df['count'], bins = np.arange(1, obj_df['count'].max()+2), color = color_class[c])
        ax.set_xlabel(f'num_of_objects')
        ax.set_ylabel('num_of_\nframes')

def hist_features(metadata, f, bin_size):
    """
    Histogram of the number of frames per aoi.

    Parameters
    ----------
    metadata : dataframe
        Frames' metadata includes: Frame name, AOI, Resolution, Sun_Elevation, Azimuth, Sun_Azimuth, Hermetic_Small_Vehicle
    f : str
        Selected feature for histogram display.
    bin_size : int
        The range size of each bar in the histogram.

    Returns
    -------
    -
    """
    if f == 'AOI':
        metadata.loc[metadata.AOI =='unspecified', 'AOI'] = 23.0
        metadata['AOI'] = metadata.AOI.astype('float')

        bins=np.arange(metadata.AOI.min(), metadata.AOI.max()+2)
        x_ticks = np.append(bins[:-2], np.array(['unspecified']))


        fig, ax = plt.subplots()
        plt.title(f"Frames per {f} Histogram")
        _, edges, _ = ax.hist(x=f, data=metadata, bins=bins)
        ticks  = edges[:-1] + np.diff(edges) / 2

        ax.set_xticks(ticks)
        ax.set_xticklabels(x_ticks, rotation=40)
        plt.ylabel('Counts')
        plt.xlabel(f)
        metadata.loc[metadata.AOI ==23.0, 'AOI'] = 'unspecified'
    else:
        if bin_size>1:
            res_min = math.floor(metadata[f].min()/10)*10
            res_max = math.ceil(metadata[f].max()/10)*10
        else:
            res_min = math.floor(metadata[f].min()*10)/10
            res_max = math.ceil(metadata[f].max()*10)/10

        bins = np.arange(res_min, res_max+2*bin_size, bin_size)
        x_ticks = np.arange(res_min, res_max+bin_size,bin_size)

        fig, ax = plt.subplots()
        plt.title(f"Frames per {f} Histogram")
        plt.hist(x=f, data=metadata, bins=bins)

        plt.xticks(x_ticks)
        plt.ylabel('Counts')
        plt.xlabel(f)


def box_plot_aoi(metadata, f):
    """
    Box plot of the feature in different aoi.

    Parameters
    ----------
    metadata : dataframe
        Frames' metadata includes: Frame name, AOI, Resolution, Sun_Elevation, Hermetic_Small_Vehicle
    f : str
        Feature selected to visualize its box plot in each aoi.

    Returns
    -------
    -
    """
    metadata.loc[metadata.AOI =='unspecified', 'AOI'] = 23.0
    metadata['AOI'] = metadata.AOI.astype('float')

    g = sns.boxplot(data=metadata, x="AOI", y=f, color="green", saturation = 2.7)

    xtickes = [t.get_text()  for t in g.get_xticklabels()]
    xtickes[-1] = 'unspecified'
    g.set_xticklabels(xtickes)

    plt.title(f"Box plot of {f} for each AOI")
    plt.show()

    metadata.loc[metadata.AOI ==23.0, 'AOI'] = 'unspecified'

def total_ann_heatmap(annotations, class_obj):
    """
    Heatmap of the number of the class's annotations in each AOI.

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.

    Returns
    -------
    -
    """
    annotations.loc[annotations.AOI =='unspecified', 'AOI'] = 23.0
    annotations['AOI'] = annotations.AOI.astype('float')

    #GroupBy df by AOI and classes for counting each category in each AOI
    class_count_aoi = annotations.groupby(['AOI','category_id'])['category_id'].count()
    class_count_aoi = class_count_aoi.to_frame()
    class_count_aoi = class_count_aoi.rename(columns ={'category_id':'count_each_class'}).reset_index()

    #Create custom df of classes in each region for heatmap visualization
    list_aoi = []
    for aoi_num in class_count_aoi.AOI.unique():
        slice_aoi = class_count_aoi[class_count_aoi.AOI==aoi_num]
        slice_aoi = slice_aoi[['category_id','count_each_class']].T
        slice_aoi.columns = slice_aoi.iloc[0,:]
        slice_aoi.drop(index = 'category_id', axis = 0, inplace = True)
        slice_aoi.insert(0, 'AOI', [aoi_num])
        list_aoi.append(slice_aoi)

    class_count_aoi_df = pd.concat(list_aoi, axis = 0)
    class_count_aoi_df.reset_index(drop = True, inplace = True)
    class_count_aoi_df.fillna(0.0, inplace = True)
    class_count_aoi_df = class_count_aoi_df.astype('float')
    class_count_aoi_df['AOI'] = class_count_aoi_df.AOI.astype('int')
    class_count_aoi_df.set_index(['AOI'], drop = True, inplace =True)

    plt.figure(figsize = (20,14))
    plt.title("Total annotations in each region AOI")
    g = sns.heatmap(class_count_aoi_df[class_obj],  annot=True ,annot_kws={"fontsize":14},fmt=".0f", linewidth=.5, vmin=0, vmax=class_count_aoi.count_each_class.max(), cmap='pink')

    ytickes = [t.get_text()  for t in g.get_yticklabels()]
    ytickes[-1] = 'unspecified'
    g.set_yticklabels(ytickes, rotation = 30)

    annotations.loc[annotations.AOI ==23.0, 'AOI'] = 'unspecified'


def heatmap_res(annotations, class_obj):
    """
    Heatmap of the Resolution distribution for each class.

    Parameters
    ----------
    annotations : dataframe
        Annotations dataframe includes category_id and Resolution.

    Returns
    -------
    -
    """
    #Setting the resolution range in 0.1 increments
    res_min = math.floor(annotations.Resolution.min()*10)/10
    res_max = math.ceil(annotations.Resolution.max()*10)/10
    bins = np.arange(res_min, res_max+0.1, 0.1)

    #Create a custom df of resolution for each class for heatmap visualization
    category_resolution_list = []
    for c in class_obj:
        tagged_df_copy_class = annotations[annotations.category_id ==c]
        tagged_df_copy_class.reset_index(drop=True, inplace=True)
        category_df = pd.cut(x = tagged_df_copy_class.Resolution,bins = bins.tolist(), include_lowest=True)
        category_df_normalize = category_df.value_counts(normalize=True)
        category_df_normalize = category_df_normalize.to_frame().rename(columns = {"Resolution":c})
        category_resolution_list.append(category_df_normalize)   

    category_resolution = pd.concat(category_resolution_list, axis = 1)
    
    plt.figure(figsize = (14,14))
    plt.title("Resolution Distribution for each class")
    sns.heatmap(category_resolution.T)

def frame_with_annotation(frame_path, ann_frame):
    """
    Display of the selected frame and its annotations.

    Parameters
    ----------
    img_path : str
        The file location of the frame.
    ann_img : str
        The location of the corresponding annotations file to the frame.
    Returns
    -------
    -
    """
    try:
        img = Image.open(frame_path)
    except:
        print("Wrong frame name")
        return

    fig, ax = plt.subplots(1, figsize=(15, 15))

    # Display the image
    ax.imshow(img, cmap = 'gray')
    ax.grid(False)

    for index, row in ann_frame.iterrows():
        class_color = color_class[row['category_id']]

        rec_coor = [[row['x1'],row['y1']], [row['x2'],row['y2']], [row['x3'],row['y3']], [row['x4'],row['y4']]]
        # Create a Rectangle patch
        rect = patches.Polygon(rec_coor, linewidth=2.0,
                                edgecolor=class_color, facecolor="none")
        ax.add_patch(rect)

    plt.show()

def show_frames_by_AOI(anno):
    """
    Display of randomly sampled frames grouped by AOI.

    Parameters
    ----------
    anno : DataFrame
        The Pandas DataFrame containing the annotations.
    Returns
    -------
    -
    """
    sampled_frames_per_AOI = anno.drop_duplicates(subset=['Frame']).groupby(['AOI']).apply(lambda x: x.sample(n=5))["Frame"]

    fig, axs = plt.subplots(nrows=len(sampled_frames_per_AOI.groupby(level=0)), ncols=5)
    fig.tight_layout()

    for i, (AOI, new_df) in enumerate(sampled_frames_per_AOI.groupby(level=0)):
        for j, frame_to_show in enumerate(new_df):
            ann_frame = anno[anno.Frame == frame_to_show]
            frame_path = f"/content/images/{frame_to_show}.tiff"

            img = Image.open(frame_path)
            axs[i][j].imshow(img, cmap = 'gray')
            axs[i][j].axis('off')
            axs[i][j].title.set_text(f"AOI: {AOI}, Frame: {frame_to_show}")

    plt.show()


def frames_by_metadata(anno, column):
    """
    Display of randomly sampled frames grouped by the bins of an continous metadata feature.

    Parameters
    ----------
    anno : DataFrame
        The Pandas DataFrame containing the annotations.
    column : str
        The metadata feature to groupby the Frames. One of: Resolution, Sun_Elevation, Azimuth, Sun_Azimuth.
    Returns
    -------
    -
    """
    ann_unique = ann_train.drop_duplicates(subset=['Frame'])
    ann_unique_bins = ann_unique.copy()
    ann_unique_bins.loc[:,column] = pd.cut(ann_unique[column], 9, labels=False)
    sampled_frames = ann_unique_bins.groupby([column]).apply(lambda x: x.sample(n=5))["Frame"]

    fig, axs = plt.subplots(nrows=len(sampled_frames.groupby(level=0)), ncols=5)
    fig.tight_layout()

    for i, (AOI, new_df) in enumerate(sampled_frames.groupby(level=0)):
        for j, frame_to_show in enumerate(new_df):
            ann_frame = anno[anno.Frame == frame_to_show]
            frame_path = f"/content/images/{frame_to_show}.tiff"

            img = Image.open(frame_path)
            axs[i][j].imshow(img, cmap = 'gray')
            axs[i][j].axis('off')
            axs[i][j].title.set_text(f"{column}: {ann_frame[column].iloc[0]:.2f}, Frame: {frame_to_show}")

    plt.show()
