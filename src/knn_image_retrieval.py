import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
import os
import shutil
from src.constants import (
    IMAGE_DATASET_PATH,
    RETRIEVED_IMAGES_PATH,
    CSV_FILE_PATH,
    IMAGE_FILE_EXTENSION,
    K_NEIGHBORS,
    LEAF_SIZE,
)


def load_histograms_from_csv(csv_file_path, method="yeo-johnson", standardize=True):
    """
    Loads all image histograms from the CSV file, skipping the header,
    and removes columns that consist only of zeros. Applies Power Transform
    to the remaining columns and selects specified columns.

    Args:
        csv_file_path (str): Path to the CSV file containing the histograms.
        method (str): The method used for power transformation ('box-cox' or 'yeo-johnson').
        standardize (bool): Whether to standardize the transformed data.

    Returns:
        np.ndarray: Array of transformed histograms with non-zero columns only and selected columns.
    """
    # Load the CSV file into a pandas DataFrame, specifying no header
    histograms_df = pd.read_csv(csv_file_path, header=None)

    # Remove columns that are all zeros
    non_zero_columns_df = histograms_df.loc[:, (histograms_df != 0).any(axis=0)]

    # Instantiate the Power Transformer with specified parameters
    power_transformer = PowerTransformer(method=method, standardize=standardize)

    # Fit the transformer and transform the data
    transformed_data = power_transformer.fit_transform(non_zero_columns_df)

    # Specify the columns to select (ensure they are within the valid range)
    selected_columns = [
        1,
        5,
        13,
        14,
        20,
        26,
        28,
        29,
        40,
        47,
        50,
        51,
        61,
        69,
        70,
        81,
        93,
        99,
        106,
        107,
        111,
        112,
        114,
        115,
        116,
        121,
        123,
        124,
        126,
        128,
        134,
        136,
        147,
        148,
        159,
        163,
        164,
        166,
        170,
        182,
        184,
        190,
        191,
        199,
        204,
        205,
        212,
        219,
        224,
        231,
        241,
        245,
        247,
        250,
        257,
        258,
        261,
        265,
        268,
        274,
        279,
        283,
        285,
        286,
        287,
        290,
        295,
        296,
        298,
        305,
        306,
        308,
        309,
        310,
        312,
        313,
        314,
        315,
        320,
        324,
        328,
        329,
        330,
        334,
        335,
        336,
        337,
        339,
        350,
        353,
        354,
        357,
        362,
        364,
        366,
        367,
        372,
        375,
        376,
        383,
        387,
        388,
        390,
        391,
        395,
        399,
        400,
        402,
        404,
        409,
        412,
        413,
        414,
        415,
        418,
        423,
        424,
        428,
        432,
        436,
        442,
        444,
        447,
        449,
        450,
        451,
        452,
        454,
        455,
        458,
        462,
        463,
        465,
        469,
        470,
        471,
        473,
        475,
        476,
        477,
        478,
        479,
        480,
        485,
        492,
        493,
        494,
        495,
        501,
        503,
        506,
        507,
        512,
        513,
        514,
        517,
        518,
        519,
        520,
        528,
        529,
        535,
        536,
        541,
        543,
        544,
        546,
        558,
        565,
        566,
        568,
        570,
        572,
        573,
        576,
        577,
        580,
        590,
        594,
        597,
        598,
        600,
        602,
        609,
        612,
        614,
        617,
        619,
        621,
        622,
        623,
        624,
        641,
        643,
        648,
        650,
        652,
        653,
        654,
        655,
        656,
        658,
        662,
        666,
        667,
        670,
        672,
        675,
        676,
        678,
        679,
        680,
        681,
        683,
        686,
        689,
        690,
        691,
        694,
        698,
        699,
        702,
        703,
        704,
        711,
        716,
        719,
        721,
        722,
        727,
        729,
        732,
        734,
        749,
        750,
        758,
        760,
        761,
        770,
        771,
        774,
        775,
        789,
        791,
        793,
        796,
        805,
        807,
        819,
        820,
        821,
        828,
        831,
        835,
        840,
        842,
        844,
        849,
        855,
        859,
        870,
        871,
        883,
        890,
        897,
        899,
        900,
        907,
        909,
        911,
        915,
        918,
        919,
        926,
        931,
        935,
        945,
        950,
        953,
        961,
        962,
        963,
        965,
        967,
        973,
        974,
        977,
        979,
        983,
        987,
        992,
        1001,
        1002,
        1003,
        1004,
        1007,
        1012,
        1016,
        1017,
        1020,
        1026,
        1029,
        1038,
        1043,
        1045,
        1048,
        1050,
        1052,
        1064,
        1068,
        1071,
        1079,
        1085,
        1089,
        1091,
        1095,
        1099,
        1100,
        1104,
        1105,
        1107,
        1114,
        1117,
        1119,
        1120,
        1122,
        1125,
        1130,
        1133,
        1135,
        1138,
        1145,
        1146,
        1147,
        1148,
        1156,
        1168,
        1170,
        1171,
        1174,
        1175,
        1176,
        1178,
        1181,
        1184,
        1185,
        1190,
        1195,
        1198,
        1201,
        1206,
        1213,
        1216,
        1217,
        1219,
        1220,
        1226,
        1229,
        1230,
        1236,
        1239,
        1240,
        1244,
        1245,
        1261,
        1265,
        1270,
        1273,
        1276,
        1280,
        1281,
        1285,
        1289,
        1295,
        1298,
        1299,
        1302,
        1304,
        1305,
        1313,
        1315,
        1318,
        1320,
        1324,
        1325,
        1326,
        1328,
        1330,
        1331,
        1332,
        1334,
        1339,
        1346,
        1347,
        1348,
        1349,
        1352,
        1353,
        1354,
        1356,
        1358,
        1359,
        1366,
        1368,
        1370,
        1372,
        1374,
        1377,
        1384,
        1386,
        1387,
        1389,
        1391,
        1394,
        1395,
        1396,
        1411,
        1423,
        1424,
        1425,
        1427,
        1448,
        1454,
        1455,
        1456,
        1457,
        1458,
        1479,
        1486,
        1488,
        1489,
        1490,
        1491,
        1493,
        1494,
        1495,
        1496,
        1497,
        1499,
        1501,
        1502,
        1504,
        1515,
        1519,
        1520,
        1526,
        1527,
        1528,
        1530,
        1531,
        1532,
        1533,
        1536,
        1537,
        1542,
        1543,
        1546,
        1551,
        1552,
        1555,
        1561,
        1563,
        1564,
        1565,
        1566,
        1568,
        1569,
        1571,
        1574,
        1576,
        1580,
        1586,
        1588,
        1593,
        1595,
        1596,
        1598,
        1601,
        1602,
        1604,
        1605,
        1606,
        1607,
        1608,
        1609,
        1612,
        1614,
        1617,
        1620,
        1623,
        1626,
        1627,
        1629,
        1633,
        1641,
        1642,
        1643,
        1644,
        1645,
        1647,
        1648,
        1649,
        1650,
        1653,
        1657,
        1658,
        1664,
        1668,
        1669,
        1670,
        1671,
        1672,
        1674,
        1675,
        1684,
        1685,
        1686,
        1687,
        1690,
        1692,
        1694,
        1696,
        1698,
        1703,
        1705,
        1707,
        1708,
        1711,
        1712,
        1713,
        1714,
        1717,
        1720,
        1725,
        1728,
        1729,
        1731,
        1732,
        1735,
        1737,
        1743,
        1744,
        1746,
        1748,
        1752,
        1753,
        1756,
        1765,
        1767,
        1771,
        1773,
        1777,
        1780,
        1781,
        1782,
        1788,
        1792,
        1794,
        1795,
        1798,
        1801,
        1803,
        1809,
        1813,
        1816,
        1818,
        1822,
        1823,
        1825,
        1829,
        1831,
        1835,
        1837,
        1840,
        1841,
        1844,
        1849,
        1851,
        1852,
        1854,
        1856,
        1858,
        1859,
        1862,
        1864,
        1865,
        1872,
        1873,
        1879,
        1881,
        1885,
        1887,
        1888,
        1889,
        1893,
        1896,
        1898,
        1913,
        1919,
        1921,
        1925,
        1926,
        1927,
        1929,
        1930,
        1933,
        1934,
        1939,
        1941,
        1945,
        1951,
        1952,
        1955,
        1963,
        1964,
        1968,
        1969,
        1971,
        1976,
        1978,
        1986,
        1989,
        1995,
        2000,
        2004,
        2006,
        2007,
        2012,
        2017,
        2022,
        2024,
        2027,
        2029,
        2030,
        2031,
        2036,
        2038,
        2040,
        2042,
        2048,
        2052,
        2055,
        2057,
        2062,
        2064,
        2066,
        2067,
        2068,
        2070,
        2071,
        2072,
        2073,
        2077,
        2078,
        2080,
        2085,
        2087,
        2089,
        2090,
        2091,
        2101,
        2105,
        2106,
        2108,
        2110,
        2112,
        2131,
        2134,
        2137,
        2138,
        2139,
        2140,
        2141,
        2142,
        2148,
        2151,
        2153,
        2154,
        2156,
        2165,
        2167,
        2170,
        2172,
        2173,
        2174,
        2176,
        2177,
        2178,
        2181,
        2183,
        2185,
        2195,
        2201,
        2216,
        2220,
        2221,
        2224,
        2227,
        2229,
        2233,
        2235,
        2238,
        2240,
        2243,
        2246,
        2247,
        2249,
        2251,
        2252,
        2256,
        2260,
        2263,
        2267,
        2271,
        2272,
        2276,
        2279,
        2280,
        2283,
        2285,
        2288,
        2291,
        2292,
        2294,
        2295,
        2298,
        2299,
        2300,
        2301,
        2303,
        2304,
        2306,
        2310,
        2311,
        2312,
        2319,
        2322,
        2325,
        2327,
        2333,
        2334,
        2339,
        2340,
        2342,
        2347,
        2355,
        2356,
        2359,
        2360,
        2361,
        2362,
        2363,
        2364,
        2365,
        2366,
        2371,
        2372,
        2373,
        2376,
        2386,
        2395,
        2396,
        2397,
        2400,
        2401,
        2402,
        2404,
        2405,
        2407,
        2409,
        2410,
        2412,
        2414,
        2415,
        2416,
        2418,
        2421,
        2422,
        2423,
        2425,
        2426,
        2428,
        2432,
        2434,
        2435,
        2437,
        2441,
        2444,
        2447,
        2451,
        2454,
        2455,
        2458,
        2460,
        2468,
        2472,
        2473,
        2474,
        2475,
        2479,
        2482,
        2485,
        2486,
        2488,
        2492,
        2498,
        2500,
        2502,
        2505,
        2514,
        2518,
        2520,
        2530,
        2532,
        2534,
        2535,
        2538,
        2539,
        2546,
        2549,
        2551,
        2554,
        2559,
        2561,
        2562,
        2565,
        2567,
        2569,
        2571,
        2574,
        2578,
        2581,
        2589,
        2591,
        2592,
        2594,
        2595,
        2596,
        2599,
        2600,
        2601,
        2602,
        2603,
        2605,
        2606,
        2608,
        2609,
        2610,
        2615,
        2616,
        2617,
        2620,
        2626,
        2627,
        2635,
        2637,
        2638,
        2640,
        2642,
        2647,
        2648,
        2649,
        2650,
        2653,
        2661,
        2669,
        2671,
        2672,
        2673,
        2677,
        2678,
        2679,
        2680,
        2683,
        2686,
        2689,
        2691,
        2692,
        2696,
        2702,
        2706,
        2708,
        2713,
        2719,
        2723,
        2725,
        2726,
        2728,
        2730,
        2731,
        2733,
        2735,
        2736,
        2737,
        2741,
        2743,
        2744,
        2746,
        2747,
        2750,
        2752,
        2756,
        2762,
        2764,
        2769,
        2774,
        2778,
        2779,
        2780,
        2784,
        2789,
        2791,
        2794,
        2796,
        2798,
        2801,
        2810,
        2812,
        2819,
        2820,
        2821,
        2826,
        2836,
        2837,
        2840,
        2841,
        2846,
        2847,
        2850,
        2853,
        2855,
        2856,
        2857,
        2859,
        2867,
        2871,
        2873,
        2880,
        2882,
        2883,
        2885,
        2888,
        2889,
        2893,
        2898,
        2901,
        2902,
        2904,
        2905,
        2906,
        2912,
        2916,
        2920,
        2921,
    ]

    # Ensure selected columns are within the valid range
    valid_columns = [col for col in selected_columns if col < transformed_data.shape[1]]

    # Select the specified columns from the transformed data
    transformed_selected_data = transformed_data[:, valid_columns]

    # Specify the columns to select (ensure they are within the valid range)
    selected_columns = [
        8,
        16,
        18,
        23,
        29,
        32,
        43,
        46,
        52,
        55,
        61,
        67,
        72,
        81,
        88,
        91,
        92,
        94,
        97,
        102,
        103,
        107,
        114,
        118,
        120,
        124,
        126,
        131,
        133,
        135,
        138,
        140,
        154,
        156,
        157,
        158,
        164,
        166,
        169,
        170,
        174,
        175,
        187,
        192,
        196,
        197,
        198,
        203,
        204,
        205,
        208,
        210,
        213,
        221,
        222,
        225,
        229,
        230,
        235,
        239,
        241,
        244,
        250,
        256,
        262,
        273,
        285,
        291,
        292,
        295,
        296,
        302,
        307,
        308,
        312,
        314,
        320,
        322,
        326,
        336,
        341,
        350,
        352,
        353,
        354,
        357,
        363,
        366,
        373,
        375,
        380,
        383,
        388,
        390,
        394,
        407,
        410,
        413,
        421,
        423,
        424,
        432,
        435,
        436,
        439,
        446,
        447,
        449,
        451,
        452,
        455,
        456,
        458,
        461,
        462,
        464,
        468,
        472,
        478,
        480,
        481,
        486,
        490,
        491,
        496,
        497,
        498,
        501,
        512,
        513,
        515,
        516,
        518,
        524,
        525,
        528,
        538,
        545,
        546,
        547,
        548,
        552,
        554,
        556,
        562,
        563,
        567,
        570,
        572,
        574,
        576,
        580,
        582,
        586,
        590,
        591,
        592,
        595,
        596,
        598,
        600,
        601,
        610,
        614,
        615,
        618,
        622,
        626,
        636,
        637,
        640,
        646,
        648,
        649,
        651,
        654,
        655,
        657,
        659,
        674,
        680,
        684,
        696,
        697,
        703,
        706,
        707,
        713,
        714,
        722,
        723,
        727,
        730,
        733,
        734,
        738,
        739,
        742,
        746,
        751,
        752,
        754,
        764,
        765,
        767,
        769,
        770,
        771,
        773,
        774,
        775,
        779,
        780,
        782,
        788,
        789,
        792,
        793,
        795,
        796,
        798,
        804,
        805,
        807,
        815,
        816,
        817,
        824,
        826,
        831,
        834,
        838,
        839,
        842,
        843,
        845,
        846,
        847,
        848,
        851,
        857,
        866,
        871,
        873,
        876,
        878,
        881,
        884,
        891,
        893,
        898,
        909,
        914,
        916,
        928,
        930,
        931,
        935,
        936,
        939,
        949,
        950,
        952,
    ]

    # Ensure selected columns are within the valid range
    valid_columns = [col for col in selected_columns if col < transformed_data.shape[1]]

    # Select the specified columns from the transformed data
    transformed_selected_data = transformed_selected_data[:, valid_columns]

    # Return the transformed and selected data as a numpy array
    return transformed_selected_data


def retrieve_similar_images(query_histogram, histograms, k=K_NEIGHBORS):
    """
    Retrieves the most similar images based on KNN.

    Args:
        query_histogram (np.ndarray): The histogram of the query image.
        histograms (np.ndarray): Histograms of all images.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        tuple: Distances and indices of the retrieved images.
    """
    # Initialize the Nearest Neighbors model
    knn = KNeighborsClassifier(
        n_neighbors=k + 1,  # +1 to account for excluding the query image
        metric="canberra",  # Using Canberra distance
        weights="distance",  # Weight neighbors by their distance
        algorithm="auto",  # Or 'auto' for the best choice
    )
    knn.fit(histograms, np.arange(histograms.shape[0]))

    # Reshape the query histogram to 2D array
    query_histogram = query_histogram.reshape(1, -1)

    # Find the k-nearest neighbors
    distances, indices = knn.kneighbors(query_histogram)

    return distances.flatten(), indices.flatten()  # Return distances and indices


def process_image_histogram(image_index, histograms):
    """
    Extracts the histogram for a specific image based on its index in the dataset.

    Args:
        image_index (int): Index of the image in the dataset.
        histograms (np.ndarray): Array of histograms of all images.

    Returns:
        np.ndarray: The histogram of the queried image.
    """
    return histograms[image_index]  # Return the histogram of the query image


def retrieve_and_save_images_for_all_dataset():
    """
    Process each image in the dataset, retrieve similar images, and save them.
    """
    # Ensure the main output directory exists
    os.makedirs(RETRIEVED_IMAGES_PATH, exist_ok=True)

    # Load histograms from the CSV file
    histograms = load_histograms_from_csv(CSV_FILE_PATH)
    print(
        f"csv file with {histograms.shape[0]} rows and {histograms.shape[1]} columns loaded."
    )

    # Iterate over all images in the dataset
    for i in range(histograms.shape[0]):
        # Retrieve similar images for the current image
        query_histogram = process_image_histogram(i, histograms)
        distances, retrieved_indices = retrieve_similar_images(
            query_histogram, histograms
        )

        # Retrieve corresponding image filenames
        image_filenames = [
            f"{index}{IMAGE_FILE_EXTENSION}" for index in retrieved_indices
        ]

        # Create a folder to save the retrieved images
        retrieval_folder = os.path.join(RETRIEVED_IMAGES_PATH, str(i))
        os.makedirs(retrieval_folder, exist_ok=True)

        # Save the retrieved images in the folder
        save_retrieved_images(image_filenames, retrieval_folder)

        # Save the retrieval rank information to CSV, excluding the query image
        save_retrieval_rank_csv(
            distances, image_filenames, retrieved_indices, i, retrieval_folder
        )

        print(f"Retrieved images for image {i} saved in '{retrieval_folder}'")


def save_retrieved_images(retrieved_image_filenames, output_folder):
    """
    Save the retrieved images in the specified folder.

    Args:
        retrieved_image_filenames (list): List of filenames of the retrieved images.
        output_folder (str): Directory to save the retrieved images.
    """
    for image_filename in retrieved_image_filenames:
        source_path = os.path.join(IMAGE_DATASET_PATH, image_filename)
        destination_path = os.path.join(output_folder, image_filename)

        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
        else:
            print(f"Source image '{source_path}' not found. Skipping.")


def save_retrieval_rank_csv(
    distances, image_filenames, retrieved_indices, query_index, output_folder
):
    """
    Save the retrieval rank, filename, and distance to a CSV file, excluding the query image itself.

    Args:
        distances (np.ndarray): Array of distances of the retrieved images.
        image_filenames (list): List of retrieved image filenames.
        retrieved_indices (np.ndarray): List of indices of the retrieved images.
        query_index (int): Index of the query image.
        output_folder (str): Directory to save the rank CSV.
    """
    rank_data = {"Retrieval Rank": [], "Retrieved Image Filename": [], "Distance": []}

    rank = 1
    for idx, (distance, image_filename, retrieved_idx) in enumerate(
        zip(distances, image_filenames, retrieved_indices)
    ):
        if retrieved_idx != query_index:  # Skip the query image
            rank_data["Retrieval Rank"].append(rank)
            rank_data["Retrieved Image Filename"].append(image_filename)
            rank_data["Distance"].append(distance)
            rank += 1

    # Save to CSV
    rank_csv_path = os.path.join(output_folder, "rank.csv")
    rank_df = pd.DataFrame(rank_data)
    rank_df.to_csv(rank_csv_path, index=False)
    print(f"Rank CSV saved at '{rank_csv_path}'")
