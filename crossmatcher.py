import numpy as np


def import_bright_source_survey() -> list:
    """
    Reads the Bright Source Survey Catalogue and converts each entry to their respective coordinates.
    :returns a list of tuples containing the coordinates in the form of (ascension, declination)
    """
    catalogue = np.loadtxt('data/bright_source_survey.dat', usecols=range(1, 7))
    data = []

    # Entries 0 - 2 represents Right Ascension in Hours-Minutes-Seconds notation
    # Entries 3 - 5 represent Declination in Degrees-Minutes-Seconds notation
    for row, entry in enumerate(catalogue):
        hms = 15 * (entry[0] + entry[1] / 60 + entry[2] / 3600)
        dms = abs(entry[3]) + entry[4] / 60 + entry[5] / 3600
        dms = -dms if entry[3] < 0 else dms

        data.append((hms, dms))

    return data


def import_allsky_catalogue() -> list:
    """
    Reads the All Sky Galaxy Catalogue.
    :returns a list of tuples containing the coordinates in the form of (ascension, declination)
    """
    catalogue = np.loadtxt('data/allsky_galaxy_catalogue.csv', skiprows=1, delimiter=',', usecols=[0, 1])
    data = []

    for row, (ascension, declination) in enumerate(catalogue):
        data.append((ascension, declination))

    return data


def angular_dist(ascension1, declination1, ascension2, declination2) -> float:
    """
    Haversine Formula to find the great-circle distance between two points using their Right Ascensions and Declinations
    :returns: the distance between the 2 points
    """
    a = np.sin(
        (np.abs(declination1 - declination2) / 2)
    ) ** 2

    b = np.cos(declination1) * np.cos(declination2) * (np.sin(
        (np.abs(ascension1 - ascension2) / 2)
    ) ** 2)

    return 2 * np.arcsin(np.sqrt(a + b))


def crossmatch(catalogue1, catalogue2, max_radius=5) -> tuple:
    """
    Performs positional cross-matching to find the closest counterpart with a given radius.
    :param max_radius: radian threshold within which the object needs to be searched for
    :returns: tuple containing lists of matched and unmatched entries.
              matches -> list that contains the ids of matched entries from both catalogues and their distance
              no_matches -> list that contains only the id of unmatched objects from the first catalogue
    """
    matches = []
    no_matches = []

    # Converting arguments to radians as numpy trig functions work with radians
    catalogue1 = np.radians(catalogue1)
    catalogue2 = np.radians(catalogue2)
    max_radius = np.radians(max_radius)

    sorted_indices2 = np.argsort(catalogue2[:, 1])
    sorted_catalogue2 = catalogue2[sorted_indices2]  # sorting catalogue according to ascending declinations
    declinations = sorted_catalogue2[:, 1]  # fetching only declinations

    for id1, (ascension1, declination1) in enumerate(catalogue1):
        closest_distance = np.inf
        closest_id2 = None

        # Boxing in the search with the range [declination - max_radius, declination + max_radius] to
        # cut down distance calculation of the second catalogue
        start_index = declinations.searchsorted(declination1 - max_radius, side='left')
        end_index = declinations.searchsorted(declination1 + max_radius, side='right')

        for id2, (ascension2, declination2) in enumerate(sorted_catalogue2[start_index:end_index + 1],
                                                         start=start_index):
            distance = angular_dist(ascension1, declination1, ascension2, declination2)

            # New closest match if the calculated distance is smaller than the previous closest distance
            if distance < closest_distance:
                closest_distance = distance
                closest_id2 = id2

        if closest_distance > max_radius:
            no_matches.append(id1)
        else:
            matches.append([id1, sorted_indices2[closest_id2], np.degrees(closest_distance)])

    return matches, no_matches


if __name__ == '__main__':
    catalogue1 = import_bright_source_survey()
    catalogue2 = import_allsky_catalogue()
    matches, no_matches = crossmatch(catalogue1, catalogue2, 5)
    print('matches:', matches)
    print('unmatched:', no_matches)

