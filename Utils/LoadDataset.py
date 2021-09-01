from io import DEFAULT_BUFFER_SIZE
import os
import re
import math
import itertools
import os.path as path
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd

Dataset = namedtuple('Dataset', ['x', 'y', 'person'])
SamplesInfo = namedtuple('SamplesInfo', ['start', 'end', 'label', 'user', 'experiment'])

class Dataloader:

    """

    The input data and labels information should like this,

    input data of each csv file:

        The name of the file should be in the format "acc_exp(the experiment number)_user(the user number)".
        for example, the filename of data from user 01 and experiment 02 should be "acc_exp02_user01" and raw data should look like,
        
        time, x, y, z
        time, x, y, z
        time, x, y, z

        the time attribute is optional.

    labels information:

        the labels information should like,
        "experiment user activity_label the_start_index_of_samples the_end_index_of_the samples"

        for example,

        1 2 5 250 350 -> experiment = 1
                         user = 2
                         activity_label = 5
                         the_start_index_of_samples = 250
                         the_end_index_of_the samples = 350

    Arguments:
        data_path : the relative location of the data files.
        labels_path : the relative path of the labels information.
        test_ratio: float, the ratio of test observations taken from each group
            of (person, activity).
        validation_ratio: float, the ratio of validation observations taken from
            each group of (person, activity).
        classnames: List[str], list of gesture-names as strings.
        seed: int, the seed used to sample observations into train, validation,
            and test datasets.
        aeperator: The sperator used to seperate raw data in the csv file.
        sampling_rate : The rate at which the input data need to be sampled.
        length_in_seconds : The length of the sampling window.
        overlap_ratio : The percentage of samples that need to overlapped by window in each slide.

    Attributes:
        train: Dataset(x, y), named tuple with the train data.
        validation: Dataset(x, y), named tuple with the validation data.
        test: Dataset(x, y), named tuple with the test data.
        classnames: List[str], list of classnames, can be used to decode the
            target/prediction labels.
    """


    def __init__(self, data_path, labels_path, classnames, seperator, test_ratio = 0.1, validation_ratio = 0.1, seed = 0, sampling_rate = 50, length_in_seconds = 1, overlap_ratio = None):

        self.data_path = data_path
        self.labels_path = labels_path
        self.seperator = seperator
        self.classnames = classnames
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.sampling_rate = sampling_rate
        self.length_in_seconds = length_in_seconds
        self.overlap_ratio = overlap_ratio
        self.seed = seed
        self.normalized_files = self.seperate_files()
        self.all, self.subjects = self.window_sampling()
        (self.train,
         self.validation,
         self.test) = self.stratified_split()
      
    def seperate_files(self):
        '''
            This method returns a dictionary whose key-value pairs are the file path and samples information regarding
            file.
        '''

        activityInfo = pd.read_csv(self.labels_path, sep = " ", header = None)
        files_samples_pairs = defaultdict(list)

        index = 0

        while (index < len(activityInfo)):
            
            info = activityInfo.iloc[index]
            exp = str(info[0]) if (int(info[0]) > 9) else ("0" + str(info[0]))
            user = str(info[1]) if (int(info[1]) > 9) else ("0" + str(info[1]))
            label = info[2]-1
            start = info[3]-1
            end = info[4]
            filename = self.data_path + "/acc_exp" + exp + "_user" + user + ".txt"
            files_samples_pairs[filename].append(SamplesInfo(start, end, label, info[0], info[1]))
            
            index += 1

        return files_samples_pairs

    def window_sampling(self):

        """
        This method returns the complete dataset and number of subjects involved in the dataset after applying the window sampling method to the raw data.
        """

        x_list = []
        y_list = []
        personList = []

        for filename,sample_data in self.normalized_files.items():

            data = pd.read_csv(filename, sep = self.seperator, comment = ";", header = None)

            for sample in sample_data:

                curr = sample.start
                end = sample.end
                label = sample.label
                user = sample.user

                overlapping_elements = 0
                win_len = int((self.length_in_seconds) * (self.sampling_rate))

                if self.overlap_ratio != None:
                    overlapping_elements = int(((self.overlap_ratio) / 100) * (win_len))
                    if overlapping_elements >= win_len:
                        print('Number of overlapping elements exceeds window size.')
                        break

                while (curr < end - win_len):
                    x_list.append(data[curr:curr + win_len])
                    y_list.append(label)
                    personList.append(user)
                    curr = curr + win_len - overlapping_elements

                    
        x_numpy = np.asarray(x_list, dtype = np.float32)
        x_numpy = x_numpy[:, :, np.newaxis, :]
        y_numpy = np.asarray(y_list)
        person_numpy = np.asarray(personList)
        subjects = list(set(personList))

        return Dataset(x_numpy, y_numpy, person_numpy), subjects

    def stratified_split(self):

        """
        This method perfroms the stratified splitting of the modelling dataset and returns the train, test and validation datasets.
        """
        train_x, train_y, train_person = [], [], []
        validation_x, validation_y, validation_person = [], [], []
        test_x, test_y, test_person = [], [], []

        # create stratified splits

        rng = np.random.RandomState(self.seed)

        for person in self.subjects:

            for classname_index in range(len(self.classnames)):
                
                # subset observations for this person and activity match.
                person_classname_match = np.logical_and(self.all.y == classname_index, self.all.person == person)
                x_subset = self.all.x[person_classname_match, ...]
                num_of_obsv = x_subset.shape[0]
                
                #There may be some combination of (person, y) that doesn't have any observations.
                if num_of_obsv == 0:
                    continue
                
                # calculate dataset sizes
                validation_size = math.ceil(num_of_obsv * (self.validation_ratio))
                test_size = math.ceil(num_of_obsv * (self.test_ratio))
                train_size = num_of_obsv - (validation_size + test_size)

                # generate permutated indices
                indices = rng.permutation(num_of_obsv)


                # append splits to final dataset
                validation_x.append(x_subset[indices[0:validation_size]])
                validation_y.append(np.full(validation_size, classname_index, dtype = self.all.y.dtype))
                validation_person.append(np.full(validation_size, person, dtype = self.all.person.dtype))

                test_x.append(x_subset[indices[validation_size:validation_size + test_size]])
                test_y.append(np.full(test_size, classname_index, dtype = self.all.y.dtype))
                test_person.append(np.full(test_size, person, dtype = self.all.person.dtype))

                train_x.append(x_subset[indices[validation_size + test_size:]])
                train_y.append(np.full(train_size, classname_index, dtype = self.all.y.dtype))
                train_person.append(np.full(train_size, person, dtype = self.all.person.dtype))


        return (
            Dataset(np.concatenate(train_x), np.concatenate(train_y),
                    np.concatenate(train_person)),
            Dataset(np.concatenate(validation_x), np.concatenate(validation_y),
                    np.concatenate(validation_person)),
            Dataset(np.concatenate(test_x), np.concatenate(test_y),
                    np.concatenate(test_person))
        )