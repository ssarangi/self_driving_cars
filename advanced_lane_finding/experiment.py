import pandas as pd
import os
import logging
import matplotlib.image as mpimg
import typing

class ExperimentManagerOptions:
    def __init__(self):
        self.overwrite_if_experiment_exists = False

class ExperimentManager:
    def __init__(self, options : ExperimentManagerOptions):
        self._logger = logging.getLogger(__name__)
        if os.path.exists("experiments.csv") and os.path.exists("experiments"):
            self._df = pd.read_csv("experiments.csv")
        else:
            self._df = pd.DataFrame()
            if not os.path.exists("experiments"):
                try:
                    os.mkdir("experiments")
                except:
                    raise Exception("Cannot create an experiments directory. Please create one and rerun")

        self._options = options
        self._new_experiments = []
        self._new_experiments_dict = {}


    def set_logger(self, loggerobj):
        """
        Set the custom logger object from your application
        :param loggerobj: Any object which is obtained from logging.getLogger()
        :return: None
        """
        self._logger = loggerobj

    def new_experiment(self, name):
        if name is None or name.strip() == "":
            raise Exception("A New experiment needs a name")

        if 'experiment' in self._df.columns:
            if name in self._df.experiment.unique():
                if self._options.overwrite_if_experiment_exists is False:
                    raise Exception("Experiment already exists. If you want to overwrite set it in options - overwrite_if_experiment_exists")
                else:
                    self._df = self._df[self._df.experiment != name]

        experiment = Experiment(self, name, self._logger)
        self._new_experiments.append(experiment)
        self._new_experiments_dict[name] = experiment

        return experiment

    def commit_experiment(self, s: pd.Series):
        """
        Commit a particular experiment if it is not committed. Checkpoints are automatically done after every certain
        number of operations
        :param experiment: object
        :return: 
        """
        self._df = self._df.append(s)
        self._df.to_csv('experiments.csv', index=False)

    def get_experiment(self, experiment_name : str):
        """
        Get the Experiment Object for the name specified
        :param experiment_name: String for the experiment name
        :return: Experiment Object
        """
        return self._df[self._df.experiment == experiment_name]

    def to_markdown(self, experiment_name : str, filename : str):
        """
        Creates a markdown file from the experiments data. This has limited support for now
        :param filename: str
        :return: None
        """
        f = open(filename, 'w')

        # First create all the image references
        # [ //]:  # (Image References)
        #
        # [image1]:./ examples / undistort_output.png
        # "Undistorted"
        # [image2]:./ test_images / test1.jpg
        # "Road Transformed"
        # [image3]:./ examples / binary_combo_example.jpg
        # "Binary Example"
        # [image4]:./ examples / warped_straight_lines.jpg
        # "Warp Example"
        # [image5]:./ examples / color_fit_lines.jpg
        # "Fit Visual"
        # [image6]:./ examples / example_output.jpg
        # "Output"
        # [video1]:./ project_video.mp4
        # "Video"
        lines = []
        lines.append("[//]: # (Image References)")

        df = self._df[self._df.experiment == experiment_name]
        images_df = df[df.index == 'image']

        for idx, row in images_df.iterrows():
            image_name = os.path.basename(row.filename)
            s = '[%s]: %s "%s"' % (image_name, row.filename, row.title)
            lines.append(s)

        final_str = "\n".join(lines)
        print(final_str)

class Image:
    def __init__(self, exp_name, img, input_or_output, filename, title, description=""):
        self._exp_name = exp_name
        self._img = img
        self._filename = filename
        self._title = title
        self._description = description
        self._input_or_output = input_or_output

    def commit(self):
        mpimg.imsave(self._filename, self._img)

    def to_series(self):
        s = pd.Series([self._exp_name, self._input_or_output, self._filename, self._title, self._description],
                      index=['experiment', 'input_or_output', 'filename', 'title', 'description'], name="image")
        return s

class Parameter:
    def __init__(self, exp_name, input_or_output, param_name, param_value, title, description):
        self._exp_name = exp_name
        self._title = title
        self._description = description
        self._input_or_output = input_or_output
        self._param_name = param_name
        self._param_value = param_value

    def to_series(self):
        s = pd.Series([self._exp_name, self._input_or_output, self._param_name, self._param_value, self._title, self._description],
                      index=['experiment', 'input_or_output', 'parameter_name', 'parameter_value', 'title', 'description'],
                      name='parameter')

class Text:
    def __init__(self, exp_name, title, description):
        self._exp_name = exp_name
        self._title = title
        self._description = description

    def to_series(self):
        s = pd.Series([self._exp_name, self._title, self._description],
                      index=['experiment', 'title', 'description'])

class Experiment:
    """
    This is the Experiment Object. This should not be instantiated directly but should be done from the Experiment
    Manager which will set up certain defaults for it
    """
    def __init__(self, exp_mgr : ExperimentManager, name : str, logger):
        self._logger = logger
        self._name = name
        self._dirname = "experiments/" + name
        self._exp_mgr = exp_mgr

        if not os.path.exists(self._dirname):
            os.mkdir(self._dirname)

        # See if the images folder exists
        self._img_folder = self._dirname + "/images"
        if not os.path.exists(self._img_folder):
            os.mkdir(self._img_folder)

        self._images = []
        self._parameters = []

    @property
    def name(self):
        return self._name

    def _commit(self, s):
        self._exp_mgr.commit_experiment(s)

    def add_image(self, img, input_or_output, filename, title, description=""):
        """
        Adds an image to the experiment
        :param img: The raw image data 
        :param input_or_output Whether the image is a part of the inputs or outputs
        :param filename: Filename to save the image data
        :param description: Associate a description with the image data
        :return: None
        """
        if input_or_output.lower() != "input" and input_or_output.lower() != "output":
            raise Exception("input_or_output parameter can only contain 'input' or 'output'")

        image_object = Image(self._name, img, input_or_output, self._img_folder + "/" + filename, title, description="")
        image_object.commit()
        s = image_object.to_series()
        self._commit(s)

    def add_input_parameter(self, param_name, param_value, title, description):
        """
        Adds a parameter to the experiment. This could be either an input or output
        :param param_name: Parameter Name
        :param param_value: Parameter Value. Could be a single value or could be a list or dictionary
        :param title: Title of the parameter
        :param description: Description of the parameter
        :return: None
        """
        param_object = Parameter(self._name, "input", param_name, param_value, title, description)
        s = param_object.to_series()
        self._commit(s)