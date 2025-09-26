import h5py
import numpy as np


class StartingPoint:
    def __init__(self, parameters):
        self.parameters = parameters
        self.starting_function = None
        self.X0 = None
        self.T0 = None
        self.F0 = None
        self.tgt0 = None

    def set_starting_function(self, fxn):
        self.starting_function = fxn

    def get_starting_values(self):
        if self.parameters["starting_point"]["source"] == "function":
            self.starting_values_from_function()
        elif self.parameters["starting_point"]["source"] == "file":
            self.starting_values_from_file()

    def starting_values_from_function(self):
        self.X0, self.T0 = self.starting_function()

        if "force" in self.parameters["continuation"]["parameter"]:
            self.T0 = 1 / self.parameters["forcing"]["frequency"]
            self.F0 = self.parameters["forcing"]["amplitude"]
        else:
            self.F0 = np.float64(0.0)

    def starting_values_from_file(self):
        file_data = h5py.File(
            self.parameters["starting_point"]["file_info"]["file_name"] + ".h5", "r+"
        )
        index = self.parameters["starting_point"]["file_info"]["solution_index"]

        self.T0 = file_data["/T"][index]
        x0 = file_data["/Config/INC"][:, index]
        v0 = file_data["/Config/VEL"][:, index]
        self.X0 = np.concatenate([x0, v0])

        try:
            self.tgt0 = file_data["/Tangent"][:, index]
        except:
            self.tgt0 = None
