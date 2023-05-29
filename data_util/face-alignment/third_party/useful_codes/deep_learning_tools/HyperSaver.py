import pandas as pd
import os
import time
import warnings
from collections import OrderedDict

class HyperSaver(object):
    """
    A helper class to save hyperparameters and performance
    after neural network training and testing.
    """
    def __init__(self, init_template=None,
                 set_id_by_time=True):
        if not init_template:
            raise NotImplementedError("Method not implented!")
        if isinstance(init_template, str):
            ext = os.path.splitext(init_template)[1]
        else:
            raise NotImplementedError("File extension {} not supported".format(ext))
        if ext == '.xlsx':
            template_df = pd.read_excel(init_template)
        elif ext == '.csv':
            template_df = pd.read_csv(init_template)
        else:
            raise Exception('Extension name {} not supported!'.format(ext))
        self.set_id_by_time = set_id_by_time
        self.opt_names = template_df.columns
        self.output_dict = OrderedDict.fromkeys(self.opt_names)
        # self.output_dict
        self.time_str = time.strftime('%Y%m%d%H%M')

    def get_config_from_class(self, opts):
        for opt_name in self.opt_names:
            opt_value = getattr(opts, opt_name, None)
            if opt_value and opt_name in self.output_dict.keys():
                if isinstance(opt_value, tuple) or isinstance(opt_value, list):
                    opt_value = str(opt_value)
                self.output_dic['opt_name'] = opt_value

    def set_config(self, input_dict, match_template=True):
        set_names = input_dict.keys()
        for set_name in set_names:
            if set_name not in self.output_dict.keys():
                if match_template:
                    continue
                warnings.warn("The name {} is not included in the template will be added in.".format(set_name))

            opt_value = input_dict[set_name]
            if isinstance(opt_value, tuple) or isinstance(opt_value, list):
                opt_value = str(opt_value)
            self.output_dict[set_name] = opt_value

    def save_config(self, save_path):
        if self.set_id_by_time:
            self.output_dict['ID'] = self.time_str
        df_save = pd.DataFrame(self.output_dict, index=[0])
        df_save.to_csv(save_path, index=False)

    def append_config(self, append_to_dest_csv):
        dest_csv = pd.read_csv(append_to_dest_csv)
        if self.set_id_by_time:
            self.output_dict['ID'] = self.time_str
        df_save = pd.DataFrame(self.output_dict, index=[0])
        dest_csv = pd.concat([dest_csv, df_save], ignore_index=True)
        # print("df_save: ", df_save)
        dest_csv.to_csv(append_to_dest_csv, index=False)
