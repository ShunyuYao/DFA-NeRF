import pandas as pd
import os
import time
import json
import requests
import warnings
from collections import OrderedDict


class HyperSaver(object):
    """
    A helper class to save hyperparameters and performance
    after neural network training and testing.
    """

    def __init__(self, init_template=None,
                 set_id_by_time=True, webhook_url=''):
        """
        init_template: a template file to init.
        set_id_by_time: ID is the necessay identification for one training log.
            We reconmmend to use the current time string '%Y%m%d%H%M' for ID.
        webhook_url: The webhook url for FeiShu online document.
        """
        # if not init_template:
        #     raise NotImplementedError("Method not implented!")
        
        self.set_id_by_time = set_id_by_time
        if init_template:
            if isinstance(init_template, str):
                ext = os.path.splitext(init_template)[1]
            else:
                raise NotImplementedError(
                    "File extension {} not supported".format(ext))
            if ext == '.xlsx':
                template_df = pd.read_excel(init_template)
            elif ext == '.csv':
                template_df = pd.read_csv(init_template)
            else:
                raise Exception('Extension name {} not supported!'.format(ext))
            self.opt_names = template_df.columns
            self.output_dict = OrderedDict.fromkeys(self.opt_names)
        else:
            self.output_dict = {}
        # self.output_dict
        self.time_str = time.strftime('%Y%m%d%H%M')
        self.webhook_url = webhook_url
        self.output_dict['ID'] = self.time_str

    def get_config_from_class(self, opts, match_template=True):
        """
        Get argumentparser like class to extract hyperparameters.
        opts: The argumentparser
        """
        self.configs = opts
        if match_template:
            for opt_name in self.opt_names:
                # opt_value = getattr(opts, opt_name, None)
                opt_split_names = opt_name.split('.')
                opt_value = get_attr_tree(
                    opts, opt_split_names, len(opt_split_names)-1)
                if opt_value is not None and opt_name in self.output_dict.keys():
                    # if isinstance(opt_value, tuple) or isinstance(opt_value, list):
                    #     opt_value = str(opt_value)
                    self.output_dict[opt_name] = opt_value
        else:
            data = vars(self.configs)
            for key, value in data.items():
                self.output_dict[key] = value

    def set_time_str(self, time_str):
        """
        Set the time string for ID youself.
        """
        self.time_str = time_str

    def set_config(self, input_dict, match_template=True):
        """
        Set model performance config.
        input_dict: model performance dict
        match_template: if ture, ignore the performance not exists
        in the template file.
        """
        set_names = input_dict.keys()
        self.configs.performance = {}
        for set_name in set_names:
            if set_name not in self.output_dict.keys():
                if match_template:
                    continue
                warnings.warn(
                    "The name {} is not included in the template will be added in.".format(set_name))
            opt_value = input_dict[set_name]
            # if isinstance(opt_value, tuple) or isinstance(opt_value, list) or isinstance(opt_value, dict):
            #     opt_value = str(opt_value)
            self.output_dict[set_name] = opt_value
            self.configs.performance[set_name] = opt_value

    def save_config(self, save_path, append_if_exist=True):
        """
        Save config to path.
        """
        if self.set_id_by_time:
            self.output_dict['ID'] = self.time_str
        self._convert_nonNum_to_str()
        if os.path.exists(save_path) and append_if_exist:
            self.append_config(save_path)
        else:
            df_save = pd.DataFrame(self.output_dict, index=[0])
            df_save.to_csv(save_path, index=False)
    
    def save_all_configs_to_json(self, path):
        """
        Save config to local path.
        """
        data = vars(self.configs)
        with open(path, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4, separators=(', ', ': '))

    def append_config(self, append_to_dest_csv, add_if_not_exist=True):
        dest_csv = pd.read_csv(append_to_dest_csv)
        if self.set_id_by_time:
            self.output_dict['ID'] = self.time_str
        self._convert_nonNum_to_str()
        df_save = pd.DataFrame(self.output_dict, index=[0])
        if add_if_not_exist:
            dest_cols = dest_csv.columns
            for col in df_save.columns:
                if col not in dest_cols:
                    dest_csv.columns.append(col)
        dest_csv = pd.concat([dest_csv, df_save],
                             ignore_index=True, sort=False)
        # print("df_save: ", df_save)
        dest_csv.to_csv(append_to_dest_csv, index=False)

    def send_webhook_message(self):
        """
        send webhook to FeiShu, data sample:
        data = {
            "events":[
                {
                    "id":123,
                    "dataset":"h36m",
                    "arch":"resnet"
                },
                {
                    "id":234,
                    "dataset":"h36m",
                    "arch":"resnet"
                }
            ]
        }
        """
        # webhook_data = self.output_dict
        # data = json.dumps(webhook_data)
        self._generate_json_output()
        res = requests.post(url=self.webhook_url, data=self.json_out)
        return True

    def _show_serialized_json(self):
        """
        Show serialized json format for FeiShu webhook.
        """
        self._generate_json_output()
        print(self.json_out)
    
    def _generate_json_output(self):
        self.json_out = json.dumps(self.output_dict, sort_keys=True,
              indent=4, separators=(', ', ': '))
        # self.json_parse = json.loads(self.json_out)
    
    def _convert_nonNum_to_str(self):
        temp = self.output_dict.copy()
        for key, value in temp.items():
            if not isinstance(value, int) and not isinstance(value, float): # and not isinstance(value, long):
                value = str(value)
                self.output_dict[key] = value


def get_attr_tree(obj, attr_names, i):
    if i > 0:
        obj = get_attr_tree(obj, attr_names, i-1)
    return getattr(obj, attr_names[i], None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HyperSaver Test')
    parser.add_argument('--name', default='debug_hyper',
                        type=str, help='The name of this training')
    parser.add_argument('--epoch', default=50, type=int, help='train epochs')
    parser.add_argument('--batch_size', default=4, type=int)

    args = parser.parse_args()
    # Init
    hyperSaver = HyperSaver(
        init_template='', set_id_by_time=True, webhook_url='https://www.feishu.cn/flow/api/trigger-webhook/42c2c823ae4dda5213d04a46e27473a5')
    # ./hyperSaverTemplate.xlsx
    # Get hyperparameters from ArgumentParser
    hyperSaver.get_config_from_class(args, match_template=False)
    # Get model performance and add to HyperSaver
    model_perf = {
        'loss': [0.1, 0.2],
        'accuracy': {'s1': 98, 's2': 73},
    }
    hyperSaver.set_config(model_perf, match_template=False)
    # hyperSaver.send_webhook_message()
    # Save to local
    print(hyperSaver._show_serialized_json())
    hyperSaver.save_config('./results.csv')
    hyperSaver.save_all_configs_to_json('./results.json')
    # Use FeiShu webhook
    # hyperSaver.send_webhook_message()