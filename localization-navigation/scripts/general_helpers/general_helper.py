#!/usr/bin/env python3

# Be wary of importing matplotlib with cv2 in the same workspace environment, due to reimporting pyqt dependency twice.
# import cv2

import datetime
import json
import os
import pickle
import requests
import shutil
import sys
import uuid
import yaml

from colorama import Fore, Style

from slugify import slugify
from bisect import bisect_left

class GeneralHelper:
    def establish_package_root(self, execution_file_path):
        self.package_root = os.path.dirname(execution_file_path)
        sys.path.append(self.package_root)

    def configure_separator_output(self):
        try:
            term_size = os.get_terminal_size()
            num_sep_chars = term_size.columns
        except Exception as e:
            num_sep_chars = 80

        self.emph_sep = ('=' * num_sep_chars) + '\n'
        self.trans_sep = ('-' * num_sep_chars) + '\n'

    def check_running_jupyter_notebook(self):
        self.is_running_jupyter_notebook = False

        try:
            shell = get_ipython().__class__.__name__

            if shell == 'ZMQInteractiveShell':
                self.is_running_jupyter_notebook = True

        except NameError:
            pass

    def tqdm_import(self):
        self.check_running_jupyter_notebook()

        if self.is_running_jupyter_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.tqdm = tqdm

    def internet_connection_test(self, test_url='https://google.com'):
        self.internet_connection = False

        connection_string = 'Internet connection: '

        connection_good_string = f'{connection_string}{Fore.GREEN}Good{Style.RESET_ALL}'

        connection_bad_string = f'{connection_string}{Fore.RED}Bad{Style.RESET_ALL}'

        connection_status_string = connection_bad_string

        try:
            response = requests.get(test_url)

            if response.status_code == 200:
                self.internet_connection = True
                connection_status_string = connection_good_string

        except requests.ConnectionError:
            self.internet_connection = False

        return connection_status_string

    def __init__(self):
        self.establish_package_root(execution_file_path=os.path.abspath(__file__))
        self.configure_separator_output()

        self.tqdm_import()

    def sep(self):
        print()
        print(self.trans_sep)

    def verify_url_endpoint(self, url):
        response = requests.get(url)

        if response.status_code == 200:
            return True

        return False

    def dir_create(self, dir_path, debug=False):
        dir_prompt = f'dir path @ ({dir_path})'

        if not os.path.exists(dir_path):
            if (debug):
                print(f'{dir_prompt} does not exist. Creating...')

            os.makedirs(dir_path)

            if (debug):
                print(f'{dir_prompt} created.')

        else:
            if (debug):
                print(f'{dir_prompt} already exists. Skipping dir creation...')

    def dir_remove(self, dir_path, debug=False):
        dir_prompt = f'dir path @ ({dir_path})'

        if not os.path.exists(dir_path):
            if (debug):
                print(f'{dir_prompt} does not exist. Skipping dir remove...')

        else:
            if (debug):
                print(f'{dir_prompt} exists. Removing dir...')

            shutil.rmtree(dir_path)

            if (debug):
                print(f'{dir_prompt} removed.')

    def clean_filename(self, chosen_file):
        file_name = os.path.basename(chosen_file)
        file_split = os.path.splitext(file_name)

        file_basename = file_split[0]
        file_extension = file_split[1]

        file_slug = slugify(file_basename)

        new_file_name = f'{file_slug}{file_extension}'

        return new_file_name

    def recursive_rename_files(self, file_data_path):
        for root, dirs, files in os.walk(file_data_path):
            for file in files:
                orig_path = os.path.join(root, file)
                new_file_name = self.clean_filename(file)
                new_path = os.path.join(root, new_file_name)
                os.rename(orig_path, new_path)

    def process_file_entry(self, file_path):
        file_entry = {}

        file_entry['path'] = file_path

        file_basename = os.path.basename(file_path)
        file_entry['basename'] = file_basename

        file_extension = os.path.splitext(file_basename)[1]
        file_entry['extension'] = file_extension

        return file_entry

    def recursive_get_file_list(self, dir_path, filter_dirs=None):
        file_list = []

        for root, dirs, files in os.walk(dir_path):
            root_basename = os.path.basename(root)

            if filter_dirs:
                if root_basename in filter_dirs:
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        file_entry = self.process_file_entry(file_path=file_path)

                        file_list.append(file_entry)

            else:
                for filename in files:
                    file_path = os.path.join(root, filename)
                    file_entry = self.process_file_entry(file_path=file_path)

                    file_list.append(file_entry)

        return file_list

    def filter_file_list_by_file_type(self, file_list, file_types):
        filtered_file_list = []

        for file in file_list:
            file_extension = file['extension']

            if file_extension in file_types:
                filtered_file_list.append(file)

        return filtered_file_list

    def binary_search(self, sorted_list, target):
        i = bisect_left(sorted_list, target)
        if i != len(sorted_list) and sorted_list[i] == target:
            return i
        return -1

    def write_txt_file(self, text, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        with open(data_file_path, mode='w', encoding='utf-8') as outfile:
            outfile.write(text)

    def write_json_file(self, data_dict, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        json_object = json.dumps(data_dict, indent=4)

        with open(data_file_path, mode='w', encoding='utf-8') as outfile:
            outfile.write(json_object)

    def read_json_file(self, data_file_path):
        if os.path.isfile(data_file_path):
            with open(data_file_path, mode='r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def write_jsonl_file(self, data_list, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        with open(data_file_path, mode='w', encoding='utf-8') as f:
            for entry in data_list:
                entry_dump = json.dumps(entry)
                f.write(f'{entry_dump}\n')

    def read_jsonl_file(self, data_file_path):
        if os.path.isfile(data_file_path):
            with open(data_file_path, mode='r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            return data

    def write_html_file(self, data_text, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        with open(data_file_path, mode='w', encoding='utf-8') as f:
            f.write(data_text)

    def read_html_file(self, data_file_path):
        if os.path.isfile(data_file_path):
            with open(data_file_path, mode='r', encoding='utf-8') as f:
                data = f.read()

            return data

    def write_pickle_file(self, data_object, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)
        with open(data_file_path, mode='wb') as f:
            pickle.dump(obj=data_object, file=f)

    def read_pickle_file(self, data_file_path):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        if os.path.isfile(data_file_path):
            with open(data_file_path, mode='rb') as f:
                pickle_object = pickle.load(f)

            return pickle_object

    def write_large_pickle_file(self, data_object, data_file_path, chunk_size=100000):
        par_dir = os.path.dirname(data_file_path)

        self.dir_create(par_dir)

        with open(data_file_path, mode='wb') as f:
            pickle.dump(len(data_object), f, protocol=pickle.HIGHEST_PROTOCOL)
            for i, (key,value) in enumerate(data_object.items()):
                if i % chunk_size == 0:
                    print(f"Saving chunk {i//chunk_size + 1}")
                pickle.dump((key, value), f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_large_pickle_file(self, data_file_path):
        """Reads a large dictionary from a pickle file in chunks."""
        data_object = {}

        with open(data_file_path, mode='rb') as f:
            num_items = pickle.load(f)  # Read dictionary size

            for _ in range(num_items):
                key, value = pickle.load(f)
                data_object[key] = value  # Reconstruct dictionary

        return data_object

    def read_yaml_file(self, data_file_path):
        if os.path.isfile(data_file_path):
            with open(data_file_path, mode='r') as f:
                data = yaml.safe_load(f)

            return data

    def write_yaml_file(self, data_dict, data_file_path):
        with open(data_file_path, mode='w') as f:
            yaml.dump(data_dict, f)

    def get_max_workers(self):
        return os.cpu_count()-1

    def generate_id(self):
        return str(uuid.uuid4())

    def generate_local_time_str(self):
        now = datetime.datetime.now()
        local_now = now.astimezone()
        local_tz = local_now.tzinfo
        local_tzname = local_tz.tzname(local_now)

        time_str = f'{now.strftime("%Y%m%d_%H%M%S")}_{local_tzname.strip().lower()}'

        return time_str

    def pprint_iterable(self, iterable):
        print()
        print(self.emph_sep)

        print(*iterable, sep='\n')

        print()
        print(self.emph_sep)

    def pprint_dict(self, dict):
        print(f'{json.dumps(dict, indent=4)}')

    def view_single_img(self, header_title, img):
        import cv2
        while True:
            cv2.imshow(header_title, img)

            key = cv2.waitKey(1)

            # keys 'q'
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

class GeneralTester(GeneralHelper):
    def __init__(self):
        super().__init__()

        self.dummy_dir_path = os.path.join(self.package_root, 'dummy_dir')

    def test_dir_create(self):
        print('Test dir_create...')
        self.dir_create(self.dummy_dir_path)

    def test_dir_remove(self):
        print('Test dir_remove...')
        self.dir_remove(self.dummy_dir_path)

if __name__ == '__main__':
    tester = GeneralTester()
    print(tester.emph_sep)

    tester.test_dir_create()
    tester.sep()

    tester.test_dir_remove()
    tester.sep()

    print(tester.emph_sep)
