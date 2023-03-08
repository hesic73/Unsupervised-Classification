"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        if database == 'cifar-10':
            return './cifar-10'
        
        elif database == 'cifar-20':
            return '/path/to/cifar-20/'

        elif database == 'stl-10':
            return '/path/to/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/path/to/imagenet/'
        
        elif database in ['proteasome-12']:
            return '/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/proteasome12_raw'
        
        elif database in ['proteasome-11']:
            return '/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/proteasome11_balanced'
        
        elif database =='cng':
            return '/Share/UserHome/tzhao/2023/data/cng-simu/'
        
        else:
            raise NotImplementedError
