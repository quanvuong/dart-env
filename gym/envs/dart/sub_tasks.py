__author__ = 'yuwenhao'

import numpy as np

# used to defined various type of control tasks for a specific environment
class TaskList:
    def __init__(self, task_num):
        self.world_choice = []
        self.fix_param_tasks = [] # format: [id, [params]]
        self.range_param_tasks = [] # format: [id, [ranges]]
        self.joint_limit_tasks = [] # format: [joint id, [ranges]]
        self.task_num = task_num
        self.current_range_params = []
        self.range_param_expand_ratio = []

    def add_world_choice_tasks(self, args):
        # only one world choice task can be specified
        self.world_choice = np.copy(args)

    def add_fix_param_tasks(self, args):
        self.fix_param_tasks.append(args)

    def add_range_param_tasks(self, args, expand=None):
        self.range_param_tasks.append(args)
        if expand is None:
            self.range_param_expand_ratio.append(0)
        else:
            self.range_param_expand_ratio.append(expand)

    def expand_range_param_tasks(self):
        for i in range(len(self.range_param_tasks)):
            expan_ratio = self.range_param_expand_ratio[i]
            for j in range(len(self.range_param_tasks[i][1])):
                self.range_param_tasks[i][1][j][0] -= expan_ratio
                self.range_param_tasks[i][1][j][1] += expan_ratio

    def add_joint_limit_tasks(self, args):
        self.joint_limit_tasks.append(args)

    def get_task_inputs(self, taskid):
        if len(self.fix_param_tasks) > 0 or len(self.joint_limit_tasks) > 0 or len(self.world_choice) > 0:
            split_vec = np.zeros(self.task_num)
            split_vec[taskid] = 1
            return np.concatenate([split_vec, self.current_range_params])
        else:
            return np.array(self.current_range_params)

    def task_input_dim(self):
        if len(self.fix_param_tasks) > 0 or len(self.joint_limit_tasks) > 0 or len(self.world_choice) > 0:
            return self.task_num + len(self.range_param_tasks)
        else:
            return len(self.range_param_tasks)

    def resample_task(self, taskid):
        self.current_range_params = []
        param_id_list = []
        param_val_list = []
        jt_id_list = []
        jt_val_list = []
        world_choice=0
        if len(self.world_choice) > 0:
            world_choice = self.world_choice[taskid]
        for fix_param_task in self.fix_param_tasks:
            param_id_list.append(fix_param_task[0])
            param_val_list.append(fix_param_task[1][taskid])
        for range_param_task in self.range_param_tasks:
            samp_range = range_param_task[1][taskid]
            samp = np.random.uniform(samp_range[0], samp_range[1])
            self.current_range_params.append(samp)
            param_id_list.append(range_param_task[0])
            param_val_list.append(samp)
        for joint_task in self.joint_limit_tasks:
            jt_id_list.append(joint_task[0])
            jt_val_list.append(joint_task[1][taskid])
        return world_choice, param_id_list, param_val_list, jt_id_list, jt_val_list


    def reset_task(self):
        self.fix_param_tasks = [] # format: [id, [params]]
        self.range_param_tasks = [] # format: [id, [ranges]]
        self.joint_limit_tasks = [] # format: [joint id, [ranges]]
        self.current_range_params = []