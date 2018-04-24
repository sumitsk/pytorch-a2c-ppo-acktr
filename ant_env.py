from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
import os
import xml.etree.ElementTree as ET

class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def change_attribute(self, root=None, tag_name=None, tag_identifier=None, attribute=None, value=None):
        if tag_identifier == '':
            root.find(tag_name).set(attribute, value)
            print(root.find(tag_name))

        else:
            # error check
            all_tags_found = 0

            # Get the identifier structure
            num_identifiers = len(tag_identifier)
            tag_identifiers_list = list(tag_identifier)

            tag_names_split = tag_name.split('/')
            num_tags = len(tag_names_split)
            starting_tag_name = '/'.join(tag_names_split[0:num_tags-num_identifiers])

            start_idx = num_tags - num_identifiers
            curr_idx = 0
            current_tag = None
            current_tag_name = starting_tag_name

            if starting_tag_name == '':
                current_tag = root
            else:
                current_tag = root.find(starting_tag_name)

            for i in range(start_idx, num_tags):
                attribute_name = tag_identifiers_list[curr_idx][0]
                attribute_value = tag_identifiers_list[curr_idx][1]
                all_tags_found = 0
                for child in current_tag:
                    if child.get(attribute_name) == attribute_value:
                        current_tag_name = current_tag_name + '/' + tag_names_split[i]
                        current_tag = child
                        curr_idx += 1
                        all_tags_found = 1

                        if i == num_tags-1:
                            child.set(attribute, value)
                        break

                if all_tags_found==0:
                    print("\n\nERROR: Tags hierarchy or tag_identifier list is wrong\n\n")
                    print("Tags hierarchy: {}".format(tag_name))
                    print("Tag identifier list: {}".format(tag_identifier))
                    print("Can't find {}=\"{}\" in {}\n\n".format(attribute_name, attribute_value, child))
                    assert all_tags_found==1

    def change_model(self, tag_names=None, tag_identifiers=None, attributes=None, values=None, xml_file_name=None, xml_file=None):
    # def change_model(self, *args, **kwargs):
        xmlFile = None

        assert len(tag_names) == len(attributes)
        assert len(tag_names) == len(values)

        if not xml_file == None:
            xmlFile = xml_file

        else:
            # Set the attribute using the value
            baseXMLFilePath = os.path.join(os.path.dirname(__file__), "../../../vendor/mujoco_models")
            baseXMLFile = os.path.join(baseXMLFilePath, 'ant.xml')
            print(baseXMLFile)
            tree = ET.parse(baseXMLFile)
            root = tree.getroot()

            # Writing the new value
            for tag_name, tag_identifier, attribute, value in zip(tag_names, tag_identifiers, attributes, values):
                self.change_attribute(root, tag_name, tag_identifier, attribute, value)

            if xml_file_name == None:
                xmlFile = os.path.join(baseXMLFilePath, 'tmp_ant.xml')
            else:
                xmlFile = os.path.join(baseXMLFilePath, xml_file_name)
            tree.write(xmlFile)

        FILE = 'tmp_ant.xml'

        MujocoEnv.__init__(self, 0.0, xmlFile)
        Serializable.__init__(self, 0, xmlFile)

        return xmlFile


    def my_init(self, tag_names=None, tag_identifiers=None, attributes=None, values=None, xml_file=None):
        self.change_model(tag_names, tag_identifiers, attributes, values, xml_file)

    def __init__(self, *args, **kwargs):
        self.velocity_dir = 'posx'
        self.use_gym_obs = False
        self.use_gym_reward = False

        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_forward_reward_rllab(self):
        #print(self.velocity_dir)
        
        comvel = self.get_body_comvel("torso")
        if self.velocity_dir == 'posx':
            forward_reward = comvel[0] - 2*abs(comvel[1])
        elif self.velocity_dir == 'posy':
            forward_reward = comvel[1] - 2*abs(comvel[0])
        elif self.velocity_dir == 'negx':
            forward_reward = -comvel[0] - 2*abs(comvel[1])
        elif self.velocity_dir == 'negy':
            forward_reward = -comvel[1] - 2*abs(comvel[0])
        else:
            raise NotImplementedError
        return forward_reward

    def get_forward_reward_gym(self, posbefore, posafter):
        vel = (posafter - posbefore) / 0.05

        if self.velocity_dir == 'posx':
            forward_reward = vel[0]
        elif self.velocity_dir == 'posy':
            forward_reward = vel[1] 
        elif self.velocity_dir == 'negx':
            forward_reward = -vel[0] 
        elif self.velocity_dir == 'negy':
            forward_reward = -vel[1] 
        else:
            raise NotImplementedError
        return forward_reward
     

    def get_current_obs(self):
        if self.use_gym_obs:
            return np.concatenate([
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ])
            
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)
        
    def step(self, action):
        posbefore = self.get_body_com("torso")
        self.forward_dynamics(action)
        posafter = self.get_body_com("torso")

        '''
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        '''
        if self.use_gym_reward:
            forward_reward = self.get_forward_reward_gym(posbefore, posafter)
            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-3 * np.sum(
                np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0

        else:    
            forward_reward = self.get_forward_reward_rllab()
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
            contact_cost = 0.5 * 1e-3 * np.sum(
                np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
            survive_reward = 0.05
        
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))





