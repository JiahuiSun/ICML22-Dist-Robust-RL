import contextlib
import os
import tempfile

import xml.etree.ElementTree as ET

import roboschool
from roboschool.gym_mujoco_walkers import (
    RoboschoolForwardWalkerMujocoXML, RoboschoolHalfCheetah, RoboschoolHopper, RoboschoolWalker2d, RoboschoolHumanoid, RoboschoolAnt
)
from roboschool.gym_pendulums import (
    RoboschoolInvertedPendulum, RoboschoolInvertedDoublePendulum)
from roboschool.gym_reacher import RoboschoolReacher

from .base import EnvBinarySuccessMixin
from .classic_control import uniform_exclude_inner
from .utils import EnvParamSampler

# Determine Roboschool asset location based on its module path.
ROBOSCHOOL_ASSETS = os.path.join(roboschool.__path__[0], 'mujoco_assets')


class RoboschoolTrackDistSuccessMixin(EnvBinarySuccessMixin):
    """Treat reaching certain distance on track as a success."""

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        TODO: 我猜作者的意思是，horizon最多1000，agent永远走不到终点，也就无法success
         x=100 correlates to the end of the track on Roboschool,
         but with the default 1000 max episode length most (all?) agents
         won't reach it (DD PPO2 Hopper reaches ~40), so we use something lower
         """
        target_dist = 20
        if self.robot_body.pose().xyz()[0] >= target_dist:
             #print("[SUCCESS]: xyz is {}, reached x-target {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
            return True
        else:
             #print("[NO SUCCESS]: xyz is {}, x-target is {}".format(
             #      self.robot_body.pose().xyz(), target_dist))
            return False


class RoboschoolXMLModifierMixin:
    """Mixin with XML modification methods."""
    @contextlib.contextmanager
    def modify_xml(self, asset):
        """Context manager allowing XML asset modifcation."""

        # tree = ET.ElementTree(ET.Element(os.path.join(ROBOSCHOOL_ASSETS, asset)))
        tree = ET.parse(os.path.join(ROBOSCHOOL_ASSETS, asset))
        yield tree

        # Create a new temporary .xml file
        # mkstemp returns (int(file_descriptor), str(full_path))
        fd, path = tempfile.mkstemp(suffix='.xml')
        # Close the file to prevent a file descriptor leak
        # See: https://www.logilab.org/blogentry/17873
        # We can also wrap tree.write in 'with os.fdopen(fd, 'w')' instead
        os.close(fd)
        tree.write(path)

        # Delete previous file before overwriting self.model_xml
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)
        self.model_xml = path

        # Original fix using mktemp:
        # mktemp (depreciated) returns str(full_path)
        #   modified_asset = tempfile.mktemp(suffix='.xml')
        #   tree.write(modified_asset)
        #   self.model_xml = modified_asset

    def __del__(self):
        """Deletes last remaining xml files after use"""
        # (Note: this won't ensure the final tmp file is deleted on a crash/SIGBREAK/etc.)
        if os.path.isfile(self.model_xml):
            os.remove(self.model_xml)


# =============== Reacher ===================
class ModifiableRoboschoolReacher(RoboschoolReacher, RoboschoolXMLModifierMixin,  RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_BODY_SIZE = 0.008
    RANDOM_UPPER_BODY_SIZE = 0.05

    RANDOM_LOWER_BODY_LENGTH = 0.1
    RANDOM_UPPER_BODY_LENGTH = 0.13

    size, length = 0.029, 0.115
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_BODY_SIZE, RANDOM_LOWER_BODY_LENGTH],
                              param_end=[RANDOM_UPPER_BODY_SIZE, RANDOM_UPPER_BODY_LENGTH])

    def _reset(self, new=True):
        with self.modify_xml('reacher.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('fromto', "0 0 0 " + str(self.length) + " 0 0")
                if elem.attrib['name'] == "link0":
                    elem.set('size', str(self.size))
            for elem in tree.iterfind('worldbody/body/body'):
                elem.set('pos', str(self.length) + " 0 0")
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('size', str(self.size))
        return super(ModifiableRoboschoolReacher, self)._reset()

    @property
    def parameters(self):
        return [self.size, self.length]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_BODY_SIZE, self.RANDOM_UPPER_BODY_SIZE,
                self.RANDOM_LOWER_BODY_LENGTH, self.RANDOM_UPPER_BODY_LENGTH]


class UniformReacher(ModifiableRoboschoolReacher):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformReacher, self)._reset()


class GaussianReacher(ModifiableRoboschoolReacher):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianReacher, self)._reset() 


# ============== InvertedPendulum ===============
class ModifiableRoboschoolInvertedPendulum(RoboschoolInvertedPendulum, RoboschoolXMLModifierMixin, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_CART_SIZE = 0.05
    RANDOM_UPPER_CART_SIZE = 0.25

    RANDOM_LOWER_POLE_LENGTH = 0.5
    RANDOM_UPPER_POLE_LENGTH = 2.0

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1

    RANDOM_LOWER_POLE_SIZE = 0.03
    RANDOM_UPPER_POLE_SIZE = 0.068

    RANDOM_LOWER_RAIL_SIZE = 0.01
    RANDOM_UPPER_RAIL_SIZE = 0.03

    length, cartsize = 1.25, 0.15
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_POLE_LENGTH, RANDOM_LOWER_CART_SIZE],
                              param_end=[RANDOM_UPPER_POLE_LENGTH, RANDOM_UPPER_CART_SIZE])

    def _reset(self, new=True):
        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
        return super(ModifiableRoboschoolInvertedPendulum, self)._reset()
    
    @property
    def parameters(self):
        return [self.length, self.cartsize]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH,
                self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE]


class UniformInvertedPendulum(ModifiableRoboschoolInvertedPendulum):
    def _reset(self, new=True):
        self.length, self.cartsize = self.sampler.uniform_sample().squeeze()
        return super(UniformInvertedPendulum, self)._reset()


class GaussianInvertedPendulum(ModifiableRoboschoolInvertedPendulum):
    def _reset(self, new=True):
        self.length, self.cartsize = self.sampler.gaussian_sample().squeeze()
        return super(GaussianInvertedPendulum, self)._reset() 


# =========== Ant ================
class ModifiableRoboschoolAnt(RoboschoolAnt, RoboschoolXMLModifierMixin,  RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 2.5

    RANDOM_LOWER_DAMPING = 0.5
    RANDOM_UPPER_DAMPING = 2.5
    RANDOM_LOWER_FOOTLEN = 0.1
    RANDOM_UPPER_FOOTLEN = 1.8

    density, friction = 1000, 0.8
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_DENSITY, RANDOM_LOWER_FRICTION],
                              param_end=[RANDOM_UPPER_DENSITY, RANDOM_UPPER_FRICTION])

    def _reset(self, new=True):
        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolAnt, self)._reset()

    @property
    def parameters(self):
        return [self.density, self.friction]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY,
                self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION]


class UniformAnt(ModifiableRoboschoolAnt):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformAnt, self)._reset()


class GaussianAnt(ModifiableRoboschoolAnt):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianAnt, self)._reset()     


class RandomNormalFDAnt(ModifiableRoboschoolAnt):
    def randomize_env(self):
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.damping = self.np_random.uniform(
            self.RANDOM_LOWER_DAMPING, self.RANDOM_UPPER_DAMPING)

        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/joint'):
                elem.set('damping', str(self.damping))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalFDAnt, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalFDAnt, self).parameters
        parameters.update(
            {'damping': self.damping, 'friction': self.friction, })
        return parameters


class RandomNormalFootAnt(ModifiableRoboschoolAnt):
    def randomize_env(self):
        # self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        # self.friction = self.np_random.uniform(self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.damping = self.np_random.uniform(self.RANDOM_LOWER_DAMPING, self.RANDOM_UPPER_DAMPING)
        self.front_left_foot = self.np_random.uniform(
            self.RANDOM_LOWER_FOOTLEN, self.RANDOM_UPPER_FOOTLEN)
        self.front_right_foot = - \
            self.np_random.uniform(
                self.RANDOM_LOWER_FOOTLEN, self.RANDOM_UPPER_FOOTLEN)

        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('ant.xml') as tree:
            # for elem in tree.iterfind('default/joint'):
            #     elem.set('damping', str(self.damping))
            # for elem in tree.iterfind('default/geom'):
            #     elem.set('friction', str(self.friction) + ' .1 .1')
            for elem in tree.iterfind('worldbody/body/body/body/body/geom'):
                # print(elem.attrib['name'])
                if elem.attrib['name'] == "left_ankle_geom":
                    # print(elem.attrib['name'])
                    elem.set('fromto', "0.0 0.0 0.0 " +
                             str(self.front_left_foot)+" 0.4 0.0")
                elif elem.attrib['name'] == "right_ankle_geom":
                    # print(elem.attrib['name'])
                    elem.set('fromto', "0.0 0.0 0.0 " +
                             str(self.front_right_foot) + " 0.4 0.0")

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomNormalFootAnt, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalFootAnt, self).parameters
        parameters.update({'front_left_foot': self.front_left_foot,
                           'front_right_foot': self.front_right_foot, })
        return parameters


# =============== Humanoid =============
class ModifiableRoboschoolHumanoid(RoboschoolHumanoid, RoboschoolXMLModifierMixin,  RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    density, friction = 1000, 0.8
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_DENSITY, RANDOM_LOWER_FRICTION],
                              param_end=[RANDOM_UPPER_DENSITY, RANDOM_UPPER_FRICTION])

    def _reset(self, new=True):
        with self.modify_xml('humanoid_symmetric.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHumanoid, self)._reset()

    @property
    def parameters(self):
        return [self.density, self.friction]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY,
                self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION]


class UniformHumanoid(ModifiableRoboschoolHumanoid):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHumanoid, self)._reset()


class GaussianHumanoid(ModifiableRoboschoolHumanoid):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHumanoid, self)._reset()


class RandomFrictionHumanoid(ModifiableRoboschoolHumanoid):
    def randomize_env(self):
        # self.density = self.np_random.uniform(self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        # self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('humanoid_symmetric.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomFrictionHumanoid, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomFrictionHumanoid, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomFrictionHumanoidEval(ModifiableRoboschoolHumanoid):

    def set_envparam(self, friction):
        self.friction = friction
        with self.modify_xml('humanoid_symmetric.xml') as tree:
            # for elem in tree.iterfind('worldbody/body/geom'):
            #     elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            with self.modify_xml('humanoid_symmetric.xml') as tree:
                # for elem in tree.iterfind('worldbody/body/geom'):
                #     elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')
        return super(RandomFrictionHumanoidEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomFrictionHumanoidEval, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


# ================ Walker2d =====================
class ModifiableRoboschoolWalker2d(RoboschoolWalker2d, RoboschoolXMLModifierMixin, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    # RANDOM_LOWER_POWER = 0.7
    # RANDOM_UPPER_POWER = 1.1
    # EXTREME_LOWER_POWER = 0.5
    # EXTREME_UPPER_POWER = 1.3

    density, friction = 1000, 0.8
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_DENSITY, RANDOM_LOWER_FRICTION],
                              param_end=[RANDOM_UPPER_DENSITY, RANDOM_UPPER_FRICTION])

    def reset(self, new=True):
        with self.modify_xml('walker2d.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density) + ' .1 .1')
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolWalker2d, self).reset()

    def set_envparam(self, new_density, new_friction):
        self.density = new_density
        self.friction = new_friction
        return True

    @property
    def parameters(self):
        return [self.density, self.friction]

    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY,
                self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION]


class UniformWalker2d(ModifiableRoboschoolWalker2d):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformWalker2d, self)._reset()


class GaussianWalker2d(ModifiableRoboschoolWalker2d):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianWalker2d, self)._reset()


# ============== Half Cheetah =================
class ModifiableRoboschoolHalfCheetah(RoboschoolHalfCheetah, RoboschoolXMLModifierMixin, RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 2.25

    RANDOM_LOWER_POWER = 0.7
    RANDOM_UPPER_POWER = 1.1
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.3

    density, friction = 1000, 0.8
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_DENSITY, RANDOM_LOWER_FRICTION],
                              param_end=[RANDOM_UPPER_DENSITY, RANDOM_UPPER_FRICTION])

    def _reset(self, new=True):
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHalfCheetah, self)._reset()

    def set_envparam(self, new_density, new_friction):
        self.density = new_density
        self.friction = new_friction

    @property
    def parameters(self):
        return [self.density, self.friction]

    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY,
                self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION]


class UniformHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHalfCheetah, self)._reset()


class GaussianHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHalfCheetah, self)._reset()
 

class StrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=1.3)

    @property
    def parameters(self):
        parameters = super(StrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'half_cheetah.xml', 'torso', action_dim=6, obs_dim=26, power=0.5)

    @property
    def parameters(self):
        parameters = super(WeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomWeakHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.EXTREME_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHalfCheetah, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.density = 500
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(LightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHalfCheetah, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=6, obs_dim=26, power=0.9)

    @property
    def parameters(self):
        parameters = super(RoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHalfCheetah, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHalfcheetahEvaluate(ModifiableRoboschoolHalfCheetah):
    def set_envparam(self, density, friction):
        self.density = density
        self.friction = friction

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in reset ==========')
            print(self.density)
            print(self.friction)
            with self.modify_xml('half_cheetah.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomNormalHalfcheetahEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHalfcheetahEvaluate, self).parameters
        parameters.update(
            {'density': self.density, 'friction': self.friction, })


class RandomExtremeHalfCheetah(ModifiableRoboschoolHalfCheetah):

    def randomize_env(self):
        '''
        # self.armature = self.np_random.uniform(0.2, 0.5)
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
                                             self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
                                              self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
                                           self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHalfCheetah, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHalfCheetah, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


# =========== Hopper ===============
class ModifiableRoboschoolHopper(RoboschoolHopper, RoboschoolXMLModifierMixin,  RoboschoolTrackDistSuccessMixin):
    RANDOM_LOWER_DENSITY = 750
    RANDOM_UPPER_DENSITY = 1250
    EXTREME_LOWER_DENSITY = 500
    EXTREME_UPPER_DENSITY = 1500

    RANDOM_LOWER_FRICTION = 0.5
    RANDOM_UPPER_FRICTION = 1.1
    EXTREME_LOWER_FRICTION = 0.2
    EXTREME_UPPER_FRICTION = 1.4

    RANDOM_LOWER_POWER = 0.6
    RANDOM_UPPER_POWER = 0.9
    EXTREME_LOWER_POWER = 0.4
    EXTREME_UPPER_POWER = 1.1

    density, friction = 1000, 0.8
    sampler = EnvParamSampler(param_start=[RANDOM_LOWER_DENSITY, RANDOM_LOWER_FRICTION],
                              param_end=[RANDOM_UPPER_DENSITY, RANDOM_UPPER_FRICTION])

    def _reset(self, new=True):
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHopper, self)._reset()

    @property
    def parameters(self):
        return [self.density, self.friction]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY,
                self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION]


class UniformHopper(ModifiableRoboschoolHopper):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHopper, self)._reset()


class GaussianHopper(ModifiableRoboschoolHopper):
    def _reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHopper, self)._reset()


class StrongHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=1.1)

    @property
    def parameters(self):
        parameters = super(StrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class WeakHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(
            self, 'hopper.xml', 'torso', action_dim=3, obs_dim=15, power=0.4)

    @property
    def parameters(self):
        parameters = super(WeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomStrongHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomStrongHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomStrongHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class RandomWeakHopper(ModifiableRoboschoolHopper):
    def randomize_power(self):
        self.power = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

    def _reset(self, new=True):
        if new:
            self.randomize_power()
        return super(RandomWeakHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomWeakHopper, self).parameters
        parameters.update({'power': self.power, })
        return parameters


class HeavyTorsoHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 1500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(HeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class LightTorsoHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        self.density = 500
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(LightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomHeavyTorsoHopper(ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = self.np_random.uniform(
            self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomHeavyTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomHeavyTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class RandomLightTorsoHopper(ModifiableRoboschoolHopper):
    def randomize_mass(self):
        self.density = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))

    def _reset(self, new=True):
        if new:
            self.randomize_mass()
        return super(RandomLightTorsoHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomLightTorsoHopper, self).parameters
        parameters.update({'density': self.density, })
        return parameters


class SlipperyJointsHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 0.2
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(SlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RoughJointsHopper(ModifiableRoboschoolHopper):
    def __init__(self):
        self.friction = 1.4
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction)+' .1 .1')

        RoboschoolForwardWalkerMujocoXML.__init__(
            self, self.model_xml, 'torso', action_dim=3, obs_dim=15, power=0.75)

    @property
    def parameters(self):
        parameters = super(RoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomRoughJointsHopper(ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomRoughJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomRoughJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHopper(ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomSlipperyJointsHopper(ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = uniform_exclude_inner(self.np_random.uniform, self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomExtremeSlipperyJointsHopper(ModifiableRoboschoolHopper):
    def randomize_friction(self):
        self.friction = self.np_random.uniform(
            self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION)

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_friction()
        return super(RandomExtremeSlipperyJointsHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeSlipperyJointsHopper, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomExtremeSlipperyJointsHopperEval(ModifiableRoboschoolHopper):
    def set_envparam(self, friction):
        self.friction = friction

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            with self.modify_xml('hopper.xml') as tree:
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomExtremeSlipperyJointsHopperEval, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(
            RandomExtremeSlipperyJointsHopperEval, self).parameters
        parameters.update({'friction': self.friction, })
        return parameters


class RandomNormalHopperEvaluate(ModifiableRoboschoolHopper):
    def set_envparam(self, density, friction):
        self.density = density
        self.friction = friction

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            # self.randomize_env()
            print('==============in reset ==========')
            print(self.density)
            print(self.friction)
            with self.modify_xml('hopper.xml') as tree:
                for elem in tree.iterfind('worldbody/body/geom'):
                    elem.set('density', str(self.density))
                for elem in tree.iterfind('default/geom'):
                    elem.set('friction', str(self.friction) + ' .1 .1')

        return super(RandomNormalHopperEvaluate, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomNormalHopperEvaluate, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters


class RandomExtremeHopper(ModifiableRoboschoolHopper):

    def randomize_env(self):
        '''
        self.density = self.np_random.uniform(self.LOWER_DENSITY, self.UPPER_DENSITY)
        self.friction = self.np_random.uniform(self.LOWER_FRICTION, self.UPPER_FRICTION)
        self.power = self.np_random.uniform(self.LOWER_POWER, self.UPPER_POWER)
        '''

        self.density = uniform_exclude_inner(self.np_random.uniform,
                                             self.EXTREME_LOWER_DENSITY, self.EXTREME_UPPER_DENSITY,
                                             self.RANDOM_LOWER_DENSITY, self.RANDOM_UPPER_DENSITY)
        self.friction = uniform_exclude_inner(self.np_random.uniform,
                                              self.EXTREME_LOWER_FRICTION, self.EXTREME_UPPER_FRICTION,
                                              self.RANDOM_LOWER_FRICTION, self.RANDOM_UPPER_FRICTION)
        self.power = uniform_exclude_inner(self.np_random.uniform,
                                           self.EXTREME_LOWER_POWER, self.EXTREME_UPPER_POWER,
                                           self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)

        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')

    def _reset(self, new=True):
        if new:
            self.randomize_env()
        return super(RandomExtremeHopper, self)._reset(new)

    @property
    def parameters(self):
        parameters = super(RandomExtremeHopper, self).parameters
        parameters.update(
            {'power': self.power, 'density': self.density, 'friction': self.friction, })
        return parameters
