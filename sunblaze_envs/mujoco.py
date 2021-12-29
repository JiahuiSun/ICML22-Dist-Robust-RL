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

    def reset(self, new=True):
        with self.modify_xml('reacher.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('fromto', "0 0 0 " + str(self.length) + " 0 0")
                if elem.attrib['name'] == "link0":
                    elem.set('size', str(self.size))
            for elem in tree.iterfind('worldbody/body/body'):
                elem.set('pos', str(self.length) + " 0 0")
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('size', str(self.size))
        return super(ModifiableRoboschoolReacher, self).reset()
    
    def set_envparam(self, new_size, new_length):
        self.size = new_size
        self.length = new_length
        return True

    @property
    def parameters(self):
        return [self.size, self.length]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_BODY_SIZE, self.RANDOM_UPPER_BODY_SIZE,
                self.RANDOM_LOWER_BODY_LENGTH, self.RANDOM_UPPER_BODY_LENGTH]


class UniformReacher(ModifiableRoboschoolReacher):
    def reset(self, new=True):
        self.size, self.length = self.sampler.uniform_sample().squeeze()
        return super(UniformReacher, self).reset()


class GaussianReacher(ModifiableRoboschoolReacher):
    def reset(self, new=True):
        self.size, self.length = self.sampler.gaussian_sample().squeeze()
        return super(GaussianReacher, self).reset() 


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

    def reset(self, new=True):
        with self.modify_xml('inverted_pendulum.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('size', str(self.cartsize) + ' ' + str(self.cartsize))
            for elem in tree.iterfind('worldbody/body/body/geom'):
                elem.set('fromto', "0 0 0 0.001 0 " + str(self.length))
        return super(ModifiableRoboschoolInvertedPendulum, self).reset()
    
    def set_envparam(self, new_length, new_cartsize):
        self.length = new_length
        self.cartsize = new_cartsize
        return True

    @property
    def parameters(self):
        return [self.length, self.cartsize]
    
    @property
    def lower_upper_bound(self):
        return [self.RANDOM_LOWER_POLE_LENGTH, self.RANDOM_UPPER_POLE_LENGTH,
                self.RANDOM_LOWER_CART_SIZE, self.RANDOM_UPPER_CART_SIZE]


class UniformInvertedPendulum(ModifiableRoboschoolInvertedPendulum):
    def reset(self, new=True):
        self.length, self.cartsize = self.sampler.uniform_sample().squeeze()
        return super(UniformInvertedPendulum, self).reset()


class GaussianInvertedPendulum(ModifiableRoboschoolInvertedPendulum):
    def reset(self, new=True):
        self.length, self.cartsize = self.sampler.gaussian_sample().squeeze()
        return super(GaussianInvertedPendulum, self).reset() 


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

    def reset(self, new=True):
        with self.modify_xml('ant.xml') as tree:
            for elem in tree.iterfind('default/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolAnt, self).reset()

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


class UniformAnt(ModifiableRoboschoolAnt):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformAnt, self).reset()


class GaussianAnt(ModifiableRoboschoolAnt):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianAnt, self).reset()     


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

    def reset(self, new=True):
        with self.modify_xml('humanoid_symmetric.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHumanoid, self).reset()

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


class UniformHumanoid(ModifiableRoboschoolHumanoid):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHumanoid, self).reset()


class GaussianHumanoid(ModifiableRoboschoolHumanoid):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHumanoid, self).reset()


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
    def reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformWalker2d, self).reset()


class GaussianWalker2d(ModifiableRoboschoolWalker2d):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianWalker2d, self).reset()


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

    def reset(self, new=True):
        with self.modify_xml('half_cheetah.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHalfCheetah, self).reset()

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


class UniformHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHalfCheetah, self).reset()


class GaussianHalfCheetah(ModifiableRoboschoolHalfCheetah):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHalfCheetah, self).reset()
 

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

    def reset(self, new=True):
        with self.modify_xml('hopper.xml') as tree:
            for elem in tree.iterfind('worldbody/body/geom'):
                elem.set('density', str(self.density))
            for elem in tree.iterfind('default/geom'):
                elem.set('friction', str(self.friction) + ' .1 .1')
        return super(ModifiableRoboschoolHopper, self).reset()

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


class UniformHopper(ModifiableRoboschoolHopper):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.uniform_sample().squeeze()
        return super(UniformHopper, self).reset()


class GaussianHopper(ModifiableRoboschoolHopper):
    def reset(self, new=True):
        self.density, self.friction = self.sampler.gaussian_sample().squeeze()
        return super(GaussianHopper, self).reset()
