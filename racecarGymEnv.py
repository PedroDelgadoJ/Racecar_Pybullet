import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import racecar
import random
from pybullet_utils import bullet_client as bc

from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class RacecarGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=os.getcwd(),
               actionRepeat=50,
               isEnableSelfCollision=True,
               renders=False):
    print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._ballUniqueId = -1
    self._envStepCounter = 0
    self._renders = renders
    if self._renders:
      self._p = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._p = bc.BulletClient()

    self.seed()
    observationDim = 2 
    observation_high = np.ones(observationDim) * 1000  #np.inf
    self.action_space = spaces.Discrete(9)
    self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)

  def reset(self):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)
    stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot, "Urdf_and_meshes/stadium.sdf"))

    dist = 5 + 2. * random.random()
    ang = 2. * 3.1415925438 * random.random()

    ballx = dist * math.sin(ang)
    bally = dist * math.cos(ang)
    ballz = 1

    self._ballUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "Urdf_and_meshes/sphere2.urdf"),
                                          [ballx, bally, ballz])
    
    self._p.setGravity(0, 0, -10)
    self._racecar = racecar.Racecar(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0

    for i in range(1000):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    self._p = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = []  #self._racecar.getObservation()
    carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    ballpos, ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
    invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
    ballPosInCar, ballOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)

    self._observation.extend([ballPosInCar[0], ballPosInCar[1]])
    return self._observation

  def step(self, action):
    if (self._renders):
      basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

    velocidades = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    angulos = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
    velocidad = velocidades[action]
    angulo = angulos[action]
    realaction = [velocidad, angulo]
    

    self._racecar.applyAction(realaction)

    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()

      if self._termination():
        break
      self._envStepCounter += 1

    reward = self._reward()
    done = self._termination()

    return np.array(self._observation), reward, done, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    return self._envStepCounter > 1000

  def _reward(self):
    closestPoints = self._p.getClosestPoints(self._racecar.racecarUniqueId, self._ballUniqueId,
                                             10000)

    numPt = len(closestPoints)
    reward = -1000
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8]
      #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
