import pybullet as p 
import time 
import pybullet_data
import os
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import math
import cv2
import threading
import collections
import traceback

FNAME=os.path.join(os.path.dirname(__file__),"robot.urdf")

class Robot:
    def __init__(self,connId=None,startPos=[+0,+0,1.435],startOri=[0,0,0]):
        self.TARGET_VELOCITY=3.0
        self.TARGET_FORCE=1000
        self.RENDER_WIDTH=100
        self.RENDER_HEIGHT=100
        self.POSITION_GAIN=0.01
        self.VELOCITY_GAIN=1
        self.connId=connId
        self.startPos=startPos
        self.startOri=startOri
        self.infopart=[]
        self.senspart=[]
        self.loadRobot()
        self.coff=[-0.08,0,0]

    def loadRobot(self):
        self.robId=p.loadURDF(FNAME,self.startPos,p.getQuaternionFromEuler(self.startOri),globalScaling=1)
        self.joints={}
        #p.enableJointForceTorqueSensor(self.robId,0,1)
        p.enableJointForceTorqueSensor(self.robId,1,1)
        for i in range(p.getNumJoints(self.robId)):
            info=p.getJointInfo(self.robId,i)
            name=info[1].decode()
            if info[2]!=p.JOINT_FIXED:
                d=self.joints[name]={}
                d["index"]=info[0]
                d["name"]=name
                d["type"]=info[2]
                d["target_pos"]=0
                self.infopart.append((name,info[0]))
                #p.setJointMotorControl2(self.robId,i,p.POSITION_CONTROL,targetPosition=0,targetVelocity=self.TARGET_VELOCITY,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)
                p.setJointMotorControl(self.robId,i,p.POSITION_CONTROL,0,self.TARGET_FORCE)#,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)        
                self.loadJointState(d)
            elif name in ("right_foot_x_h","left_foot_x_l","right_hand_x_h","left_hand_x_l","head_z_h"):
                p.enableJointForceTorqueSensor(self.robId,info[0],1)
                self.senspart.append((name,info[0]))
                #print (self.senspart)
            else:
                pass
                #print (info)

    def setAllJoints(self):
        self.getState()
        for name,idx in self.infopart:
            d=self.joints[name]
            #p.setJointMotorControl2(self.robId,idx,p.POSITION_CONTROL,targetPosition=0,targetVelocity=abs(d["pos"]-d["target_pos"])*self.TARGET_VELOCITY,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)
            p.setJointMotorControl(self.robId,idx,p.POSITION_CONTROL,d["target_pos"],self.TARGET_FORCE)#,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)        

    def _mkImage(self,p0,p1,up):
        vmatrix=p.computeViewMatrix(p0,p1,up)
        pmatrix=p.computeProjectionMatrixFOV(fov=60,aspect=float(self.RENDER_WIDTH)/self.RENDER_HEIGHT,nearVal=0.1,farVal=100)
        img=p.getCameraImage(width=self.RENDER_WIDTH,height=self.RENDER_HEIGHT,viewMatrix=vmatrix,projectionMatrix=pmatrix)
        img=cv2.cvtColor(img[2],cv2.COLOR_RGBA2GRAY)
        return img

    def getEyeImages(self):
        head=p.getLinkState(self.robId,0)
        #link_left=p.getLinkState(self.robId,1)
        #link_right=p.getLinkState(self.robId,2)
        m=np.array(p.getMatrixFromQuaternion(head[1])).reshape((3,3))
        c=head[0]-np.dot(m,self.coff)
        delta=np.dot(m,[0.07,0,0])
        p0=np.array(c)+delta
        q0=np.array(c)-delta
        p1=q1=c+np.dot(m,[0,5,0])
        up=np.dot(m,[0,0,1])
        ri=self._mkImage(p0,p1,up)
        li=self._mkImage(q0,q1,up)
        return (li,ri)
        

    def loadJointState(self,d):
        idx=d["index"]
        state=p.getJointState(self.robId,idx)
        d["pos"]=state[0]
        d["vel"]=state[1]
        d["force"]=state[2]
        d["torque"]=state[3]        

    def setServoPos(self,name,pos):
        d=self.joints[name]
        if pos>math.pi:
            d["target_pos"]=math.pi
        elif pos<-math.pi:
            d["target_pos"]=-math.pi
        else:
            d["target_pos"]=pos
        servoidx=d["index"]
        p.setJointMotorControl(self.robId,servoidx,p.POSITION_CONTROL,pos,self.TARGET_FORCE)#,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)        

    def setServos(self,pos):
        for i,(name,idx) in enumerate(self.infopart):
            d=self.joints[name]
            #p.setJointMotorControl2(self.robId,idx,p.POSITION_CONTROL,targetPosition=0,targetVelocity=abs(d["pos"]-d["target_pos"])*self.TARGET_VELOCITY,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)
            if pos[i]>math.pi:
                d["target_pos"]=math.pi
            elif pos[i]<-math.pi:
                d["target_pos"]=-math.pi
            else:
                d["target_pos"]=pos[i]
            p.setJointMotorControl(self.robId,idx,p.POSITION_CONTROL,d["target_pos"],self.TARGET_FORCE)#,force=self.TARGET_FORCE,positionGain=self.POSITION_GAIN,velocityGain=self.VELOCITY_GAIN)

    def getServos(self):
        ret=[]
        for i,(name,idx) in enumerate(self.infopart):
            d=self.joints[name]
            ret.append(d["target_pos"])
        return ret

    def getState(self):
        ids=[]
        names=[]
        for name,idx in self.infopart:
            ids.append(idx)
            names.append(name)
        states=p.getJointStates(self.robId,ids)
        ret=[]
        for id,name,state in zip(ids,names,states):
            if name in self.joints:
                d=self.joints[name]
                d["pos"]=state[0]
                d["vel"]=state[1]
                d["force"]=state[2]
                d["torque"]=state[3]
                ret.append(state[0])
        return ret

    def getSensors(self):
        ids=[]
        names=[]
        for name,idx in self.senspart:
            ids.append(idx)
            names.append(name)
        states=p.getJointStates(self.robId,ids)
        ret=[]
        for id,name,state in zip(ids,names,states):
            ret.extend(state[2])
        return ret

    def calcReward(self,target):
        base_pos,orn=p.getBasePositionAndOrientation(self.robId)
        self.base_pos=base_pos
        ret=0
        r0=5*(2-pow(base_pos[2]-1.45,2))
        r1=abs(base_pos[0]-target[0])
        r2=abs(base_pos[1]-target[1])
        ret=r0+(1-math.sqrt(r1*r1+r2*r2))
        #print (r0,r1,r2,ret)
        return ret
    

class World:
    def __init__(self):
        self.physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version, p.GUI for visible
        self.robot=None
        self.RENDER_WIDTH=640
        self.RENDER_HEIGHT=480
        p.resetSimulation(self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally 
        p.setGravity(0,0,-5)
        p.setTimeStep(0.01)
        self.user_cam_dist=3
        self.user_cam_yaw=0
        self.user_cam_pitch=0
        self.user_cam_roll=0
        self.planeId = p.loadURDF("plane.urdf",[0,0,0])
        pos=np.random.randn(3)*5
        pos[2]=3
        if abs(pos[0])<1:
            pos[0]=1
        if abs(pos[1])<1:
            pos[0]=1
        self.sphere=p.loadURDF("sphere2.urdf",pos)
        self.robot=Robot()
        self.stateid=p.saveState()
        self.reset()
    
    def reset(self):
        p.restoreState(self.stateid)
        return self.getState()

    def getState(self):
        s0=self.robot.getState()
        s1=self.robot.getSensors()
        i0,i1=self.robot.getEyeImages()
        return s0,s1,i0,i1

    def render(self):
        base_pos,orn=p.getBasePositionAndOrientation(self.robot.robId)
        view_matrix=p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self.user_cam_dist,  
            yaw=self.user_cam_yaw,
            pitch=self.user_cam_pitch,
            roll=self.user_cam_roll,
            upAxisIndex=2)
        proj_matrix=p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.RENDER_WIDTH)/self.RENDER_HEIGHT,
            nearVal=0.1,farVal=100)
        (_, _, px, _, _) = p.getCameraImage(
            width=self.RENDER_WIDTH,height=self.RENDER_HEIGHT,viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        return px[:,:,:3]

    def calcReward(self):
        base_pos,orn=p.getBasePositionAndOrientation(self.sphere)
        return self.robot.calcReward(base_pos)

class RenderThread(threading.Thread):
    name="RenderThread"
    def  run(self):
        try:
            self.calls=collections.deque()
            self.keys=collections.deque([],1000)
            self.delay=100
            while True:
                while len(self.calls)>0:
                    f,args,kwargs=self.calls.popleft()
                    try:
                        f(*args,**kwargs)
                    except:
                        traceback.print_exc()
                c=cv2.waitKey(self.delay)
                if (c!=-1):
                    self.keys.append(c)
        except:
            traceback.print_exc()
            del self.delay
            del self.keys
    def waitKey(self,delay):
        self.delay=delay
        try:
            return self.keys.popleft()
        except:
            return -1

class HumanRobotEnvBase(gym.Env):
    metadata={'render.modes': ['human'],
                'video.frames_per_second':50}

    MAX_STEPS=1000
    
    def __init__(self):
        self.world=World()
        flag=True
        for t in threading.enumerate():
            if t.__class__.__name__=="RenderThread" and hasattr(t,"keys") and hasattr(t,"delay"):
                flag=False
                self.rt=t
        if flag:
            self.rt=RenderThread()
            self.rt.start()
        s0,s1,i0,i1=self.world.getState()
        self.s0_shape=(len(s0),)
        self.observation_space=spaces.Dict({
            "servos":spaces.Box(low=-math.pi,high=math.pi,dtype=np.float,shape=(len(s0),)),
            "sensors":spaces.Box(low=-np.inf,high=np.inf,dtype=np.float,shape=(len(s1),)),
            "eye_left":spaces.Box(low=0,high=255,dtype=np.uint8,shape=i0.shape),
            "eye_right":spaces.Box(low=0,high=255,dtype=np.uint8,shape=i1.shape)
            })
        self.steps=0

    def getState(self):
        s0,s1,i0,i1=self.world.getState()
        d=collections.OrderedDict()
        d["servos"]=np.array(s0)
        d["sensors"]=np.array(s1)
        d["eye_left"]=i0
        d["eye_right"]=i1
        return d

    def waitKey(self,delay):
        return self.rt.waitKey(delay)

    def reset(self):
        self.world.reset()
        self.steps=0
        return self.getState()

    def render(self, mode='human', close=False):
        img=self.world.render()
        le,re=self.world.robot.getEyeImages()
        self.rt.calls.append((cv2.imshow,("robot",img),{}))
        self.rt.calls.append((cv2.imshow,("left",le),{}))
        self.rt.calls.append((cv2.imshow,("right",re),{}))
        if close:
            self.rt.calls.append((cv2.destroyWindow,("robot",),{}))

    def _step(self,ra):
        p.stepSimulation()
        self.steps+=1
        return self.getState(),self.world.calcReward()+ra,(self.steps>self.MAX_STEPS or self.world.robot.base_pos[2]<0.2),{}

    def step(self,action):
        p=self.world.robot.getServos()
        self.world.robot.setServos(action)
        r=0
        for i in range(len(p)):
            r+=abs(p[i]-action[i])
        return self._step(-r)

class HumanRobotEnvContinuous(HumanRobotEnvBase):
    metadata={'render.modes': ['human'],
                'video.frames_per_second':50}
    
    def __init__(self):
        HumanRobotEnvBase.__init__(self)
        self.action_space=spaces.Box(-math.pi,high=math.pi,dtype=np.float,shape=self.s0_shape)



class HumanRobotEnvDiscrete(HumanRobotEnvBase):
    metadata={'render.modes': ['human'],
                'video.frames_per_second':50}

    STEP_SIZE=math.pi/100

    def __init__(self):
        HumanRobotEnvBase.__init__(self)
        self.action_space=spaces.MultiDiscrete([3]*self.s0_shape[0])

    def step(self,action):
        p=self.world.robot.getServos()
        r=0
        for i in range(len(p)):
            if action[i]==1:
                r+=self.STEP_SIZE
                if p[i]<math.pi:
                    p[i]+=self.STEP_SIZE
            elif action[i]==2:
                r+=self.STEP_SIZE
                if p[i]>-math.pi:
                    p[i]-=self.STEP_SIZE
            else:
                pass
        self.world.robot.setServos(p)
        return self._step()


    
if __name__=="__main__":
    pass
    #import gym
    #e=gym.make('HRobotEnvContinous-v0')
    """
    import cv2
    w=World()

    def mkSetPos(name):
        def setPos(deg):
            print (name,deg)
            w.robot.setServoPos(name,(deg-180)*math.pi/180)
        return setPos
    
    #img=w.render()
    #cv2.imshow("robot",img)
    positions={}
    cv2.namedWindow("joints", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("joints", 500, 500)
    for i,(name,idx) in enumerate(w.robot.infopart):
        f=mkSetPos(name)
        cv2.createTrackbar(name,"joints",180,360,f)
    w.robot.setAllJoints()
    while True:
        p.stepSimulation()
        res=np.zeros((480,640*3,3),dtype=np.uint8)
        li,ri=w.robot.getEyeImages()
        res[:,0:640,:]=img=w.render()
        res[:,640:2*640,:]=li
        res[:,2*640:3*640,:]=ri
        w.calcReward()
        state=w.getState()
        print (state)
        #print (img.shape)
        
        #cv2.imshow("left_eye",li)
        #cv2.imshow("right_eye",ri)
        #cv2.imshow("robot",img)
        cv2.imshow("img",res)
        c=cv2.waitKey(10)
        if (c&0xFF)==27:
            break
        elif c==-1:
            pass
        elif c==113:
            pos=positions["neck_c"]=positions.get("neck_c",0)+10
            w.robot.setServoPos("neck_c",pos*math.pi/180)
            print (pos)
        elif c==97:
            pos=positions["neck_c"]=positions.get("neck_c",0)-10
            w.robot.setServoPos("neck_c",pos*math.pi/180)
            print (pos)
        elif c==119:
            pos=positions["neck_b"]=positions.get("neck_b",0)+10
            w.robot.setServoPos("neck_b",pos*math.pi/180)
            print (pos)
        elif c==115:
            pos=positions["neck_b"]=positions.get("neck_b",0)-10
            w.robot.setServoPos("neck_b",pos*math.pi/180)
            print (pos)
        elif c==101:
            pos=positions["neck_a"]=positions.get("neck_a",0)+10
            w.robot.setServoPos("neck_a",pos*math.pi/180)
            print (pos)
        elif c==100:
            pos=positions["neck_a"]=positions.get("neck_a",0)-10
            w.robot.setServoPos("neck_a",pos*math.pi/180)
            print (pos)
        elif c==102:
            w.robot.coff[0]+=0.01
            print (w.robot.coff)
        elif c==114:
            w.robot.coff[0]-=0.01
            print (w.robot.coff)
        else:
            print (c,positions)
        
    #    print (w.getState())

"""
