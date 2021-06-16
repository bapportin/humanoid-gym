import collections


class LinkRef:
    def __init__(self,link,a,d):
        self.link=link
        self.a=a
        self.d=d
        
    def __repr__(self):
        return ''+self.link.name+"_"+self.a+"_"+self.d
        
    def __str__(self):
        return rept(self)

class Link:
    def __init__(self,name,x,y,z,dx=0.01,dy=0.01,dz=0.01,up=('z','h')):
        self.name=name
        self.x,self.y,self.z=x,y,z
        self.dx,self.dy,self.dz=dx,dy,dz
        self.up=up
        self.offsets={}

    def mkXml(self,used=None):
        ret=[]
        ret.append('<link name="'+self.name+'"><collision>')
        ret.append('<geometry><box size="'+str(self.x)+' '+str(self.y)+' '+str(self.z)+'"/></geometry>')
        ret.append('</collision></link>\n')
        for i in range(3):
            for j in range(2):
                np=("xyz"[i])+'_'+("lh"[j])
                name=self.name+'_'+np
                coord=self.offsets.get(np,[0,0,0])
                coord[i]+=[-0.5,0.5][j]*([self.x+self.dx,self.y+self.dy,self.z+self.dy][i])
                if used is None or name in used:
                    ret.append('<link name="'+name+'"/>\n')
                    if "xyz"[i]==self.up[0] and "lh"[j]==self.up[1]:
                        ret.append('<joint name="'+name+'" type="fixed"><origin xyz="'+(" ".join(map(lambda x: str(-x),coord)))+'"/><parent link="'+name+'"/><child link="'+self.name+'"/></joint>\n')
                    else:
                        ret.append('<joint name="'+name+'" type="fixed"><origin xyz="'+(" ".join(map(str,coord)))+'"/><parent link="'+self.name+'"/><child link="'+name+'"/></joint>\n')
        return "".join(ret)

    def getXMin(self):
        return LinkRef(self,'x','l')

    def getXMax(self):
        return LinkRef(self,'x','h')

    def getYMin(self):
        return LinkRef(self,'y','l')

    def getYMax(self):
        return LinkRef(self,'y','h')

    def getZMin(self):
        return LinkRef(self,'z','l')

    def getZMax(self):
        return LinkRef(self,'z','h')


class Joint:
    def __init__(self,name,parent,child,axis=(1,0,0),typus="revolute"):
        self.name=name
        self.axis=axis
        self.parent,self.child=parent,child
        self.typus=typus

    def mkXml(self):
        ret=[]
        ret.append('<joint name="'+self.name+'" type="'+self.typus+'"><axis xyz="'+(' '.join(map(str,self.axis)))+'"/><parent link="'+repr(self.parent)+'"/><child link="'+repr(self.child)+'"/><limit effort="1000.0" velocity="5"/>"</joint>\n')
        return "".join(ret)


class Robot:
    def __init__(self,name):
        self.name=name
        self.links=collections.OrderedDict()#{}
        self.joints=collections.OrderedDict()#{}
        self.used=set()

    def addLink(self,name,x,y,z,d=0.01,up=('z','h')):
        ret=self.links[name]=Link(name,x,y,z,d,d,d,up)
        return ret

    def addJoint(self,name,src,sx,dst,dx,axis,typus="revolute"):
        #name=src.name+"_"+"_".join(sx)+"_"+dst.name+"_"+"_".join(dx)
        ret=self.joints[name]=Joint(name,LinkRef(src,*sx),LinkRef(dst,*dx),axis,typus)
        self.used.add(repr(ret.parent))
        self.used.add(repr(ret.child))
        return ret

    def mkXml(self):
        ret=['<robot name="'+self.name+'">\n']
        for k,v in self.links.items():
            ret.append(v.mkXml(self.used))
        for k,v in self.joints.items():
            ret.append(v.mkXml())
        ret.append('</robot>\n')
        return "".join(ret)

    def mkArm(self,sa,prefix="left_"):
        r=self
        sb=r.addLink(prefix+'sb',0.02,0.02,0.02)
        r.addJoint(prefix+'s_b',sa,('z','l'),sb,('z','h'),axis=(0,1,0))

        upper_arm=r.addLink(prefix+'upper_arm',0.02,0.02,0.26)
        r.addJoint(prefix+'s_c',sb,('z','l'),upper_arm,('z','h'),axis=(1,0,0))

        lower_arm=r.addLink(prefix+'lower_arm',0.02,0.02,0.24)
        r.addJoint(prefix+'ellbow',upper_arm,('z','l'),lower_arm,('z','h'),axis=(1,0,0))

        wr1=r.addLink(prefix+"w1",0.02,0.02,0.02)
        r.addJoint(prefix+"wra",lower_arm,('z','l'),wr1,('z','h'),axis=(1,0,0))
        wr2=r.addLink(prefix+"w2",0.02,0.02,0.02)
        r.addJoint(prefix+"wrb",wr1,('z','l'),wr2,('z','h'),axis=(0,1,0))

        hand=r.addLink(prefix+"hand",0.08,0.02,0.06)
        r.addJoint(prefix+"wrc",wr2,('z','l'),hand,('z','h'),axis=(0,0,1))

    def mkLeg(self,ha,prefix="left_"):
        r=self
        hb=r.addLink(prefix+'hb',0.02,0.02,0.02)
        r.addJoint(prefix+'h_b',ha,('z','l'),hb,('z','h'),axis=(0,1,0))

        upper_leg=r.addLink(prefix+'upper_leg',0.02,0.02,0.40)
        r.addJoint(prefix+'h_c',hb,('z','l'),upper_leg,('z','h'),axis=(1,0,0))

        lower_leg=r.addLink(prefix+'lower_leg',0.02,0.02,0.30)
        r.addJoint(prefix+'knee',upper_leg,('z','l'),lower_leg,('z','h'),axis=(1,0,0))

        aa=r.addLink(prefix+'aa',0.02,0.02,0.02)
        r.addJoint(prefix+'aa',lower_leg,('z','l'),aa,('z','h'),axis=(1,0,0))

        ab=r.addLink(prefix+'ab',0.02,0.02,0.02)
        r.addJoint(prefix+'ab',aa,('z','l'),ab,('z','h'),axis=(0,1,0))

        foot=r.addLink(prefix+"foot",0.07,0.20,0.02)
        foot.offsets['z_h']=[0,-0.05,0]
        r.addJoint(prefix+"foot",ab,('z','l'),foot,('z','h'),axis=(1,0,0),typus="fixed")
        
                    

if __name__=="__main__":
    r=Robot("test")
    head=r.addLink("head",0.16,0.02,0.14)
    #r.used.add("head_x_l")
    r.used.add("head_z_h")
    un=r.addLink("upper_neck",0.02,0.02,0.02)
    r.addJoint('neck_c',head,('z','l'),un,('z','h'),axis=(0,0,1))
    ln=r.addLink("lower_neck",0.02,0.02,0.02)
    r.addJoint('neck_b',un,('z','l'),ln,('z','h'),axis=(0,1,0))
    cross=r.addLink("cross",0.30,0.02,0.02)
    r.addJoint('neck_a',ln,('z','l'),cross,('z','h'),axis=(1,0,0))

    #arms
    lsa=r.addLink('ls1',0.02,0.02,0.02,up=('x','h'))
    r.addJoint('ls_a',cross,('x','l'),lsa,('x','h'),axis=(0,0,1))
    r.mkArm(lsa,"left_")
    r.used.add("left_hand_x_l")

    rsa=r.addLink('rs1',0.02,0.02,0.02,up=('x','l'))
    r.addJoint('rs_a',cross,('x','h'),rsa,('x','l'),axis=(0,0,1))
    r.mkArm(rsa,"right_")
    r.used.add("right_hand_x_h")

    #back
    upper_back=r.addLink("upper_back",0.02,0.02,0.16)
    r.addJoint("u_back",cross,('z','l'),upper_back,('z','h'),axis=(1,0,0))
    center_back=r.addLink("center_back",0.02,0.02,0.02)
    r.addJoint("cu_back",upper_back,('z','l'),center_back,('z','h'),axis=(0,1,0))
    lower_back=r.addLink("lower_back",0.02,0.02,0.2)
    r.addJoint("cl_back",center_back,('z','l'),lower_back,('z','h'),axis=(1,0,0))

    hip=r.addLink("hip",0.16,0.02,0.02)
    r.addJoint("l_back",lower_back,('z','l'),hip,('z','h'),axis=(0,0,1))

    #legs
    lha=r.addLink('lh1',0.02,0.02,0.02,up=('x','h'))
    r.addJoint('lh_a',hip,('x','l'),lha,('x','h'),axis=(0,0,1))
    r.mkLeg(lha,"left_")
    r.used.add("left_foot_x_l")

    rha=r.addLink('rh1',0.02,0.02,0.02,up=('x','l'))
    r.addJoint('rh_a',hip,('x','h'),rha,('x','l'),axis=(0,0,1))
    r.mkLeg(rha,"right_")
    r.used.add("right_foot_x_h")



    
    print (r.mkXml())
